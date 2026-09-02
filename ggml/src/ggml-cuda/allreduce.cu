#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

static __device__ __forceinline__ bool ggml_cuda_ar_token_reached(int value, int target) {
    return (int32_t) (value - target) >= 0;
}

// Streaming progress word.  The coarse signals above publish one token after a
// block finished its whole stripe.  The streaming kernels instead publish how
// many qblocks of the current call are already visible, so a peer can start
// consuming the prefix while the rest is still being written.  token and step
// count share one naturally aligned 64-bit word: the store crosses PCIe as a
// single transaction, so a reader observes either the previous call's word or
// a complete new one, never a mix.
static __device__ __forceinline__ void ggml_cuda_ar_progress_set(
        unsigned long long * p, int token, int steps) {
    *(volatile unsigned long long *) p =
        ((unsigned long long) (unsigned int) token << 32) | (unsigned int) steps;
}

// Returns the peer's published step count for this token, or 0 when the peer
// has not entered the call yet (its word still carries an older token).
static __device__ __forceinline__ int ggml_cuda_ar_progress_get(
        const unsigned long long * p, int token) {
    const unsigned long long value = *(const volatile unsigned long long *) p;
    return (int) (unsigned int) (value >> 32) == token ? (int) (unsigned int) value : 0;
}

// Byte offset of the progress word inside a 64-byte signal line.  The coarse
// token keeps offset 0, so both protocols can share one arrival array.
static constexpr int GGML_CUDA_AR_PROGRESS_OFFSET_INTS = 2;

static __device__ __forceinline__ uint64_t ggml_cuda_ar_globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// The mixed fused kernels drive the same local arrival ring with
// GGML_CUDA_MIXED_AR_BLOCKS blocks, which is larger than the chunked kernel's
// launch width.  Size the ring for the widest launch so a fused block never
// lands on another (slot, rank)'s arrival line.
static constexpr int GGML_CUDA_AR_ARRIVAL_BLOCKS =
    GGML_CUDA_AR_KERNEL_BLOCKS > (int) GGML_CUDA_MIXED_AR_BLOCKS
        ? GGML_CUDA_AR_KERNEL_BLOCKS
        : (int) GGML_CUDA_MIXED_AR_BLOCKS;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// Same host-mapped strategy as the two-device kernel above, but the staging
// allocation is shared by independent CUDA runtime instances.  Each rank is
// launched by the DSO that owns its device pointers.  Cross-runtime ordering
// is carried exclusively by cache-line-separated tokens in mapped host memory.
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_mixed_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ slot_data,
        int                         rank,
        int                         n_ranks,
        size_t                      rank_stride,
        int                         count,
        int *                       arrival_slot,
        int *                       departure_slot,
        int                         token,
        bool                        contribute) {
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int SIGNAL_INTS = (int) (GGML_CUDA_MIXED_AR_SIGNAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;
    T_wire * host_mine  = slot_data + (size_t) rank * rank_stride;

    if (departure_slot && token > (int) GGML_CUDA_MIXED_AR_SLOTS) {
        if (tid == 0) {
            const int previous_token = token - (int) GGML_CUDA_MIXED_AR_SLOTS;
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const int * peer_departure = departure_slot +
                    ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                while (!ggml_cuda_ar_token_reached(ggml_cuda_ar_signal_get(peer_departure), previous_token)) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
        }
        __syncthreads();
    }

    for (int i = gtid; i < count_vec; i += gnt) {
        const int off = i * ELEMS_PER_VEC;
        T_wire wire[ELEMS_PER_VEC];
#pragma unroll
        for (int k = 0; k < ELEMS_PER_VEC; ++k) {
            wire[k] = contribute ? ggml_cuda_cast<T_wire>(sendbuf[off + k])
                                 : ggml_cuda_cast<T_wire>(0.0f);
        }
        ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
    }
    if (bid == 0 && tid < count - tail) {
        host_mine[tail + tid] = contribute ? ggml_cuda_cast<T_wire>(sendbuf[tail + tid])
                                           : ggml_cuda_cast<T_wire>(0.0f);
    }

    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        int * my_signal = arrival_slot +
            ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
        ggml_cuda_ar_signal_set(my_signal, token);
        __threadfence_system();

        for (int peer = 0; peer < n_ranks; ++peer) {
            if (peer == rank) {
                continue;
            }
            const int * peer_signal = arrival_slot +
                ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            while (ggml_cuda_ar_signal_get(peer_signal) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
    }

    __syncthreads();
    __threadfence_system();

    for (int i = gtid; i < count_vec; i += gnt) {
        const int off = i * ELEMS_PER_VEC;
        float sum[ELEMS_PER_VEC] = {};
        for (int peer = 0; peer < n_ranks; ++peer) {
            T_wire wire[ELEMS_PER_VEC];
            if (peer == rank) {
#pragma unroll
                for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                    wire[k] = contribute ? ggml_cuda_cast<T_wire>(sendbuf[off + k])
                                         : ggml_cuda_cast<T_wire>(0.0f);
                }
            } else {
                const T_wire * host_peer = slot_data + (size_t) peer * rank_stride;
                ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_peer[off]);
            }
#pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                sum[k] += ggml_cuda_cast<float>(wire[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < ELEMS_PER_VEC; ++k) {
            recvbuf[off + k] = ggml_cuda_cast<T_dst>(sum[k]);
        }
    }
    if (bid == 0 && tid < count - tail) {
        float sum = 0.0f;
        for (int peer = 0; peer < n_ranks; ++peer) {
            const T_wire value = peer == rank
                ? (contribute ? ggml_cuda_cast<T_wire>(sendbuf[tail + tid]) : ggml_cuda_cast<T_wire>(0.0f))
                : (slot_data + (size_t) peer * rank_stride)[tail + tid];
            sum += ggml_cuda_cast<float>(value);
        }
        recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(sum);
    }

    if (departure_slot) {
        __syncthreads();
        __threadfence_system();
        if (tid == 0) {
            int * mine = departure_slot +
                ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            ggml_cuda_ar_signal_set(mine, token);
            __threadfence_system();
        }
    }
}

// Hierarchical cross-runtime step for the common 2+1 topology.  Ranks that
// share a CUDA runtime are reduced first by the regular two-GPU pipeline.
// Only the group leader publishes that aggregate to host memory; every local
// rank then adds the other runtime group's aggregate.  For 5080+5070Ti+V100
// this cuts mapped-host traffic from 12 tensor volumes to 5.
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_mixed_ar_hier_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ slot_data,
        int                         rank,
        int                         n_ranks,
        int                         leader_rank,
        int                         peer_leader_rank,
        size_t                      rank_stride,
        int                         count,
        int *                       arrival_slot,
        int *                       departure_slot,
        int                         token,
        bool                        contribute,
        ggml_cuda_mixed_ar_trace_record * trace) {
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int SIGNAL_INTS = (int) (GGML_CUDA_MIXED_AR_SIGNAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;
    const bool is_leader = rank == leader_rank;
    T_wire * host_mine = slot_data + (size_t) leader_rank * rank_stride;
    const T_wire * host_peer = slot_data + (size_t) peer_leader_rank * rank_stride;
    const uint64_t begin_ns = ggml_cuda_ar_globaltimer_ns();

    if (departure_slot && is_leader && token > (int) GGML_CUDA_MIXED_AR_SLOTS) {
        if (tid == 0) {
            const int previous_token = token - (int) GGML_CUDA_MIXED_AR_SLOTS;
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const int * peer_departure = departure_slot +
                    ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                while (!ggml_cuda_ar_token_reached(ggml_cuda_ar_signal_get(peer_departure), previous_token)) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
        }
        __syncthreads();
    }

    if (is_leader) {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
#pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = contribute ? ggml_cuda_cast<T_wire>(sendbuf[off + k])
                                     : ggml_cuda_cast<T_wire>(0.0f);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = contribute
                ? ggml_cuda_cast<T_wire>(sendbuf[tail + tid])
                : ggml_cuda_cast<T_wire>(0.0f);
        }
        __threadfence_system();
    }
    __syncthreads();
    const uint64_t publish_ns = ggml_cuda_ar_globaltimer_ns();

    if (tid == 0) {
        if (is_leader) {
            int * mine = arrival_slot +
                ((size_t) leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            ggml_cuda_ar_signal_set(mine, token);
            __threadfence_system();
        }
        const int * peer = arrival_slot +
            ((size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
        while (ggml_cuda_ar_signal_get(peer) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }

    __syncthreads();
    __threadfence_system();
    const uint64_t peer_ready_ns = ggml_cuda_ar_globaltimer_ns();

    for (int i = gtid; i < count_vec; i += gnt) {
        const int off = i * ELEMS_PER_VEC;
        T_wire peer[ELEMS_PER_VEC];
        ggml_cuda_memcpy_1<sizeof(peer)>(peer, &host_peer[off]);
#pragma unroll
        for (int k = 0; k < ELEMS_PER_VEC; ++k) {
            const T_wire local = contribute
                ? ggml_cuda_cast<T_wire>(sendbuf[off + k])
                : ggml_cuda_cast<T_wire>(0.0f);
            recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(local) + ggml_cuda_cast<float>(peer[k]));
        }
    }
    if (bid == 0 && tid < count - tail) {
        const T_wire local = contribute
            ? ggml_cuda_cast<T_wire>(sendbuf[tail + tid])
            : ggml_cuda_cast<T_wire>(0.0f);
        recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(local) + ggml_cuda_cast<float>(host_peer[tail + tid]));
    }

    if (departure_slot || trace) {
        __syncthreads();
        __threadfence_system();
        if (tid == 0) {
            if (departure_slot) {
                int * mine = departure_slot +
                    ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                ggml_cuda_ar_signal_set(mine, token);
            }
            if (trace) {
                trace[(size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid] = {
                    begin_ns,
                    publish_ns,
                    peer_ready_ns,
                    ggml_cuda_ar_globaltimer_ns(),
                };
            }
            __threadfence_system();
        }
    }
}

// Block-quantized cross-runtime transport.  The local aggregate is rounded
// through symmetric INT8 in 4096-element blocks before it is published.  All
// ranks retain that rounded local value and add the same rounded peer value,
// so the tensor-parallel replicas stay bit-equivalent while mapped-host
// traffic is roughly halved versus BF16.
static __global__ void ggml_cuda_mixed_ar_hier_int8_kernel(
        const float *              sendbuf,
        float       *              recvbuf,
        uint8_t     * __restrict__ slot_data,
        int                        rank,
        int                        n_ranks,
        int                        leader_rank,
        int                        peer_leader_rank,
        int                        count,
        int *                      arrival_slot,
        int *                      departure_slot,
        int                        token,
        bool                       contribute,
        ggml_cuda_mixed_ar_trace_record * trace) {
    constexpr int QK = 4096;
    constexpr int VALUES_PER_THREAD = QK / 256;
    constexpr int SIGNAL_INTS = (int) (GGML_CUDA_MIXED_AR_SIGNAL_STRIDE / sizeof(int));

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_qblocks = (count + QK - 1) / QK;
    const size_t scale_offset = ((size_t) count + 15) & ~(size_t) 15;
    const bool is_leader = rank == leader_rank;
    uint8_t * host_mine_base = slot_data + (size_t) leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    const uint8_t * host_peer_base = slot_data + (size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    int8_t * host_mine = reinterpret_cast<int8_t *>(host_mine_base);
    const int8_t * host_peer = reinterpret_cast<const int8_t *>(host_peer_base);
    float * host_mine_scales = reinterpret_cast<float *>(host_mine_base + scale_offset);
    const float * host_peer_scales = reinterpret_cast<const float *>(host_peer_base + scale_offset);
    __shared__ float warp_max[8];
    __shared__ float block_scale;
    __shared__ float block_inv_scale;
    __shared__ __align__(16) int8_t block_values[QK];
    const uint64_t begin_ns = ggml_cuda_ar_globaltimer_ns();

    if (departure_slot && is_leader && token > (int) GGML_CUDA_MIXED_AR_SLOTS) {
        if (tid == 0) {
            const int previous_token = token - (int) GGML_CUDA_MIXED_AR_SLOTS;
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const int * peer_departure = departure_slot +
                    ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                while (!ggml_cuda_ar_token_reached(ggml_cuda_ar_signal_get(peer_departure), previous_token)) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
        }
        __syncthreads();
    }

    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        const int block_start = qb * QK;
        float max_value = 0.0f;
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            const float value = index < count && contribute ? sendbuf[index] : 0.0f;
            max_value = fmaxf(max_value, fabsf(value));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
        }
        if (lane == 0) {
            warp_max[warp] = max_value;
        }
        __syncthreads();
        if (warp == 0) {
            max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
            }
            if (lane == 0) {
                block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                block_inv_scale = 1.0f / block_scale;
                if (is_leader) {
                    host_mine_scales[qb] = block_scale;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                const float value = contribute ? sendbuf[index] : 0.0f;
                const int quantized = max(-127, min(127, __float2int_rn(value * block_inv_scale)));
                recvbuf[index] = (float) quantized * block_scale;
                block_values[k * blockDim.x + tid] = (int8_t) quantized;
            }
        }
        __syncthreads();
        if (is_leader) {
            const int local_offset = tid * 16;
            const int index = block_start + local_offset;
            if (index + 16 <= count) {
                ggml_cuda_memcpy_1<16>(&host_mine[index], &block_values[local_offset]);
            } else {
                for (int k = 0; k < 16 && index + k < count; ++k) {
                    host_mine[index + k] = block_values[local_offset + k];
                }
            }
        }
        __syncthreads();
    }

    if (is_leader) {
        __threadfence_system();
    }
    __syncthreads();
    const uint64_t publish_ns = ggml_cuda_ar_globaltimer_ns();

    if (tid == 0) {
        if (is_leader) {
            int * mine = arrival_slot +
                ((size_t) leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            ggml_cuda_ar_signal_set(mine, token);
            __threadfence_system();
        }
        const int * peer = arrival_slot +
            ((size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
        while (ggml_cuda_ar_signal_get(peer) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }

    __syncthreads();
    __threadfence_system();
    const uint64_t peer_ready_ns = ggml_cuda_ar_globaltimer_ns();

    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        if (tid == 0) {
            block_scale = host_peer_scales[qb];
        }
        const int block_start = qb * QK;
        const int local_offset = tid * 16;
        const int vector_index = block_start + local_offset;
        if (vector_index + 16 <= count) {
            ggml_cuda_memcpy_1<16>(&block_values[local_offset], &host_peer[vector_index]);
        } else {
            for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                block_values[local_offset + k] = host_peer[vector_index + k];
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                recvbuf[index] += (float) block_values[k * blockDim.x + tid] * block_scale;
            }
        }
        __syncthreads();
    }

    if (departure_slot || trace) {
        __syncthreads();
        __threadfence_system();
        if (tid == 0) {
            if (departure_slot) {
                int * mine = departure_slot +
                    ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                ggml_cuda_ar_signal_set(mine, token);
            }
            if (trace) {
                trace[(size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid] = {
                    begin_ns,
                    publish_ns,
                    peer_ready_ns,
                    ggml_cuda_ar_globaltimer_ns(),
                };
            }
            __threadfence_system();
        }
    }
}

// Two-device INT8 transport for GPUs that share one CUDA runtime but cannot
// access each other's device memory.  Each rank publishes one block-quantized
// contribution through its mapped host buffer and consumes the peer buffer
// with aligned 16-byte loads through shared memory.
static __global__ void ggml_cuda_ar_local_int8_kernel(
        const float *              sendbuf,
        float       *              recvbuf,
        uint8_t     * __restrict__ host_mine_base,
        const uint8_t * __restrict__ host_peer_base,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_peer,
        int                         token,
        bool                        contribute) {
    constexpr int QK = 4096;
    constexpr int VALUES_PER_THREAD = QK / 256;
    constexpr int ARRIVAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_qblocks = (count + QK - 1) / QK;
    const size_t scale_offset = ((size_t) count + 15) & ~(size_t) 15;
    int8_t * host_mine = reinterpret_cast<int8_t *>(host_mine_base);
    const int8_t * host_peer = reinterpret_cast<const int8_t *>(host_peer_base);
    float * host_mine_scales = reinterpret_cast<float *>(host_mine_base + scale_offset);
    const float * host_peer_scales = reinterpret_cast<const float *>(host_peer_base + scale_offset);
    __shared__ float warp_max[8];
    __shared__ float block_scale;
    __shared__ float block_inv_scale;
    __shared__ __align__(16) int8_t block_values[QK];

    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        const int block_start = qb * QK;
        float max_value = 0.0f;
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            const float value = index < count && contribute ? sendbuf[index] : 0.0f;
            max_value = fmaxf(max_value, fabsf(value));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
        }
        if (lane == 0) {
            warp_max[warp] = max_value;
        }
        __syncthreads();
        if (warp == 0) {
            max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
            }
            if (lane == 0) {
                block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                block_inv_scale = 1.0f / block_scale;
                host_mine_scales[qb] = block_scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                const float value = contribute ? sendbuf[index] : 0.0f;
                const int quantized = max(-127, min(127, __float2int_rn(value * block_inv_scale)));
                recvbuf[index] = (float) quantized * block_scale;
                block_values[k * blockDim.x + tid] = (int8_t) quantized;
            }
        }
        __syncthreads();
        const int local_offset = tid * 16;
        const int index = block_start + local_offset;
        if (index + 16 <= count) {
            ggml_cuda_memcpy_1<16>(&host_mine[index], &block_values[local_offset]);
        } else {
            for (int k = 0; k < 16 && index + k < count; ++k) {
                host_mine[index + k] = block_values[local_offset + k];
            }
        }
        __syncthreads();
    }

    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        int * mine = arrival_mine + bid * ARRIVAL_INTS;
        const int * peer = arrival_peer + bid * ARRIVAL_INTS;
        ggml_cuda_ar_signal_set(mine, token);
        __threadfence_system();
        while (ggml_cuda_ar_signal_get(peer) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }

    __syncthreads();
    __threadfence_system();
    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        if (tid == 0) {
            block_scale = host_peer_scales[qb];
        }
        const int block_start = qb * QK;
        const int local_offset = tid * 16;
        const int vector_index = block_start + local_offset;
        if (vector_index + 16 <= count) {
            ggml_cuda_memcpy_1<16>(&block_values[local_offset], &host_peer[vector_index]);
        } else {
            for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                block_values[local_offset + k] = host_peer[vector_index + k];
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                recvbuf[index] += (float) block_values[k * blockDim.x + tid] * block_scale;
            }
        }
        __syncthreads();
    }
}

// Fused 2+1 INT8 reduction.  The two local ranks first exchange their own
// quantized contributions, round the local aggregate through INT8, and let
// only the leader publish that aggregate to the other runtime.  The one-rank
// runtime publishes directly.  Keeping all three phases in one kernel removes
// the full-tensor boundary between local reduction and cross-runtime exchange.
static __global__ void ggml_cuda_mixed_ar_hier_fused_int8_kernel(
        const float *              sendbuf,
        float       *              recvbuf,
        uint8_t     * __restrict__ cross_slot_data,
        uint8_t     * __restrict__ local_mine_base,
        const uint8_t * __restrict__ local_peer_base,
        int                         rank,
        int                         n_ranks,
        int                         leader_rank,
        int                         peer_leader_rank,
        int                         count,
        int *                       cross_arrival,
        int *                       departure_slot,
        int *                       local_arrival_mine,
        int *                       local_arrival_peer,
        int                         token,
        bool                        contribute,
        bool                        has_local_peer,
        ggml_cuda_mixed_ar_trace_record * trace) {
    constexpr int QK = 4096;
    constexpr int VALUES_PER_THREAD = QK / 256;
    constexpr int SIGNAL_INTS = (int) (GGML_CUDA_MIXED_AR_SIGNAL_STRIDE / sizeof(int));
    constexpr int LOCAL_SIGNAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_qblocks = (count + QK - 1) / QK;
    const size_t scale_offset = ((size_t) count + 15) & ~(size_t) 15;
    const bool is_leader = rank == leader_rank;
    uint8_t * cross_mine_base = cross_slot_data + (size_t) leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    const uint8_t * cross_peer_base = cross_slot_data +
        (size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    uint8_t * first_publish_base = has_local_peer ? local_mine_base : cross_mine_base;
    int8_t * first_values = reinterpret_cast<int8_t *>(first_publish_base);
    float * first_scales = reinterpret_cast<float *>(first_publish_base + scale_offset);
    const int8_t * local_peer_values = has_local_peer
        ? reinterpret_cast<const int8_t *>(local_peer_base) : nullptr;
    const float * local_peer_scales = has_local_peer
        ? reinterpret_cast<const float *>(local_peer_base + scale_offset) : nullptr;
    int8_t * cross_mine_values = reinterpret_cast<int8_t *>(cross_mine_base);
    float * cross_mine_scales = reinterpret_cast<float *>(cross_mine_base + scale_offset);
    const int8_t * cross_peer_values = reinterpret_cast<const int8_t *>(cross_peer_base);
    const float * cross_peer_scales = reinterpret_cast<const float *>(cross_peer_base + scale_offset);
    __shared__ float warp_max[8];
    __shared__ float block_scale;
    __shared__ float block_inv_scale;
    __shared__ __align__(16) int8_t block_values[QK];
    const uint64_t begin_ns = ggml_cuda_ar_globaltimer_ns();

    if (departure_slot && token > (int) GGML_CUDA_MIXED_AR_SLOTS) {
        if (tid == 0) {
            const int previous_token = token - (int) GGML_CUDA_MIXED_AR_SLOTS;
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const int * peer_departure = departure_slot +
                    ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                while (!ggml_cuda_ar_token_reached(ggml_cuda_ar_signal_get(peer_departure), previous_token)) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
        }
        __syncthreads();
    }

    // Publish the original local contribution.  On the V100 side this is
    // already the cross-runtime publication; Blackwell ranks use local host
    // staging first.
    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        const int block_start = qb * QK;
        float max_value = 0.0f;
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            const float value = index < count && contribute ? sendbuf[index] : 0.0f;
            max_value = fmaxf(max_value, fabsf(value));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
        }
        if (lane == 0) {
            warp_max[warp] = max_value;
        }
        __syncthreads();
        if (warp == 0) {
            max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
            }
            if (lane == 0) {
                block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                block_inv_scale = 1.0f / block_scale;
                first_scales[qb] = block_scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                const float value = contribute ? sendbuf[index] : 0.0f;
                const int quantized = max(-127, min(127, __float2int_rn(value * block_inv_scale)));
                recvbuf[index] = (float) quantized * block_scale;
                block_values[k * blockDim.x + tid] = (int8_t) quantized;
            }
        }
        __syncthreads();
        const int local_offset = tid * 16;
        const int index = block_start + local_offset;
        if (index + 16 <= count) {
            ggml_cuda_memcpy_1<16>(&first_values[index], &block_values[local_offset]);
        } else {
            for (int k = 0; k < 16 && index + k < count; ++k) {
                first_values[index + k] = block_values[local_offset + k];
            }
        }
        __syncthreads();
    }

    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        if (has_local_peer) {
            ggml_cuda_ar_signal_set(local_arrival_mine + bid * LOCAL_SIGNAL_INTS, token);
        } else {
            int * mine = cross_arrival +
                ((size_t) leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            ggml_cuda_ar_signal_set(mine, token);
        }
        __threadfence_system();
        if (has_local_peer) {
            while (ggml_cuda_ar_signal_get(local_arrival_peer + bid * LOCAL_SIGNAL_INTS) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
    }
    __syncthreads();
    __threadfence_system();

    // Build and publish the local Blackwell aggregate.  Both local ranks do
    // the same rounding so they remain bit-equivalent; only the leader writes
    // the cross-runtime buffer.
    if (has_local_peer) {
        for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
            if (tid == 0) {
                block_scale = local_peer_scales[qb];
            }
            const int block_start = qb * QK;
            const int local_offset = tid * 16;
            const int vector_index = block_start + local_offset;
            if (vector_index + 16 <= count) {
                ggml_cuda_memcpy_1<16>(&block_values[local_offset], &local_peer_values[vector_index]);
            } else {
                for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                    block_values[local_offset + k] = local_peer_values[vector_index + k];
                }
            }
            __syncthreads();

            float aggregate[VALUES_PER_THREAD];
            float max_value = 0.0f;
#pragma unroll
            for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                const int index = block_start + k * blockDim.x + tid;
                aggregate[k] = index < count
                    ? recvbuf[index] + (float) block_values[k * blockDim.x + tid] * block_scale
                    : 0.0f;
                max_value = fmaxf(max_value, fabsf(aggregate[k]));
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
            }
            if (lane == 0) {
                warp_max[warp] = max_value;
            }
            __syncthreads();
            if (warp == 0) {
                max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
                }
                if (lane == 0) {
                    block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                    block_inv_scale = 1.0f / block_scale;
                    if (is_leader) {
                        cross_mine_scales[qb] = block_scale;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                const int index = block_start + k * blockDim.x + tid;
                if (index < count) {
                    const int quantized = max(-127, min(127, __float2int_rn(aggregate[k] * block_inv_scale)));
                    recvbuf[index] = (float) quantized * block_scale;
                    block_values[k * blockDim.x + tid] = (int8_t) quantized;
                }
            }
            __syncthreads();
            if (is_leader) {
                if (vector_index + 16 <= count) {
                    ggml_cuda_memcpy_1<16>(&cross_mine_values[vector_index], &block_values[local_offset]);
                } else {
                    for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                        cross_mine_values[vector_index + k] = block_values[local_offset + k];
                    }
                }
            }
            __syncthreads();
        }
        if (is_leader) {
            __threadfence_system();
        }
        __syncthreads();
        if (tid == 0 && is_leader) {
            int * mine = cross_arrival +
                ((size_t) leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
            ggml_cuda_ar_signal_set(mine, token);
            __threadfence_system();
        }
    }
    const uint64_t publish_ns = ggml_cuda_ar_globaltimer_ns();

    if (tid == 0) {
        const int * peer = cross_arrival +
            ((size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
        while (ggml_cuda_ar_signal_get(peer) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }
    __syncthreads();
    __threadfence_system();
    const uint64_t peer_ready_ns = ggml_cuda_ar_globaltimer_ns();

    for (int qb = bid; qb < n_qblocks; qb += gridDim.x) {
        if (tid == 0) {
            block_scale = cross_peer_scales[qb];
        }
        const int block_start = qb * QK;
        const int local_offset = tid * 16;
        const int vector_index = block_start + local_offset;
        if (vector_index + 16 <= count) {
            ggml_cuda_memcpy_1<16>(&block_values[local_offset], &cross_peer_values[vector_index]);
        } else {
            for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                block_values[local_offset + k] = cross_peer_values[vector_index + k];
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < VALUES_PER_THREAD; ++k) {
            const int index = block_start + k * blockDim.x + tid;
            if (index < count) {
                recvbuf[index] += (float) block_values[k * blockDim.x + tid] * block_scale;
            }
        }
        __syncthreads();
    }

    if (departure_slot || trace) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            if (departure_slot) {
                int * mine = departure_slot +
                    ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                ggml_cuda_ar_signal_set(mine, token);
            }
            if (trace) {
                trace[(size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid] = {
                    begin_ns,
                    publish_ns,
                    peer_ready_ns,
                    ggml_cuda_ar_globaltimer_ns(),
                };
            }
            __threadfence_system();
        }
    }
}

// Streaming variant of the fused 2+1 INT8 reduction.  The kernel above
// publishes a block's whole stripe before it signals, so the other runtime
// cannot start reading until the slower side has written its last byte; the
// profile shows that as publish plus exposed wait.  Here each block signals
// every `chunk` qblocks, so the peer starts consuming the prefix while the
// rest is still being written.  The three phases stay separate, which keeps
// the publication loop's PCIe writes pipelined -- signalling every single
// qblock instead measured slower than the coarse kernel.  Both runtimes launch
// the same grid and walk the same qblock sequence, so block bid on either side
// is always working on the same qblocks.
static __global__ void ggml_cuda_mixed_ar_hier_stream_int8_kernel(
        const float *              sendbuf,
        float       *              recvbuf,
        uint8_t     * __restrict__ cross_slot_data,
        uint8_t     * __restrict__ local_mine_base,
        const uint8_t * __restrict__ local_peer_base,
        int                         rank,
        int                         n_ranks,
        int                         leader_rank,
        int                         peer_leader_rank,
        int                         count,
        int *                       cross_arrival,
        int *                       departure_slot,
        int *                       local_arrival_mine,
        int *                       local_arrival_peer,
        int                         token,
        int                         chunk,
        bool                        contribute,
        bool                        has_local_peer,
        ggml_cuda_mixed_ar_trace_record * trace) {
    constexpr int QK = 4096;
    constexpr int VALUES_PER_THREAD = QK / 256;
    constexpr int SIGNAL_INTS = (int) (GGML_CUDA_MIXED_AR_SIGNAL_STRIDE / sizeof(int));
    constexpr int LOCAL_SIGNAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_qblocks = (count + QK - 1) / QK;
    // Number of qblocks this block owns, and how many progress steps they form.
    const int stripe_len = n_qblocks > bid ? (n_qblocks - bid + (int) gridDim.x - 1) / (int) gridDim.x : 0;
    const int n_steps = (stripe_len + chunk - 1) / chunk;
    const size_t scale_offset = ((size_t) count + 15) & ~(size_t) 15;
    const bool is_leader = rank == leader_rank;
    uint8_t * cross_mine_base = cross_slot_data + (size_t) leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    const uint8_t * cross_peer_base = cross_slot_data +
        (size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_RANK_BYTES;
    uint8_t * first_publish_base = has_local_peer ? local_mine_base : cross_mine_base;
    int8_t * first_values = reinterpret_cast<int8_t *>(first_publish_base);
    float * first_scales = reinterpret_cast<float *>(first_publish_base + scale_offset);
    const int8_t * local_peer_values = has_local_peer
        ? reinterpret_cast<const int8_t *>(local_peer_base) : nullptr;
    const float * local_peer_scales = has_local_peer
        ? reinterpret_cast<const float *>(local_peer_base + scale_offset) : nullptr;
    int8_t * cross_mine_values = reinterpret_cast<int8_t *>(cross_mine_base);
    float * cross_mine_scales = reinterpret_cast<float *>(cross_mine_base + scale_offset);
    const int8_t * cross_peer_values = reinterpret_cast<const int8_t *>(cross_peer_base);
    const float * cross_peer_scales = reinterpret_cast<const float *>(cross_peer_base + scale_offset);

    unsigned long long * cross_progress_mine = reinterpret_cast<unsigned long long *>(
        cross_arrival + ((size_t) leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS +
        GGML_CUDA_AR_PROGRESS_OFFSET_INTS);
    const unsigned long long * cross_progress_peer = reinterpret_cast<const unsigned long long *>(
        cross_arrival + ((size_t) peer_leader_rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS +
        GGML_CUDA_AR_PROGRESS_OFFSET_INTS);
    unsigned long long * local_progress_mine = has_local_peer
        ? reinterpret_cast<unsigned long long *>(
            local_arrival_mine + bid * LOCAL_SIGNAL_INTS + GGML_CUDA_AR_PROGRESS_OFFSET_INTS)
        : nullptr;
    const unsigned long long * local_progress_peer = has_local_peer
        ? reinterpret_cast<const unsigned long long *>(
            local_arrival_peer + bid * LOCAL_SIGNAL_INTS + GGML_CUDA_AR_PROGRESS_OFFSET_INTS)
        : nullptr;

    __shared__ float warp_max[8];
    __shared__ float block_scale;
    __shared__ float block_inv_scale;
    __shared__ __align__(16) int8_t block_values[QK];

    const uint64_t begin_ns = ggml_cuda_ar_globaltimer_ns();
    uint64_t work_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t mark_ns = begin_ns;

#define GGML_CUDA_AR_STREAM_MARK(accumulator)                                  \
    if (trace) {                                                               \
        const uint64_t now_ns = ggml_cuda_ar_globaltimer_ns();                 \
        accumulator += now_ns - mark_ns;                                       \
        mark_ns = now_ns;                                                      \
    }

    if (departure_slot && token > (int) GGML_CUDA_MIXED_AR_SLOTS) {
        if (tid == 0) {
            const int previous_token = token - (int) GGML_CUDA_MIXED_AR_SLOTS;
            for (int peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank) {
                    continue;
                }
                const int * peer_departure = departure_slot +
                    ((size_t) peer * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                while (!ggml_cuda_ar_token_reached(ggml_cuda_ar_signal_get(peer_departure), previous_token)) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
        }
        __syncthreads();
        GGML_CUDA_AR_STREAM_MARK(wait_ns)
    }

    // Phase 1: quantize and publish this block's own contribution, releasing
    // one progress step per chunk.  On the single-rank runtime this is already
    // the cross-runtime publication.
    for (int step = 0; step < n_steps; ++step) {
        const int first = step * chunk;
        const int last = min(first + chunk, stripe_len);
        for (int j = first; j < last; ++j) {
            const int qb = bid + j * (int) gridDim.x;
            const int block_start = qb * QK;
            const int local_offset = tid * 16;
            const int vector_index = block_start + local_offset;
            float max_value = 0.0f;
#pragma unroll
            for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                const int index = block_start + k * blockDim.x + tid;
                const float value = index < count && contribute ? sendbuf[index] : 0.0f;
                max_value = fmaxf(max_value, fabsf(value));
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
            }
            if (lane == 0) {
                warp_max[warp] = max_value;
            }
            __syncthreads();
            if (warp == 0) {
                max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
                }
                if (lane == 0) {
                    block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                    block_inv_scale = 1.0f / block_scale;
                    first_scales[qb] = block_scale;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                const int index = block_start + k * blockDim.x + tid;
                if (index < count) {
                    const float value = contribute ? sendbuf[index] : 0.0f;
                    const int quantized = max(-127, min(127, __float2int_rn(value * block_inv_scale)));
                    recvbuf[index] = (float) quantized * block_scale;
                    block_values[k * blockDim.x + tid] = (int8_t) quantized;
                }
            }
            __syncthreads();
            if (vector_index + 16 <= count) {
                ggml_cuda_memcpy_1<16>(&first_values[vector_index], &block_values[local_offset]);
            } else {
                for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                    first_values[vector_index + k] = block_values[local_offset + k];
                }
            }
            __syncthreads();
        }
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            ggml_cuda_ar_progress_set(
                has_local_peer ? local_progress_mine : cross_progress_mine, token, step + 1);
            __threadfence_system();
        }
    }
    GGML_CUDA_AR_STREAM_MARK(work_ns)

    // Phase 2: build the local aggregate chunk by chunk and let the leader
    // forward each finished chunk to the other runtime right away.
    if (has_local_peer) {
        for (int step = 0; step < n_steps; ++step) {
            if (tid == 0) {
                while (ggml_cuda_ar_progress_get(local_progress_peer, token) < step + 1) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                    __nanosleep(100);
#else
                    NO_DEVICE_CODE;
#endif
                }
            }
            __syncthreads();
            __threadfence_system();
            GGML_CUDA_AR_STREAM_MARK(wait_ns)

            const int first = step * chunk;
            const int last = min(first + chunk, stripe_len);
            for (int j = first; j < last; ++j) {
                const int qb = bid + j * (int) gridDim.x;
                const int block_start = qb * QK;
                const int local_offset = tid * 16;
                const int vector_index = block_start + local_offset;
                if (tid == 0) {
                    block_scale = local_peer_scales[qb];
                }
                if (vector_index + 16 <= count) {
                    ggml_cuda_memcpy_1<16>(&block_values[local_offset], &local_peer_values[vector_index]);
                } else {
                    for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                        block_values[local_offset + k] = local_peer_values[vector_index + k];
                    }
                }
                __syncthreads();

                float aggregate[VALUES_PER_THREAD];
                float max_value = 0.0f;
#pragma unroll
                for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                    const int index = block_start + k * blockDim.x + tid;
                    aggregate[k] = index < count
                        ? recvbuf[index] + (float) block_values[k * blockDim.x + tid] * block_scale
                        : 0.0f;
                    max_value = fmaxf(max_value, fabsf(aggregate[k]));
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
                }
                if (lane == 0) {
                    warp_max[warp] = max_value;
                }
                __syncthreads();
                if (warp == 0) {
                    max_value = lane < 8 ? warp_max[lane] : 0.0f;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffff, max_value, offset));
                    }
                    if (lane == 0) {
                        block_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
                        block_inv_scale = 1.0f / block_scale;
                        if (is_leader) {
                            cross_mine_scales[qb] = block_scale;
                        }
                    }
                }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                    const int index = block_start + k * blockDim.x + tid;
                    if (index < count) {
                        const int quantized = max(-127, min(127, __float2int_rn(aggregate[k] * block_inv_scale)));
                        recvbuf[index] = (float) quantized * block_scale;
                        block_values[k * blockDim.x + tid] = (int8_t) quantized;
                    }
                }
                __syncthreads();
                if (is_leader) {
                    if (vector_index + 16 <= count) {
                        ggml_cuda_memcpy_1<16>(&cross_mine_values[vector_index], &block_values[local_offset]);
                    } else {
                        for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                            cross_mine_values[vector_index + k] = block_values[local_offset + k];
                        }
                    }
                }
                __syncthreads();
            }
            if (is_leader) {
                __threadfence_system();
                __syncthreads();
                if (tid == 0) {
                    ggml_cuda_ar_progress_set(cross_progress_mine, token, step + 1);
                    __threadfence_system();
                }
            }
            GGML_CUDA_AR_STREAM_MARK(work_ns)
        }
    }

    // Phase 3: consume the other runtime's chunks as they become visible.
    for (int step = 0; step < n_steps; ++step) {
        if (tid == 0) {
            while (ggml_cuda_ar_progress_get(cross_progress_peer, token) < step + 1) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
        __syncthreads();
        __threadfence_system();
        GGML_CUDA_AR_STREAM_MARK(wait_ns)

        const int first = step * chunk;
        const int last = min(first + chunk, stripe_len);
        for (int j = first; j < last; ++j) {
            const int qb = bid + j * (int) gridDim.x;
            const int block_start = qb * QK;
            const int local_offset = tid * 16;
            const int vector_index = block_start + local_offset;
            if (tid == 0) {
                block_scale = cross_peer_scales[qb];
            }
            if (vector_index + 16 <= count) {
                ggml_cuda_memcpy_1<16>(&block_values[local_offset], &cross_peer_values[vector_index]);
            } else {
                for (int k = 0; k < 16 && vector_index + k < count; ++k) {
                    block_values[local_offset + k] = cross_peer_values[vector_index + k];
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < VALUES_PER_THREAD; ++k) {
                const int index = block_start + k * blockDim.x + tid;
                if (index < count) {
                    recvbuf[index] += (float) block_values[k * blockDim.x + tid] * block_scale;
                }
            }
            __syncthreads();
        }
        GGML_CUDA_AR_STREAM_MARK(work_ns)
    }

    if (departure_slot || trace) {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            if (departure_slot) {
                int * mine = departure_slot +
                    ((size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid) * SIGNAL_INTS;
                ggml_cuda_ar_signal_set(mine, token);
            }
            if (trace) {
                // The streamed phases interleave with the peer, so the record
                // carries accumulated time per category, not three boundaries.
                const uint64_t done_ns = ggml_cuda_ar_globaltimer_ns();
                const uint64_t split_ns = begin_ns + work_ns + wait_ns;
                trace[(size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + bid] = {
                    begin_ns,
                    begin_ns + work_ns,
                    split_ns,
                    done_ns > split_ns ? done_ns : split_ns,
                };
            }
            __threadfence_system();
        }
    }
#undef GGML_CUDA_AR_STREAM_MARK
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse after both peer completion events from
// AR N-2 have entered the current streams.  This keeps the dependency on the
// device and avoids blocking the CPU submission thread.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    size_t   local_int8_threshold; // F32 tensors at or above this size use mapped-host INT8; 0 disables
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    ggml_cuda_ar_host_mapping host_int8[GGML_CUDA_MAX_DEVICES];  // two-slot mapped INT8 staging
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_ARRIVAL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
    bool pool_lapped;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;
    return { slot, (int) p->call_count, pool_lapped };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    p->local_int8_threshold = ggml_cuda_ar_env_u64("GGML_CUDA_AR_LOCAL_INT8_THRESHOLD", 0);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_ARRIVAL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (p->host_int8[i].alloc((size_t) GGML_CUDA_AR_POOL_SIZE * p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for local INT8 staging failed (%zu bytes)\n",
                           __func__, (size_t) GGML_CUDA_AR_POOL_SIZE * p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20);
    if (p->local_int8_threshold > 0) {
        GGML_LOG_WARN("%s: mapped-host local INT8 enabled at %zu bytes\n",
                      __func__, p->local_int8_threshold);
    }

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        p->host_int8[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
            }
            if (p->ev_pool[i][s].h2d) { cudaEventDestroy(p->ev_pool[i][s].h2d); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->host_large_read_done[i]);
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    p->arrival.free();
    delete p;
}

struct ggml_cuda_mixed_ar_group {
    size_t n_ranks = 0;
    size_t data_bytes = 0;
    size_t arrival_offset = 0;
    size_t departure_offset = 0;
    size_t trace_offset = 0;
    size_t bf16_threshold = 0;
    bool profile = false;
    bool device_slots = false;
    bool hierarchical = false;
    bool wire_int8 = false;
    bool fused_int8 = false;
    bool stream_int8 = false;
    int stream_chunk = 1;
    int leader_rank = -1;
    int peer_leader_rank = -1;
    ggml_cuda_ar_pipeline * local_ar = nullptr;
    void * shared_host = nullptr;
    bool host_registered = false;
    std::vector<ggml_backend_t> backends;
    std::vector<int> ranks;
    std::vector<int> devices;
    std::vector<uint8_t *> device_bases;
    std::vector<cudaEvent_t> done;
    std::vector<bool> done_valid;
    std::vector<cudaEvent_t> profile_start;
    std::vector<cudaEvent_t> profile_local;
    std::vector<cudaEvent_t> profile_done;
};

void ggml_cuda_mixed_ar_group_free(void * context) {
    auto * group = static_cast<ggml_cuda_mixed_ar_group *>(context);
    if (!group) {
        return;
    }

    for (size_t i = 0; i < group->devices.size(); ++i) {
        ggml_cuda_set_device(group->devices[i]);
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(group->backends[i]->context);
        cudaStreamSynchronize(cuda_ctx->stream());
        if (!group->device_slots) {
            for (size_t slot = 0; slot < GGML_CUDA_MIXED_AR_SLOTS; ++slot) {
                cudaEvent_t & event = group->done[i * GGML_CUDA_MIXED_AR_SLOTS + slot];
                if (event) {
                    cudaEventDestroy(event);
                }
            }
        }
        if (i < group->profile_start.size() && group->profile_start[i]) {
            cudaEventDestroy(group->profile_start[i]);
        }
        if (i < group->profile_local.size() && group->profile_local[i]) {
            cudaEventDestroy(group->profile_local[i]);
        }
        if (i < group->profile_done.size() && group->profile_done[i]) {
            cudaEventDestroy(group->profile_done[i]);
        }
    }
    ggml_cuda_ar_pipeline_free(group->local_ar);
    group->local_ar = nullptr;
    if (group->host_registered && !group->devices.empty()) {
        ggml_cuda_set_device(group->devices[0]);
        cudaHostUnregister(group->shared_host);
    }
    delete group;
}

void * ggml_cuda_mixed_ar_group_init(const ggml_cuda_mixed_ar_group_config * config) {
    const size_t token_bytes = config
        ? GGML_CUDA_MIXED_AR_SLOTS * config->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS *
              GGML_CUDA_MIXED_AR_SIGNAL_STRIDE
        : 0;
    if (!config || config->abi_version != GGML_CUDA_MIXED_AR_ABI_VERSION ||
        config->n_backends == 0 || config->n_ranks < 2 ||
        config->n_backends > config->n_ranks || !config->shared_host ||
        config->data_bytes > config->shared_bytes ||
        config->arrival_offset < config->data_bytes ||
        config->arrival_offset + token_bytes > config->shared_bytes ||
        config->departure_offset < config->arrival_offset + token_bytes ||
        config->departure_offset + token_bytes > config->shared_bytes ||
        config->trace_offset < config->departure_offset + token_bytes ||
        (config->profile &&
         (config->trace_offset + config->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS *
              sizeof(ggml_cuda_mixed_ar_trace_record) > config->shared_bytes))) {
        return nullptr;
    }

    auto * group = new ggml_cuda_mixed_ar_group;
    group->n_ranks = config->n_ranks;
    group->data_bytes = config->data_bytes;
    group->arrival_offset = config->arrival_offset;
    group->departure_offset = config->departure_offset;
    group->trace_offset = config->trace_offset;
    group->bf16_threshold = config->bf16_threshold;
    group->profile = config->profile;
    group->device_slots = config->device_slots;
    group->hierarchical = config->hierarchical;
    group->wire_int8 = config->hierarchical && ggml_cuda_ar_env_u64("GGML_CUDA_MIXED_AR_INT8", 0) != 0;
    group->fused_int8 = group->wire_int8 && ggml_cuda_ar_env_u64("GGML_CUDA_MIXED_AR_FUSED_INT8", 0) != 0;
    group->stream_int8 = group->fused_int8 &&
        ggml_cuda_ar_env_u64("GGML_CUDA_MIXED_AR_FUSED_STREAM", 0) != 0;
    // Qblocks published per progress step.  One step per qblock spends more on
    // system fences than it wins back; 20 was the best measured value for the
    // 20 MiB wire at 32 blocks (80 qblocks per stripe, so four steps).  Sweep
    // it with GGML_CUDA_MIXED_AR_STREAM_CHUNK.
    group->stream_chunk = (int) ggml_cuda_ar_env_u64("GGML_CUDA_MIXED_AR_STREAM_CHUNK", 20);
    if (group->stream_chunk < 1) {
        group->stream_chunk = 1;
    }
    group->leader_rank = config->leader_rank;
    group->peer_leader_rank = config->peer_leader_rank;
    group->shared_host = config->shared_host;
    group->backends.assign(config->backends, config->backends + config->n_backends);
    group->ranks.assign(config->ranks, config->ranks + config->n_backends);
    group->devices.reserve(config->n_backends);
    group->device_bases.resize(config->n_backends, nullptr);
    if (!group->device_slots) {
        group->done.resize(config->n_backends * GGML_CUDA_MIXED_AR_SLOTS, nullptr);
        group->done_valid.resize(config->n_backends * GGML_CUDA_MIXED_AR_SLOTS, false);
    }
    if (group->profile) {
        group->profile_start.resize(config->n_backends, nullptr);
        group->profile_local.resize(config->n_backends, nullptr);
        group->profile_done.resize(config->n_backends, nullptr);
    }

    for (size_t i = 0; i < config->n_backends; ++i) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(config->backends[i]->context);
        group->devices.push_back(cuda_ctx->device);
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_WARN("%s: device %d has cc=%d, need Volta or newer\n", __func__, cuda_ctx->device, cc);
            ggml_cuda_mixed_ar_group_free(group);
            return nullptr;
        }
    }

    if (group->wire_int8) {
        GGML_LOG_WARN("%s: block-quantized INT8 cross-runtime wire enabled\n", __func__);
    }
    if (group->fused_int8) {
        GGML_LOG_WARN("%s: fused local and cross-runtime INT8 enabled\n", __func__);
    }
    if (group->stream_int8) {
        GGML_LOG_WARN("%s: chunked streaming publication enabled (%d qblocks per step)\n",
                      __func__, group->stream_chunk);
    }

    if (group->hierarchical && group->devices.size() == 2) {
        group->local_ar = ggml_cuda_ar_pipeline_init(group->devices.data(), group->devices.size());
        if (!group->local_ar) {
            GGML_LOG_WARN("%s: failed to initialize hierarchical local AllReduce\n", __func__);
            ggml_cuda_mixed_ar_group_free(group);
            return nullptr;
        }
    }

    ggml_cuda_set_device(group->devices[0]);
    cudaError_t rc = cudaHostRegister(config->shared_host, config->shared_bytes,
                                      cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (rc != cudaSuccess) {
        GGML_LOG_WARN("%s: cudaHostRegister(%zu) failed: %s\n", __func__, config->shared_bytes,
                      cudaGetErrorString(rc));
        (void) cudaGetLastError();
        ggml_cuda_mixed_ar_group_free(group);
        return nullptr;
    }
    group->host_registered = true;

    for (size_t i = 0; i < config->n_backends; ++i) {
        ggml_cuda_set_device(group->devices[i]);
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&group->device_bases[i]), config->shared_host, 0);
        if (rc != cudaSuccess) {
            GGML_LOG_WARN("%s: cudaHostGetDevicePointer failed on device %d: %s\n",
                          __func__, group->devices[i], cudaGetErrorString(rc));
            (void) cudaGetLastError();
            ggml_cuda_mixed_ar_group_free(group);
            return nullptr;
        }
        if (!group->device_slots) {
            for (size_t slot = 0; slot < GGML_CUDA_MIXED_AR_SLOTS; ++slot) {
                cudaEvent_t & event = group->done[i * GGML_CUDA_MIXED_AR_SLOTS + slot];
                if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
                    GGML_LOG_WARN("%s: cudaEventCreate failed on device %d\n", __func__, group->devices[i]);
                    (void) cudaGetLastError();
                    ggml_cuda_mixed_ar_group_free(group);
                    return nullptr;
                }
            }
        }
        if (group->profile &&
            (cudaEventCreate(&group->profile_start[i]) != cudaSuccess ||
             cudaEventCreate(&group->profile_local[i]) != cudaSuccess ||
             cudaEventCreate(&group->profile_done[i]) != cudaSuccess)) {
            GGML_LOG_WARN("%s: profiling cudaEventCreate failed on device %d\n", __func__, group->devices[i]);
            (void) cudaGetLastError();
            ggml_cuda_mixed_ar_group_free(group);
            return nullptr;
        }
    }

    return group;
}

bool ggml_cuda_mixed_ar_group_prepare(void * context, size_t slot) {
    auto * group = static_cast<ggml_cuda_mixed_ar_group *>(context);
    if (!group || slot >= GGML_CUDA_MIXED_AR_SLOTS) {
        return false;
    }
    if (group->device_slots) {
        return true;
    }
    for (size_t i = 0; i < group->backends.size(); ++i) {
        const size_t index = i * GGML_CUDA_MIXED_AR_SLOTS + slot;
        if (group->done_valid[index]) {
            ggml_cuda_set_device(group->devices[i]);
            if (cudaEventSynchronize(group->done[index]) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
        }
    }
    return true;
}

bool ggml_cuda_mixed_ar_group_enqueue(
        void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16) {
    auto * group = static_cast<ggml_cuda_mixed_ar_group *>(context);
    if (!group || !tensors || slot >= GGML_CUDA_MIXED_AR_SLOTS) {
        return false;
    }

    for (size_t i = 0; i < group->backends.size(); ++i) {
        const int rank = group->ranks[i];
        ggml_tensor * tensor = tensors[rank];
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(group->backends[i]->context);
        const int64_t ne = ggml_nelements(tensor);
        const bool contribute = (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        const ggml_type wire_type = use_bf16 ? GGML_TYPE_BF16 : tensor->type;
        const size_t rank_stride = GGML_CUDA_MIXED_AR_RANK_BYTES / ggml_type_size(wire_type);

        ggml_cuda_set_device(group->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();
        uint8_t * base = group->device_bases[i];
        void * slot_data = base + slot * group->n_ranks * GGML_CUDA_MIXED_AR_RANK_BYTES;
        int * arrival_slot = reinterpret_cast<int *>(base + group->arrival_offset +
            slot * group->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS * GGML_CUDA_MIXED_AR_SIGNAL_STRIDE);
        int * departure_slot = group->device_slots
            ? reinterpret_cast<int *>(base + group->departure_offset +
                slot * group->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS * GGML_CUDA_MIXED_AR_SIGNAL_STRIDE)
            : nullptr;

#define LAUNCH_MIXED_AR(T_dst, T_wire) \
        ggml_cuda_mixed_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_MIXED_AR_BLOCKS), dim3(256), 0, stream>>>( \
            reinterpret_cast<const T_dst *>(tensor->data), \
            reinterpret_cast<T_dst *>(tensor->data), \
            reinterpret_cast<T_wire *>(slot_data), rank, (int) group->n_ranks, rank_stride, \
            (int) ne, arrival_slot, departure_slot, token, contribute)

        if (use_bf16) {
            LAUNCH_MIXED_AR(float, nv_bfloat16);
        } else {
            switch (tensor->type) {
                case GGML_TYPE_F32:  LAUNCH_MIXED_AR(float,       float);       break;
                case GGML_TYPE_F16:  LAUNCH_MIXED_AR(half,        half);        break;
                case GGML_TYPE_BF16: LAUNCH_MIXED_AR(nv_bfloat16, nv_bfloat16); break;
                default: return false;
            }
        }
#undef LAUNCH_MIXED_AR

        CUDA_CHECK(cudaGetLastError());
        if (!group->device_slots) {
            const size_t event_index = i * GGML_CUDA_MIXED_AR_SLOTS + slot;
            CUDA_CHECK(cudaEventRecord(group->done[event_index], stream));
            group->done_valid[event_index] = true;
        }
    }
    return true;
}

bool ggml_cuda_mixed_ar_group_enqueue_hier(
        void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16) {
    auto * group = static_cast<ggml_cuda_mixed_ar_group *>(context);
    if (!group || !group->hierarchical || !tensors ||
        slot >= GGML_CUDA_MIXED_AR_SLOTS || group->peer_leader_rank < 0) {
        return false;
    }

    if (group->profile) {
        for (size_t i = 0; i < group->backends.size(); ++i) {
            ggml_cuda_set_device(group->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(group->backends[i]->context);
            CUDA_CHECK(cudaEventRecord(group->profile_start[i], cuda_ctx->stream()));
        }
    }

    const bool fused_int8 = group->fused_int8 && use_bf16;
    ggml_tensor * local_tensors[GGML_CUDA_MAX_DEVICES] = {};
    for (size_t i = 0; i < group->backends.size(); ++i) {
        local_tensors[i] = tensors[group->ranks[i]];
    }
    if (!fused_int8 && group->local_ar && !ggml_cuda_ar_allreduce(
            group->local_ar, group->backends.data(), local_tensors)) {
        return false;
    }

    if (group->profile) {
        for (size_t i = 0; i < group->backends.size(); ++i) {
            ggml_cuda_set_device(group->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(group->backends[i]->context);
            CUDA_CHECK(cudaEventRecord(group->profile_local[i], cuda_ctx->stream()));
        }
    }

    for (size_t i = 0; i < group->backends.size(); ++i) {
        const int rank = group->ranks[i];
        ggml_tensor * tensor = tensors[rank];
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(group->backends[i]->context);
        const int64_t ne = ggml_nelements(tensor);
        if (ne <= 0 || ne > std::numeric_limits<int>::max()) {
            return false;
        }
        // local_ar has already folded all local flags into an aggregate.  A
        // one-rank runtime still needs to honour its original compute flag.
        const bool contribute = fused_int8
            ? (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) != 0
            : group->backends.size() > 1 || (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        const ggml_type wire_type = use_bf16 ? GGML_TYPE_BF16 : tensor->type;
        const size_t rank_stride = GGML_CUDA_MIXED_AR_RANK_BYTES / ggml_type_size(wire_type);

        ggml_cuda_set_device(group->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();
        uint8_t * base = group->device_bases[i];
        void * slot_data = base + slot * group->n_ranks * GGML_CUDA_MIXED_AR_RANK_BYTES;
        int * arrival_slot = reinterpret_cast<int *>(base + group->arrival_offset +
            slot * group->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS * GGML_CUDA_MIXED_AR_SIGNAL_STRIDE);
        int * departure_slot = group->device_slots
            ? reinterpret_cast<int *>(base + group->departure_offset +
                slot * group->n_ranks * GGML_CUDA_MIXED_AR_BLOCKS * GGML_CUDA_MIXED_AR_SIGNAL_STRIDE)
            : nullptr;
        auto * trace = group->profile
            ? reinterpret_cast<ggml_cuda_mixed_ar_trace_record *>(base + group->trace_offset)
            : nullptr;

#define LAUNCH_MIXED_AR_HIER(T_dst, T_wire) \
        ggml_cuda_mixed_ar_hier_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_MIXED_AR_BLOCKS), dim3(256), 0, stream>>>( \
            reinterpret_cast<const T_dst *>(tensor->data), \
            reinterpret_cast<T_dst *>(tensor->data), \
            reinterpret_cast<T_wire *>(slot_data), rank, (int) group->n_ranks, \
            group->leader_rank, group->peer_leader_rank, rank_stride, (int) ne, \
            arrival_slot, departure_slot, token, contribute, trace)

        if (fused_int8 && tensor->type == GGML_TYPE_F32) {
            const bool has_local_peer = group->backends.size() > 1;
            uint8_t * local_mine = nullptr;
            const uint8_t * local_peer = nullptr;
            int * local_arrival_mine = nullptr;
            int * local_arrival_peer = nullptr;
            if (has_local_peer) {
                GGML_ASSERT(group->local_ar && group->backends.size() == 2);
                const int peer = 1 - (int) i;
                local_mine = group->local_ar->host_int8[i].dev + slot * group->local_ar->copy_bytes;
                local_peer = group->local_ar->host_int8[peer].dev + slot * group->local_ar->copy_bytes;
                local_arrival_mine = ggml_cuda_ar_arrival_ptr(group->local_ar, (int) slot, (int) i);
                local_arrival_peer = ggml_cuda_ar_arrival_ptr(group->local_ar, (int) slot, peer);
            }
            if (group->stream_int8) {
                ggml_cuda_mixed_ar_hier_stream_int8_kernel<<<
                    dim3(GGML_CUDA_MIXED_AR_BLOCKS), dim3(256), 0, stream>>>(
                    reinterpret_cast<const float *>(tensor->data),
                    reinterpret_cast<float *>(tensor->data),
                    reinterpret_cast<uint8_t *>(slot_data), local_mine, local_peer,
                    rank, (int) group->n_ranks, group->leader_rank, group->peer_leader_rank,
                    (int) ne, arrival_slot, departure_slot, local_arrival_mine,
                    local_arrival_peer, token, group->stream_chunk, contribute,
                    has_local_peer, trace);
            } else {
                ggml_cuda_mixed_ar_hier_fused_int8_kernel<<<
                    dim3(GGML_CUDA_MIXED_AR_BLOCKS), dim3(256), 0, stream>>>(
                    reinterpret_cast<const float *>(tensor->data),
                    reinterpret_cast<float *>(tensor->data),
                    reinterpret_cast<uint8_t *>(slot_data), local_mine, local_peer,
                    rank, (int) group->n_ranks, group->leader_rank, group->peer_leader_rank,
                    (int) ne, arrival_slot, departure_slot, local_arrival_mine,
                    local_arrival_peer, token, contribute, has_local_peer, trace);
            }
        } else if (use_bf16 && group->wire_int8 && tensor->type == GGML_TYPE_F32) {
            ggml_cuda_mixed_ar_hier_int8_kernel<<<dim3(GGML_CUDA_MIXED_AR_BLOCKS), dim3(256), 0, stream>>>(
                reinterpret_cast<const float *>(tensor->data),
                reinterpret_cast<float *>(tensor->data),
                reinterpret_cast<uint8_t *>(slot_data), rank, (int) group->n_ranks,
                group->leader_rank, group->peer_leader_rank, (int) ne,
                arrival_slot, departure_slot, token, contribute, trace);
        } else if (use_bf16) {
            LAUNCH_MIXED_AR_HIER(float, nv_bfloat16);
        } else {
            switch (tensor->type) {
                case GGML_TYPE_F32:  LAUNCH_MIXED_AR_HIER(float,       float);       break;
                case GGML_TYPE_F16:  LAUNCH_MIXED_AR_HIER(half,        half);        break;
                case GGML_TYPE_BF16: LAUNCH_MIXED_AR_HIER(nv_bfloat16, nv_bfloat16); break;
                default: return false;
            }
        }
#undef LAUNCH_MIXED_AR_HIER

        CUDA_CHECK(cudaGetLastError());
        if (!group->device_slots) {
            const size_t event_index = i * GGML_CUDA_MIXED_AR_SLOTS + slot;
            CUDA_CHECK(cudaEventRecord(group->done[event_index], stream));
            group->done_valid[event_index] = true;
        }
        if (group->profile) {
            CUDA_CHECK(cudaEventRecord(group->profile_done[i], stream));
        }
    }
    return true;
}

bool ggml_cuda_mixed_ar_group_profile_collect(
        void * context, ggml_cuda_mixed_ar_group_profile * profile) {
    auto * group = static_cast<ggml_cuda_mixed_ar_group *>(context);
    if (!group || !profile || !group->profile ||
        group->backends.size() > GGML_CUDA_MAX_DEVICES) {
        return false;
    }

    profile->n_entries = 0;
    auto * records = reinterpret_cast<const ggml_cuda_mixed_ar_trace_record *>(
        static_cast<const uint8_t *>(group->shared_host) + group->trace_offset);

    for (size_t i = 0; i < group->backends.size(); ++i) {
        ggml_cuda_set_device(group->devices[i]);
        if (cudaEventSynchronize(group->profile_done[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }

        float local_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&local_ms, group->profile_start[i], group->profile_local[i]));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, group->profile_start[i], group->profile_done[i]));

        const int rank = group->ranks[i];
        const ggml_cuda_mixed_ar_trace_record * critical = nullptr;
        uint64_t critical_ns = 0;
        for (size_t block = 0; block < GGML_CUDA_MIXED_AR_BLOCKS; ++block) {
            const auto & record = records[(size_t) rank * GGML_CUDA_MIXED_AR_BLOCKS + block];
            const uint64_t duration_ns = record.done_ns - record.begin_ns;
            if (!critical || duration_ns > critical_ns) {
                critical = &record;
                critical_ns = duration_ns;
            }
        }
        if (!critical || critical->publish_ns < critical->begin_ns ||
            critical->peer_ready_ns < critical->publish_ns ||
            critical->done_ns < critical->peer_ready_ns) {
            return false;
        }

        auto & entry = profile->entries[profile->n_entries++];
        entry.rank = rank;
        entry.local_ms = local_ms;
        entry.total_ms = total_ms;
        entry.publish_ms = (critical->publish_ns - critical->begin_ns) / 1000000.0f;
        entry.wait_ms = (critical->peer_ready_ns - critical->publish_ns) / 1000000.0f;
        entry.reduce_ms = (critical->done_ns - critical->peer_ready_ns) / 1000000.0f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

static bool ggml_cuda_ar_allreduce_local_int8_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        float * const           buffers[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(ne > 0 && ne <= std::numeric_limits<int>::max());
    const size_t scale_offset = ((size_t) ne + 15) & ~(size_t) 15;
    const size_t scale_bytes = ((size_t) ne + 4095) / 4096 * sizeof(float);
    GGML_ASSERT(scale_offset + scale_bytes <= p->copy_bytes);

    const auto [slot, token, pool_lapped] = ggml_cuda_ar_acquire_slot(p);

    // Queue both N-2 slot dependencies before either completion event is
    // re-recorded for N.  Doing this in the launch loop would let the first
    // rank's new record change what the second rank waits on and form a cycle.
    if (pool_lapped) {
        for (int i = 0; i < 2; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), p->ev_pool[peer][slot].ker));
        }
    }

    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[i]);
        cudaStream_t stream = cuda_ctx->stream();

        uint8_t * mine = p->host_int8[i].dev + (size_t) slot * p->copy_bytes;
        const uint8_t * other = p->host_int8[peer].dev + (size_t) slot * p->copy_bytes;
        ggml_cuda_ar_local_int8_kernel<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
            buffers[i], buffers[i], mine, other,
            (int) ne, ggml_cuda_ar_arrival_ptr(p, slot, i),
            ggml_cuda_ar_arrival_ptr(p, slot, peer), token, compute[i]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
    }
    return true;
}

static bool ggml_cuda_ar_allreduce_local_int8(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        float * const           buffers[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    // Keep room for one FP32 scale per 4096 INT8 values.  Half the staging
    // byte capacity is a conservative element cap and matches the existing
    // BF16 copy-engine outer chunk size.
    const int64_t max_elems = (int64_t) p->copy_bytes / 2;
    bool ok = true;
    for (int64_t start = 0; start < ne && ok; start += max_elems) {
        const int64_t count = std::min(max_elems, ne - start);
        float * chunk[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            chunk[i] = buffers[i] + start;
        }
        ok = ggml_cuda_ar_allreduce_local_int8_impl(p, backends, chunk, compute, count);
    }
    return ok;
}

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // Record AR-done as ev.ker for the next device-side pool-wraparound wait.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;
    const bool use_local_int8 =
        use_copy_engine && use_bf16 &&
        p->local_int8_threshold > 0 &&
        input_nbytes >= p->local_int8_threshold;

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16 && !use_local_int8) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_local_int8) {
        float * buffers[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < n; ++i) {
            buffers[i] = static_cast<float *>(tensors[i]->data);
        }
        ok = ggml_cuda_ar_allreduce_local_int8(
            p, backends, buffers, compute_flag, ne);
    } else if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for the next pool-wraparound stream wait.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token, pool_lapped] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            if (pool_lapped) {
                for (int i = 0; i < n; ++i) {
                    const int peer = 1 - i;
                    ggml_cuda_set_device(p->devices[i]);
                    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), p->ev_pool[peer][slot].ker));
                }
            }

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// HIP and MUSA lack the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void * ggml_cuda_mixed_ar_group_init(const ggml_cuda_mixed_ar_group_config *) {
    return nullptr;
}
void ggml_cuda_mixed_ar_group_free(void *) {
}
bool ggml_cuda_mixed_ar_group_prepare(void *, size_t) {
    return false;
}
bool ggml_cuda_mixed_ar_group_enqueue(void *, ggml_tensor **, size_t, int, bool) {
    return false;
}
bool ggml_cuda_mixed_ar_group_enqueue_hier(void *, ggml_tensor **, size_t, int, bool) {
    return false;
}
bool ggml_cuda_mixed_ar_group_profile_collect(void *, ggml_cuda_mixed_ar_group_profile *) {
    return false;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
