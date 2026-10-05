#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

// Small-k top-k for long rows (the samplers' top-k over the vocabulary): two passes of
// "per-thread top-k list, then k block-wide rounds picking the best head". Ordered by
// value descending, lower index first on ties - the same result as the stable descending
// argsort it replaces, without sorting 248k keys to keep 10.
#define TOPK_SMALL_MAX 16

static __device__ __forceinline__ bool topk_better(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

// one block per (row, chunk); src_idx == nullptr means the index is the column
template <int block_size>
static __global__ void k_top_k_small(const float * __restrict__ src, const int * __restrict__ src_idx,
        float * __restrict__ dst_val, int * __restrict__ dst_idx,
        const int ncols, const int k, const int chunk) {
    const int row = blockIdx.y;
    const int c0  = blockIdx.x * chunk;
    const int c1  = min(ncols, c0 + chunk);
    const float * s  = src + (int64_t) row * ncols;
    const int *   si = src_idx ? src_idx + (int64_t) row * ncols : nullptr;

    float lv[TOPK_SMALL_MAX];
    int   li[TOPK_SMALL_MAX];
    int   n = 0;
    for (int c = c0 + threadIdx.x; c < c1; c += block_size) {
        const float v  = s[c];
        const int   iv = si ? si[c] : c;
        if (n == k && !topk_better(v, iv, lv[k - 1], li[k - 1])) {
            continue;
        }
        int j = n < k ? n++ : k - 1;
        while (j > 0 && topk_better(v, iv, lv[j - 1], li[j - 1])) {
            lv[j] = lv[j - 1];
            li[j] = li[j - 1];
            j--;
        }
        lv[j] = v;
        li[j] = iv;
    }

    __shared__ float sv[block_size / 32];
    __shared__ int   si_[block_size / 32];
    __shared__ int   st[block_size / 32];
    int head = 0;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int out  = (row * gridDim.x + blockIdx.x) * k;

    for (int r = 0; r < k; ++r) {
        float bv = head < n ? lv[head] : -INFINITY;
        int   bi = head < n ? li[head] : INT_MAX;
        int   bt = head < n ? (int) threadIdx.x : -1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, off);
            const int   ot = __shfl_xor_sync(0xffffffff, bt, off);
            if (ot >= 0 && (bt < 0 || topk_better(ov, oi, bv, bi))) {
                bv = ov; bi = oi; bt = ot;
            }
        }
        if (lane == 0) {
            sv[warp] = bv; si_[warp] = bi; st[warp] = bt;
        }
        __syncthreads();
        if (warp == 0) {
            bv = lane < block_size / 32 ? sv[lane] : -INFINITY;
            bi = lane < block_size / 32 ? si_[lane] : INT_MAX;
            bt = lane < block_size / 32 ? st[lane] : -1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, off);
                const int   oi = __shfl_xor_sync(0xffffffff, bi, off);
                const int   ot = __shfl_xor_sync(0xffffffff, bt, off);
                if (ot >= 0 && (bt < 0 || topk_better(ov, oi, bv, bi))) {
                    bv = ov; bi = oi; bt = ot;
                }
            }
            if (lane == 0) {
                st[0] = bt;
                dst_val[out + r] = bv;
                dst_idx[out + r] = bt >= 0 ? bi : -1;
            }
        }
        __syncthreads();
        if ((int) threadIdx.x == st[0]) {
            head++;
        }
        __syncthreads();
    }
}

static void top_k_small_cuda(ggml_cuda_pool & pool, const float * src, int * dst,
        const int ncols, const int nrows, const int k, cudaStream_t stream) {
    constexpr int bs = 256;
    const int nchunks = std::min(32, (ncols + 4095) / 4096);
    const int chunk   = (ncols + nchunks - 1) / nchunks;
    ggml_cuda_pool_alloc<float> tv(pool, (size_t) nrows * nchunks * k);
    ggml_cuda_pool_alloc<int>   ti(pool, (size_t) nrows * nchunks * k);
    k_top_k_small<bs><<<dim3(nchunks, nrows), bs, 0, stream>>>(src, nullptr, tv.get(), ti.get(), ncols, k, chunk);
    ggml_cuda_pool_alloc<float> ov(pool, (size_t) nrows * k);
    k_top_k_small<bs><<<dim3(1, nrows), bs, 0, stream>>>(tv.get(), ti.get(), ov.get(), dst, nchunks * k, k, nchunks * k);
}

// Large-k top-k (the QSA indexer keeps ~2k of up to 160k cells per token): radix-select the k-th largest key,
// take every larger element plus the lowest-index ties, then bitonic-sort those k. Keys are the float bits
// twiddled as cub does, and the result is the first k of the stable descending sort it replaces.
#define TOPK_RADIX_MAX 4096

static __device__ __forceinline__ uint32_t topk_key(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// sorted = false: the k indices in any order (op_params[0] == 1, set by callers that only use the set)
template <int block_size, bool sorted>
static __global__ void k_top_k_radix(const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    const int row = blockIdx.x;
    const float * s = src + (int64_t) row * ncols;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t sh_prefix, sh_mask;
    __shared__ int      sh_kk;
    __shared__ uint32_t skey[TOPK_RADIX_MAX];
    __shared__ int      sidx[TOPK_RADIX_MAX];
    __shared__ int      scan[block_size];
    __shared__ int      n_gt_pos;

    if (threadIdx.x == 0) {
        sh_prefix = 0;
        sh_mask   = 0;
        sh_kk     = k;
    }
    __syncthreads();

    // radix select, most significant byte first
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        for (int i = threadIdx.x; i < 256; i += block_size) {
            hist[i] = 0;
        }
        __syncthreads();
        const uint32_t prefix = sh_prefix, mask = sh_mask;
        // uniform trip count per warp: the ballot/match below need every lane
        for (int base = 0; base < ncols; base += block_size) {
            const int      c   = base + threadIdx.x;
            const uint32_t key = c < ncols ? topk_key(s[c]) : 0;
            const bool     in  = c < ncols && (key & mask) == prefix;
            const uint32_t d   = (key >> shift) & 0xFF;
            const uint32_t act = __ballot_sync(0xffffffff, in);
            if (in) {
                // one atomic per distinct digit in the warp: ties would otherwise serialize
                const uint32_t peers  = __match_any_sync(act, d);
                const int      leader = __ffs(peers) - 1;
                if ((int) (threadIdx.x % 32) == leader) {
                    atomicAdd(&hist[d], __popc(peers));
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int cum = 0;
            const int kk = sh_kk;
            for (int d = 255; d >= 0; --d) {
                if (cum + (int) hist[d] >= kk) {
                    sh_kk     = kk - cum;
                    sh_prefix = prefix | ((uint32_t) d << shift);
                    sh_mask   = mask | (0xFFu << shift);
                    break;
                }
                cum += hist[d];
            }
        }
        __syncthreads();
    }

    const uint32_t T  = sh_prefix; // key of the k-th largest element
    const int      kk = sh_kk;     // how many elements equal to T are taken (the lowest indices)

    // contiguous chunk per thread so that ties are ranked by index
    const int chunk = (ncols + block_size - 1) / block_size;
    const int c0 = min(ncols, (int) threadIdx.x * chunk);
    const int c1 = min(ncols, c0 + chunk);
    int n_eq = 0;
    for (int c = c0; c < c1; ++c) {
        n_eq += topk_key(s[c]) == T;
    }
    scan[threadIdx.x] = n_eq;
    if (threadIdx.x == 0) {
        n_gt_pos = 0;
    }
    __syncthreads();
    // exclusive scan of n_eq (simple Hillis-Steele over the block)
    for (int off = 1; off < block_size; off <<= 1) {
        const int v = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    int eq_rank = scan[threadIdx.x] - n_eq;
    const int n_gt = k - kk;
    for (int c = c0; c < c1; ++c) {
        const uint32_t key = topk_key(s[c]);
        if (key > T) {
            const int pos = atomicAdd(&n_gt_pos, 1);
            skey[pos] = key;
            sidx[pos] = c;
        } else if (key == T) {
            if (eq_rank < kk) {
                skey[n_gt + eq_rank] = key;
                sidx[n_gt + eq_rank] = c;
            }
            eq_rank++;
        }
    }
    if constexpr (!sorted) {
        __syncthreads();
        for (int i = threadIdx.x; i < k; i += block_size) {
            dst[(int64_t) row * k + i] = sidx[i];
        }
        return;
    }
    int P = 1;
    while (P < k) {
        P <<= 1;
    }
    for (int i = k + threadIdx.x; i < P; i += block_size) {
        skey[i] = 0;
        sidx[i] = INT_MAX;
    }
    __syncthreads();

    // bitonic sort, value descending then index ascending
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < P; i += block_size) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const uint32_t ki = skey[i], kj = skey[j];
                    const int      ii = sidx[i], ij = sidx[j];
                    // i before j in the final order?
                    const bool i_first = ki > kj || (ki == kj && ii < ij);
                    if (i_first != desc) {
                        skey[i] = kj; skey[j] = ki;
                        sidx[i] = ij; sidx[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < k; i += block_size) {
        dst[(int64_t) row * k + i] = sidx[i];
    }
}

// Same selection spread over many blocks per row, for the few long rows of a decode step (one block per row
// would leave the device idle). Row state lives in global memory between the passes.
struct topk_mb_state { uint32_t prefix; uint32_t mask; int kk; int n_gt_pos; };

static __global__ void k_topk_mb_init(topk_mb_state * st, uint32_t * ghist, const int nrows, const int k) {
    const int row = blockIdx.x;
    if (threadIdx.x == 0) {
        st[row] = { 0u, 0u, k, 0 };
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        ghist[row*256 + i] = 0;
    }
    GGML_UNUSED(nrows);
}

template <int block_size>
static __global__ void k_topk_mb_hist(const float * __restrict__ src, const topk_mb_state * __restrict__ st,
        uint32_t * __restrict__ ghist, const int ncols, const int shift) {
    const int row = blockIdx.y;
    const float * s = src + (int64_t) row * ncols;
    __shared__ uint32_t hist[256];
    for (int i = threadIdx.x; i < 256; i += block_size) {
        hist[i] = 0;
    }
    __syncthreads();
    const uint32_t prefix = st[row].prefix, mask = st[row].mask;
    for (int base = blockIdx.x * block_size; base < ncols; base += gridDim.x * block_size) {
        const int      c   = base + threadIdx.x;
        const uint32_t key = c < ncols ? topk_key(s[c]) : 0;
        const bool     in  = c < ncols && (key & mask) == prefix;
        const uint32_t d   = (key >> shift) & 0xFF;
        const uint32_t act = __ballot_sync(0xffffffff, in);
        if (in) {
            const uint32_t peers  = __match_any_sync(act, d);
            const int      leader = __ffs(peers) - 1;
            if ((int) (threadIdx.x % 32) == leader) {
                atomicAdd(&hist[d], __popc(peers));
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += block_size) {
        if (hist[i]) {
            atomicAdd(&ghist[row*256 + i], hist[i]);
        }
    }
}

static __global__ void k_topk_mb_pick(topk_mb_state * st, uint32_t * ghist, const int shift) {
    const int row = blockIdx.x;
    uint32_t * h = ghist + row*256;
    if (threadIdx.x == 0) {
        int cum = 0;
        const int kk = st[row].kk;
        for (int d = 255; d >= 0; --d) {
            if (cum + (int) h[d] >= kk) {
                st[row].kk     = kk - cum;
                st[row].prefix = st[row].prefix | ((uint32_t) d << shift);
                st[row].mask   = st[row].mask | (0xFFu << shift);
                break;
            }
            cum += h[d];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        h[i] = 0;
    }
}

// per block: how many elements of its contiguous chunk equal the threshold
template <int block_size>
static __global__ void k_topk_mb_count(const float * __restrict__ src, const topk_mb_state * __restrict__ st,
        int * __restrict__ cnt, const int ncols, const int chunk) {
    const int row = blockIdx.y;
    const float * s = src + (int64_t) row * ncols;
    const uint32_t T = st[row].prefix;
    const int c0 = min(ncols, (int) blockIdx.x * chunk);
    const int c1 = min(ncols, c0 + chunk);
    int n = 0;
    for (int c = c0 + threadIdx.x; c < c1; c += block_size) {
        n += topk_key(s[c]) == T;
    }
    __shared__ int red[block_size];
    red[threadIdx.x] = n;
    __syncthreads();
    for (int off = block_size/2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        cnt[row*gridDim.x + blockIdx.x] = red[0];
    }
}

// candidates: all keys above the threshold (any order), then the ties in index order
template <int block_size>
static __global__ void k_topk_mb_emit(const float * __restrict__ src, topk_mb_state * __restrict__ st,
        const int * __restrict__ cnt, uint32_t * __restrict__ ckey, int * __restrict__ cidx,
        const int ncols, const int chunk, const int k) {
    const int row = blockIdx.y;
    const float * s = src + (int64_t) row * ncols;
    const uint32_t T  = st[row].prefix;
    const int      kk = st[row].kk;
    const int      n_gt = k - kk;
    int before = 0;
    for (int b = 0; b < (int) blockIdx.x; ++b) {
        before += cnt[row*gridDim.x + b];
    }
    const int b0 = min(ncols, (int) blockIdx.x * chunk);
    const int b1 = min(ncols, b0 + chunk);
    const int sub = (b1 - b0 + block_size - 1) / block_size;
    const int c0 = min(b1, b0 + (int) threadIdx.x * sub);
    const int c1 = min(b1, c0 + sub);
    int n_eq = 0;
    for (int c = c0; c < c1; ++c) {
        n_eq += topk_key(s[c]) == T;
    }
    __shared__ int scan[block_size];
    scan[threadIdx.x] = n_eq;
    __syncthreads();
    for (int off = 1; off < block_size; off <<= 1) {
        const int v = (int) threadIdx.x >= off ? scan[threadIdx.x - off] : 0;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    int rank = before + scan[threadIdx.x] - n_eq;
    uint32_t * rk = ckey + (int64_t) row * k;
    int      * ri = cidx + (int64_t) row * k;
    for (int c = c0; c < c1; ++c) {
        const uint32_t key = topk_key(s[c]);
        if (key > T) {
            const int pos = atomicAdd(&st[row].n_gt_pos, 1);
            rk[pos] = key;
            ri[pos] = c;
        } else if (key == T) {
            if (rank < kk) {
                rk[n_gt + rank] = key;
                ri[n_gt + rank] = c;
            }
            rank++;
        }
    }
}

template <int block_size>
static __global__ void k_topk_mb_sort(const uint32_t * __restrict__ ckey, const int * __restrict__ cidx,
        int * __restrict__ dst, const int k) {
    const int row = blockIdx.x;
    __shared__ uint32_t skey[TOPK_RADIX_MAX];
    __shared__ int      sidx[TOPK_RADIX_MAX];
    int P = 1;
    while (P < k) {
        P <<= 1;
    }
    for (int i = threadIdx.x; i < P; i += block_size) {
        skey[i] = i < k ? ckey[(int64_t) row*k + i] : 0;
        sidx[i] = i < k ? cidx[(int64_t) row*k + i] : INT_MAX;
    }
    __syncthreads();
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < P; i += block_size) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const uint32_t ki = skey[i], kj = skey[j];
                    const int      ii = sidx[i], ij = sidx[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);
                    if (i_first != desc) {
                        skey[i] = kj; skey[j] = ki;
                        sidx[i] = ij; sidx[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += block_size) {
        dst[(int64_t) row * k + i] = sidx[i];
    }
}

static void top_k_radix_mb_cuda(ggml_cuda_pool & pool, const float * src, int * dst,
        const int ncols, const int nrows, const int k, const bool sorted, cudaStream_t stream) {
    constexpr int bs = 256;
    const int nb    = std::max(1, std::min(64, ncols / 4096));
    const int chunk = (ncols + nb - 1) / nb;
    ggml_cuda_pool_alloc<topk_mb_state> st(pool, nrows);
    ggml_cuda_pool_alloc<uint32_t>      ghist(pool, (size_t) nrows * 256);
    ggml_cuda_pool_alloc<int>           cnt(pool, (size_t) nrows * nb);
    ggml_cuda_pool_alloc<uint32_t>      ckey(pool, (size_t) nrows * k);
    ggml_cuda_pool_alloc<int>           cidx(pool, (size_t) nrows * k);
    k_topk_mb_init<<<nrows, 256, 0, stream>>>(st.get(), ghist.get(), nrows, k);
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        k_topk_mb_hist<bs><<<dim3(nb, nrows), bs, 0, stream>>>(src, st.get(), ghist.get(), ncols, shift);
        k_topk_mb_pick<<<nrows, 256, 0, stream>>>(st.get(), ghist.get(), shift);
    }
    k_topk_mb_count<bs><<<dim3(nb, nrows), bs, 0, stream>>>(src, st.get(), cnt.get(), ncols, chunk);
    // unsorted: the candidates are the result
    k_topk_mb_emit<bs><<<dim3(nb, nrows), bs, 0, stream>>>(src, st.get(), cnt.get(), ckey.get(), sorted ? cidx.get() : dst, ncols, chunk, k);
    if (sorted) {
        k_topk_mb_sort<1024><<<nrows, 1024, 0, stream>>>(ckey.get(), cidx.get(), dst, k);
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // GGML_CUDA_TOPK_FAST=0 keeps the full argsort
    static const bool fast = !getenv("GGML_CUDA_TOPK_FAST") || atoi(getenv("GGML_CUDA_TOPK_FAST")) != 0;
    if (fast && k <= TOPK_SMALL_MAX && ncols >= 2048) {
        top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    if (fast && k > TOPK_SMALL_MAX && k <= TOPK_RADIX_MAX && ncols >= 4096 && nrows <= INT_MAX) {
        // few rows (a decode step): spread each row over many blocks; many rows (prefill): one block per row
        static const bool mb = !getenv("GGML_CUDA_TOPK_MB") || atoi(getenv("GGML_CUDA_TOPK_MB")) != 0;
        const bool sorted = ggml_get_op_params_i32(dst, 0) == 0;
        if (mb && nrows < 32 && ncols >= 16384) {
            top_k_radix_mb_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, sorted, stream);
        } else if (sorted) {
            k_top_k_radix<1024, true><<<(int) nrows, 1024, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
        } else {
            k_top_k_radix<1024, false><<<(int) nrows, 1024, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
        }
        return;
    }
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
