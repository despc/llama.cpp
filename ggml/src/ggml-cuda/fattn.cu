#include <mutex>
#include "common.cuh"
#include <cinttypes>
#include <vector>
#include <set>
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, const int ne30, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int query    = blockIdx.x;

    const half * mask = mask_ptr + sequence*s33 + query*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + query)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const bool selected = i < ne30 && isfinite(__half2float(mask[i]));
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = row_count;
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---------------------------------------------------------------------------
// Sparse preparation, opt-in via GGML_CUDA_FATTN_SPARSE_PREP.
//
// A sparse attention that keeps the existing compute has to first turn the
// per-query index lists into something a dense kernel can consume: the union of
// a tile's selections, the K/V rows at those positions gathered into a compact
// buffer, and a mask saying which of them each query actually chose.  Whether
// that preparation costs less than the positions it removes is the whole
// question, and it is measured here at real shapes before any kernel is written
// to consume the result.  Nothing downstream uses the output yet.
// ---------------------------------------------------------------------------

// Threads per preparation block; the block-wide scan sizes its buffer from it.
#define GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS 256

// One block per tile of queries.  Marks the tile's selections in a bitmap over
// the cache, then compacts the set bits into a sorted index list.
//
// Compaction is parallel: a block-wide scan over the bitmap's popcounts gives
// every word the slot its first bit occupies, after which each thread emits its
// own word independently.  The scan is kept because the membership mask needs
// exactly the same quantity -- the rank of a cache position within the union --
// so the two stages share it rather than each paying for its own pass.
static __global__ void fattn_sparse_union(
        const int32_t * __restrict__ indices, int32_t * __restrict__ union_idx,
        int32_t * __restrict__ union_len, uint32_t * __restrict__ bitmap_out,
        int32_t * __restrict__ prefix_out, int n_kv, int n_kv_max, int tile, int rows) {
    extern __shared__ uint32_t bitmap[];
    __shared__ int32_t scan[GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS];

    const int t0    = blockIdx.x * tile;
    const int words = (n_kv + 31) / 32;

    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        bitmap[i] = 0;
    }
    __syncthreads();

    for (int r = t0; r < t0 + tile && r < rows; ++r) {
        const int32_t * row = indices + (size_t) r * n_kv_max;
        for (int i = threadIdx.x; i < n_kv_max; i += blockDim.x) {
            const int32_t v = row[i];
            if (v >= 0 && v < n_kv) {
                atomicOr(&bitmap[v >> 5], 1u << (v & 31));
            }
        }
    }
    __syncthreads();

    uint32_t * bm = bitmap_out + (size_t) blockIdx.x * words;
    int32_t  * pf = prefix_out + (size_t) blockIdx.x * words;

    // Exclusive scan of per-word popcounts, in chunks of one block.  `carry` is
    // computed identically by every thread, so it stays uniform without a
    // broadcast.
    int32_t carry = 0;
    for (int base = 0; base < words; base += blockDim.x) {
        const int w   = base + threadIdx.x;
        const uint32_t bits = w < words ? bitmap[w] : 0u;
        const int32_t  cnt  = __popc(bits);

        scan[threadIdx.x] = cnt;
        __syncthreads();
        for (int d = 1; d < blockDim.x; d <<= 1) {
            const int32_t add = threadIdx.x >= d ? scan[threadIdx.x - d] : 0;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }
        const int32_t inclusive = scan[threadIdx.x];
        const int32_t total     = scan[blockDim.x - 1];

        if (w < words) {
            bm[w] = bits;
            pf[w] = carry + inclusive - cnt;   // exclusive: slot of this word's first bit
        }
        carry += total;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        union_len[blockIdx.x] = carry;
    }

    // Each thread emits its own words at the slots the scan assigned.  Positions
    // beyond n_kv were never set, so no bound is needed beyond the bitmap itself.
    int32_t * out = union_idx + (size_t) blockIdx.x * n_kv;
    for (int w = threadIdx.x; w < words; w += blockDim.x) {
        uint32_t m = bitmap[w];
        int      n = pf[w];
        while (m) {
            const int b = __ffs(m) - 1;
            out[n++] = (w << 5) + b;
            m &= m - 1;
        }
    }
}

// The third piece of preparation: which rows of the compact buffer does each
// query in the tile actually own?  The union is an OR over the tile, so without
// this a query would attend to positions its neighbours selected and it did not.
// One bit per (query, union slot); the slot of a cache position is its rank in
// the union, which the scan above already computed at word granularity.
static __global__ void fattn_sparse_query_mask(
        const int32_t * __restrict__ indices, const uint32_t * __restrict__ bitmap_in,
        const int32_t * __restrict__ prefix_in, uint32_t * __restrict__ qmask,
        int n_kv, int n_kv_max, int tile, int rows, int union_words) {
    const int t0    = blockIdx.x * tile;
    const int words = (n_kv + 31) / 32;

    const uint32_t * bm = bitmap_in + (size_t) blockIdx.x * words;
    const int32_t  * pf = prefix_in + (size_t) blockIdx.x * words;
    uint32_t * mask = qmask + (size_t) blockIdx.x * tile * union_words;

    for (size_t i = threadIdx.x; i < (size_t) tile * union_words; i += blockDim.x) {
        mask[i] = 0;
    }
    __syncthreads();

    for (int r = 0; r < tile && t0 + r < rows; ++r) {
        const int32_t * row = indices + (size_t) (t0 + r) * n_kv_max;
        uint32_t * dst = mask + (size_t) r * union_words;
        for (int i = threadIdx.x; i < n_kv_max; i += blockDim.x) {
            const int32_t v = row[i];
            if (v >= 0 && v < n_kv) {
                const int w    = v >> 5;
                const int rank = pf[w] + __popc(bm[w] & ((1u << (v & 31)) - 1));
                atomicOr(&dst[rank >> 5], 1u << (rank & 31));
            }
        }
    }
}

// Copies the K and V rows named by a tile's union into compact buffers, laid
// out exactly as the attention kernel reads a cache: row stride first, then
// head, then the tile as the batch dimension.  That is what lets the existing
// dense kernel run on the result without knowing anything about sparsity.
static __global__ void fattn_sparse_gather_rows(
        const char * __restrict__ srcK, char * __restrict__ dstK,
        const char * __restrict__ srcV, char * __restrict__ dstV,
        const int32_t * __restrict__ union_idx, const int32_t * __restrict__ union_len,
        int n_kv, size_t row_bytes_k, size_t src_stride_k, size_t head_stride_k,
        size_t row_bytes_v, size_t src_stride_v, size_t head_stride_v,
        int n_head_kv, int union_stride) {
    const int tile = blockIdx.y;
    const int len  = min(union_len[tile], union_stride);
    const int32_t * idx = union_idx + (size_t) tile * n_kv;

    for (int r = blockIdx.x; r < len; r += gridDim.x) {
        const int32_t pos = idx[r];
        if (pos < 0 || pos >= n_kv) {
            continue;
        }
        for (int h = 0; h < n_head_kv; ++h) {
            const size_t slot = ((size_t) tile * n_head_kv + h) * union_stride + r;
            {
                const char * s = srcK + h*head_stride_k + (size_t) pos * src_stride_k;
                char       * d = dstK + slot * row_bytes_k;
                for (size_t i = threadIdx.x*sizeof(int4); i + sizeof(int4) <= row_bytes_k; i += blockDim.x*sizeof(int4)) {
                    *(int4 *)(d + i) = *(const int4 *)(s + i);
                }
            }
            if (dstV) {
                const char * s = srcV + h*head_stride_v + (size_t) pos * src_stride_v;
                char       * d = dstV + slot * row_bytes_v;
                for (size_t i = threadIdx.x*sizeof(int4); i + sizeof(int4) <= row_bytes_v; i += blockDim.x*sizeof(int4)) {
                    *(int4 *)(d + i) = *(const int4 *)(s + i);
                }
            }
        }
    }
}

// The reference mask for one tile: the real mask's rows for those queries, over
// the whole cache, padded to the row count the kernels may read.
static __global__ void fattn_sparse_ref_mask(
        const half * __restrict__ src, half * __restrict__ dst,
        int n_kv, int row0, int tile, int mask_rows, int src_rows, int64_t s31) {
    const int r = blockIdx.y;
    half * out = dst + (size_t) r * n_kv;
    const bool live = r < tile && row0 + r < src_rows;
    const half * in = src + (size_t)(row0 + r) * s31;
    for (int j = blockIdx.x*blockDim.x + threadIdx.x; j < n_kv; j += gridDim.x*blockDim.x) {
        out[j] = live ? in[j] : __float2half(-INFINITY);
    }
    GGML_UNUSED(mask_rows);
}

// Every position, in order, as the union.  With this the compact path must
// reproduce the dense one exactly, so a disagreement is in the plumbing -- the
// query view, the buffer strides, the output layout -- and not in sparsity.
static __global__ void fattn_sparse_identity_union(
        int32_t * __restrict__ union_idx, int32_t * __restrict__ union_len, int n_kv) {
    int32_t * out = union_idx + (size_t) blockIdx.x * n_kv;
    for (int i = blockIdx.y*blockDim.x + threadIdx.x; i < n_kv; i += gridDim.y*blockDim.x) {
        out[i] = i;
    }
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        union_len[blockIdx.x] = n_kv;
    }
}

// The mask over the compact buffer is the real mask read at the union's
// positions.  Taking the value rather than synthesising a zero is both simpler
// and exact: whatever the model puts in an allowed entry travels with it, and a
// position the query did not select is already -inf in the source.  Slots past a
// tile's union, and rows past the queries it holds, are -inf -- the buffer is
// sized by the widest union in the group, so most tiles have a tail.
static __global__ void fattn_sparse_compact_mask(
        const half * __restrict__ src, const int32_t * __restrict__ union_idx,
        const int32_t * __restrict__ union_len, half * __restrict__ out,
        int n_kv, int row0, int tile, int union_n, int mask_rows, int rows,
        int src_rows, int64_t s31) {
    const int b = blockIdx.x;
    const int r = blockIdx.y;

    half * dst = out + ((size_t) b * mask_rows + r) * union_n;
    const int  len  = min(union_len[b], union_n);
    const int  row  = row0 + b*tile + r;
    const bool live = r < tile && b*tile + r < rows && row < src_rows;
    const half    * in  = src + (size_t) row * s31;
    const int32_t * idx = union_idx + (size_t) b * n_kv;

    for (int j = threadIdx.x; j < union_n; j += blockDim.x) {
        dst[j] = (live && j < len) ? in[idx[j]] : __float2half(-INFINITY);
    }
}

static void fattn_sparse_prep_check(const char * stage, cudaStream_t stream, bool synchronize) {
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && synchronize) {
        status = cudaStreamSynchronize(stream);
    }
    if (status != cudaSuccess) {
        GGML_LOG_ERROR("sparse preparation failed: stage=%s device=%d: %s\n", stage, ggml_cuda_get_device(), cudaGetErrorString(status));
    }
    CUDA_CHECK(status);
}

void ggml_cuda_flash_attn_ext_compact_mask(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, int32_t * indices, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst, indices, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, int(mask->ne[0]), n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());

    // Cost of turning the per-query lists into something a dense kernel could
    // consume.  Measured here and consumed by nobody: the point is to learn
    // whether preparation costs less than the positions it would remove, before
    // committing to the kernel that would use it. Scratch belongs to this context,
    // device and stream, outside the graph pool. It still consumes device memory.
    static const bool prep_measure = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP");
    if (prep_measure && K) {
        const int device = ggml_cuda_get_device();
        GGML_ASSERT(device == ctx.device);
        GGML_ASSERT(stream == ctx.stream());
        GGML_ASSERT(n_kv_max > 0 && mask->ne[0] > 0);
        GGML_ASSERT(mask->ne[0] == K->ne[1]);
        // This probe has no sequence offset in its gather addresses yet.
        GGML_ASSERT(mask->ne[3] == 1 && K->ne[3] == 1);
        auto & scratch = ctx.fattn_prep_scratch[device][ctx.curr_stream_no];
        // Finish the producer first so a compaction fault is not reported as a union fault.
        fattn_sparse_prep_check("compact_mask", stream, true);

        const int    rows      = int(mask->ne[1] * mask->ne[3]);
        const int    n_kv      = int(mask->ne[0]);
        const int    tile      = 16;                 // the dense path's ncols1
        const int    n_tiles   = (rows + tile - 1) / tile;
        // A union can contain every entry of all 16 lists. Truncation would undermeasure preparation and drop selected keys.
        const int    union_cap = int(std::min<int64_t>(n_kv, int64_t(tile) * n_kv_max));
        GGML_ASSERT(union_cap <= 4 * 16384);
        // Halved from the K-only budget: the buffer now holds V beside K, and the
        // prototype's F16 conversion of both sits after the output.
        const int    group     = std::max(1, std::min(4, (2 * 16384) / union_cap));
        const int    words     = (n_kv + 31) / 32;
        const size_t smem      = size_t(words) * sizeof(uint32_t);
        const size_t row_bytes   = ggml_row_size(K->type, K->ne[0]);
        const size_t row_bytes_v = ggml_row_size(V->type, V->ne[0]);
        const int    n_head_kv   = int(K->ne[2]);
        static const bool prep_gather_on = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_GATHER");

        const int    union_words = (union_cap + 31) / 32;
        const size_t need_idx  = size_t(group) * n_kv * sizeof(int32_t);
        const size_t need_gath = size_t(group) * union_cap * n_head_kv * (row_bytes + row_bytes_v);
        const size_t need_bm   = size_t(group) * words * sizeof(uint32_t);
        const size_t need_qm   = size_t(group) * tile * union_words * sizeof(uint32_t);
        static const bool prep_mask_on = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_MASK");

        if (smem <= 48*1024) {
            if (need_idx > scratch.union_idx_capacity) {
                if (scratch.union_idx) { CUDA_CHECK(cudaFree(scratch.union_idx)); }
                CUDA_CHECK(cudaMalloc(&scratch.union_idx, need_idx));
                scratch.union_idx_capacity = need_idx;
            }
            const size_t need_len = std::max<size_t>(group, 4) * sizeof(int32_t);
            if (need_len > scratch.union_len_capacity) {
                if (scratch.union_len) { CUDA_CHECK(cudaFree(scratch.union_len)); }
                CUDA_CHECK(cudaMalloc(&scratch.union_len, need_len));
                scratch.union_len_capacity = need_len;
            }
            // The bitmap and its prefix are written by the union and read by the
            // mask, so they are sized together and always allocated: compaction
            // itself now depends on the prefix.
            if (need_bm > scratch.bitmap_capacity) {
                if (scratch.bitmap) { CUDA_CHECK(cudaFree(scratch.bitmap)); }
                if (scratch.prefix) { CUDA_CHECK(cudaFree(scratch.prefix)); }
                CUDA_CHECK(cudaMalloc(&scratch.bitmap, need_bm));
                CUDA_CHECK(cudaMalloc(&scratch.prefix, need_bm));
                scratch.bitmap_capacity = need_bm;
            }
            if (prep_mask_on && need_qm > scratch.qmask_capacity) {
                if (scratch.qmask) { CUDA_CHECK(cudaFree(scratch.qmask)); }
                CUDA_CHECK(cudaMalloc(&scratch.qmask, need_qm));
                scratch.qmask_capacity = need_qm;
            }
            if (prep_gather_on) {
                // int4 loads require aligned rows; do not silently omit a short byte tail.
                GGML_ASSERT(reinterpret_cast<uintptr_t>(K->data) % alignof(int4) == 0);
                GGML_ASSERT(reinterpret_cast<uintptr_t>(V->data) % alignof(int4) == 0);
                GGML_ASSERT(K->nb[1] % alignof(int4) == 0 && K->nb[2] % alignof(int4) == 0);
                GGML_ASSERT(V->nb[1] % alignof(int4) == 0 && V->nb[2] % alignof(int4) == 0);
                GGML_ASSERT(row_bytes % sizeof(int4) == 0 && row_bytes_v % sizeof(int4) == 0);
                if (need_gath > scratch.gathered_capacity) {
                    if (scratch.gathered) { CUDA_CHECK(cudaFree(scratch.gathered)); }
                    CUDA_CHECK(cudaMalloc(&scratch.gathered, need_gath));
                    scratch.gathered_capacity = need_gath;
                }
            }

            for (int t0 = 0; t0 < n_tiles; t0 += group) {
                const int nt = std::min(group, n_tiles - t0);
                {
                    fattn_stage_timer t("prep_union", ggml_cuda_get_device(), rows, stream);
                    fattn_sparse_union<<<nt, GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS, smem, stream>>>(
                        indices + (size_t) t0 * tile * n_kv_max, scratch.union_idx, scratch.union_len,
                        scratch.bitmap, scratch.prefix, n_kv, n_kv_max, tile, rows - t0*tile);
                    fattn_sparse_prep_check("prep_union", stream, !fattn_stage_profile::enabled());
                }
                if (prep_mask_on) {
                    fattn_stage_timer t("prep_mask", ggml_cuda_get_device(), rows, stream);
                    fattn_sparse_query_mask<<<nt, GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS, 0, stream>>>(
                        indices + (size_t) t0 * tile * n_kv_max, scratch.bitmap, scratch.prefix,
                        scratch.qmask, n_kv, n_kv_max, tile, rows - t0*tile, union_words);
                    fattn_sparse_prep_check("prep_mask", stream, !fattn_stage_profile::enabled());
                }
                // Correctness of the preparation, checked against the index lists it
                // was built from.  A membership mask that is merely plausible is
                // worthless: an extra bit makes a query attend to a neighbour's
                // position, a missing one silently drops a selected key, and both
                // survive every performance measurement.  Opt-in, and it copies the
                // whole working set back, so it is a gate to pass once and not a
                // thing to leave on.
                static const bool verify_on = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_VERIFY");
                if (verify_on && prep_mask_on) {
                    static bool announced = false;
                    const size_t n_rows_g = size_t(nt) * tile;
                    std::vector<int32_t>  h_idx(n_rows_g * n_kv_max);
                    std::vector<int32_t>  h_uni(size_t(nt) * n_kv);
                    std::vector<int32_t>  h_len(nt);
                    std::vector<uint32_t> h_msk(size_t(nt) * tile * union_words);
                    CUDA_CHECK(cudaMemcpyAsync(h_idx.data(), indices + (size_t) t0 * tile * n_kv_max,
                        h_idx.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(h_uni.data(), scratch.union_idx,
                        h_uni.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(h_len.data(), scratch.union_len,
                        h_len.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaMemcpyAsync(h_msk.data(), scratch.qmask,
                        h_msk.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    int failures = 0;
                    for (int b = 0; b < nt; ++b) {
                        const int len = h_len[b];
                        const int32_t * uni = h_uni.data() + (size_t) b * n_kv;
                        if (len < 0 || len > union_cap) {
                            GGML_LOG_ERROR("prep_verify tile=%d union_len=%d exceeds cap %d\n", b, len, union_cap);
                            failures++;
                            continue;
                        }
                        std::set<int32_t> all;
                        for (int r = 0; r < tile && (t0+b)*tile + r < rows; ++r) {
                            const int32_t * row = h_idx.data() + (size_t)(b*tile + r) * n_kv_max;
                            std::set<int32_t> want, got;
                            for (int i = 0; i < n_kv_max; ++i) {
                                const int32_t v = row[i];
                                if (v >= 0 && v < n_kv) { want.insert(v); all.insert(v); }
                            }
                            const uint32_t * m = h_msk.data() + (size_t)(b*tile + r) * union_words;
                            for (int j = 0; j < len; ++j) {
                                if (m[j >> 5] & (1u << (j & 31))) { got.insert(uni[j]); }
                            }
                            // A bit set past the union's length would address a row the
                            // gather never wrote, so it has to be caught separately.
                            for (int j = len; j < union_words*32; ++j) {
                                if (m[j >> 5] & (1u << (j & 31))) {
                                    GGML_LOG_ERROR("prep_verify tile=%d row=%d bit %d set beyond union_len %d\n", b, r, j, len);
                                    failures++;
                                    break;
                                }
                            }
                            if (want != got) {
                                GGML_LOG_ERROR("prep_verify tile=%d row=%d membership mismatch: want %zu got %zu\n",
                                               b, r, want.size(), got.size());
                                failures++;
                            }
                        }
                        std::set<int32_t> uset(uni, uni + len);
                        if ((int) uset.size() != len) {
                            GGML_LOG_ERROR("prep_verify tile=%d union has duplicates: %d entries, %zu distinct\n", b, len, uset.size());
                            failures++;
                        }
                        if (uset != all) {
                            GGML_LOG_ERROR("prep_verify tile=%d union != OR of rows: %zu vs %zu\n", b, uset.size(), all.size());
                            failures++;
                        }
                        for (int j = 1; j < len; ++j) {
                            if (uni[j] <= uni[j-1]) {
                                GGML_LOG_ERROR("prep_verify tile=%d union not ascending at %d\n", b, j);
                                failures++;
                                break;
                            }
                        }
                    }
                    GGML_ASSERT(failures == 0);
                    if (!announced) {
                        announced = true;
                        GGML_LOG_WARN("prep_verify: union and per-query membership agree with the index lists (n_kv=%d, tile=%d)\n", n_kv, tile);
                    }
                }
                char * gK = scratch.gathered;
                char * gV = gK + size_t(group) * union_cap * n_head_kv * row_bytes;
                // Isolation switch: replace the union with every position in order.
                static const bool identity_union = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_IDENTITY");
                if (identity_union) {
                    GGML_ASSERT(union_cap >= n_kv);
                    const dim3 g(nt, 64, 1);
                    fattn_sparse_identity_union<<<g, 256, 0, stream>>>(scratch.union_idx, scratch.union_len, n_kv);
                    fattn_sparse_prep_check("identity_union", stream, true);
                }
                if (prep_gather_on) {
                    fattn_stage_timer t("prep_gather", ggml_cuda_get_device(), rows, stream);
                    const dim3 grid(256, nt, 1);
                    fattn_sparse_gather_rows<<<grid, 64, 0, stream>>>(
                        (const char *) K->data, gK, (const char *) V->data, gV,
                        scratch.union_idx, scratch.union_len, n_kv,
                        row_bytes,   K->nb[1], K->nb[2],
                        row_bytes_v, V->nb[1], V->nb[2],
                        n_head_kv, union_cap);
                    fattn_sparse_prep_check("prep_gather", stream, !fattn_stage_profile::enabled());

                    // The compact buffer has to hold the bytes of the rows the union
                    // names, in the order the attention kernel will read them.  A
                    // wrong stride here still produces plausible timings and wrong
                    // attention, so it is checked against the source rather than
                    // assumed from the index arithmetic.
                    static const bool verify_gather = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_VERIFY");
                    if (verify_gather) {
                        static bool gather_announced = false;
                        std::vector<int32_t> h_len(nt);
                        CUDA_CHECK(cudaMemcpyAsync(h_len.data(), scratch.union_len,
                            h_len.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        int bad = 0;
                        for (int b = 0; b < nt && bad == 0; ++b) {
                            const int len = std::min(h_len[b], union_cap);
                            std::vector<int32_t> uni(len);
                            CUDA_CHECK(cudaMemcpyAsync(uni.data(), scratch.union_idx + (size_t) b * n_kv,
                                size_t(len)*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));

                            // sample rows across the union rather than all of them
                            const int step = std::max(1, len / 64);
                            for (int r = 0; r < len && bad == 0; r += step) {
                                for (int h = 0; h < n_head_kv; ++h) {
                                    const size_t slot = ((size_t) b * n_head_kv + h) * union_cap + r;
                                    std::vector<char> want(row_bytes), got(row_bytes);
                                    CUDA_CHECK(cudaMemcpyAsync(want.data(),
                                        (const char *) K->data + h*K->nb[2] + (size_t) uni[r] * K->nb[1],
                                        row_bytes, cudaMemcpyDeviceToHost, stream));
                                    CUDA_CHECK(cudaMemcpyAsync(got.data(), gK + slot * row_bytes,
                                        row_bytes, cudaMemcpyDeviceToHost, stream));
                                    CUDA_CHECK(cudaStreamSynchronize(stream));
                                    if (want != got) {
                                        GGML_LOG_ERROR("prep_verify gather K mismatch tile=%d slot=%d head=%d pos=%d\n", b, r, h, uni[r]);
                                        bad++;
                                        break;
                                    }
                                    std::vector<char> wantv(row_bytes_v), gotv(row_bytes_v);
                                    CUDA_CHECK(cudaMemcpyAsync(wantv.data(),
                                        (const char *) V->data + h*V->nb[2] + (size_t) uni[r] * V->nb[1],
                                        row_bytes_v, cudaMemcpyDeviceToHost, stream));
                                    CUDA_CHECK(cudaMemcpyAsync(gotv.data(), gV + slot * row_bytes_v,
                                        row_bytes_v, cudaMemcpyDeviceToHost, stream));
                                    CUDA_CHECK(cudaStreamSynchronize(stream));
                                    if (wantv != gotv) {
                                        GGML_LOG_ERROR("prep_verify gather V mismatch tile=%d slot=%d head=%d pos=%d\n", b, r, h, uni[r]);
                                        bad++;
                                        break;
                                    }
                                }
                            }
                        }
                        GGML_ASSERT(bad == 0);
                        if (!gather_announced) {
                            gather_announced = true;
                            GGML_LOG_WARN("prep_verify: gathered K and V rows match the source at the union's positions\n");
                        }
                    }
                }

                // The prototype: run the ordinary dense dispatch on the compact
                // buffer.  Everything it is handed is a plain tensor -- gathered
                // K and V, an F16 mask over the union, a strided view of this
                // group's queries -- so the attention itself is the code already
                // in production, and what is measured is a real kernel on real
                // selections rather than a microbenchmark standing in for one.
                static const bool prep_attn_on = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_ATTN");
                if (prep_attn_on && prep_gather_on && dst->src[4] == nullptr) {
                    // The buffer is sized by the group's widest union, so its length
                    // must come back to the host before the launch can be shaped.
                    // A real implementation would pad to a fixed granularity and
                    // avoid this synchronisation; the prototype pays it to keep the
                    // measured kernel honest about the rows it actually visits.
                    std::vector<int32_t> h_len(nt);
                    CUDA_CHECK(cudaMemcpyAsync(h_len.data(), scratch.union_len,
                        h_len.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    int union_n = 0;
                    for (int b = 0; b < nt; ++b) {
                        union_n = std::max(union_n, h_len[b]);
                    }
                    union_n = std::min(GGML_PAD(union_n, 256), union_cap);

                    // llama.cpp pads mask rows to 64 so a wide query tile can read
                    // past the queries it holds; the tail is -inf.
                    const int      mask_rows = 64;
                    const int64_t  n_head    = Q->ne[2];
                    const size_t   need_cm   = size_t(nt) * mask_rows * union_n * sizeof(half);
                    // dst, then the F16 K and V that launch_fattn expects to find after it
                    const size_t   dst_bytes = size_t(V->ne[0]) * n_head * tile * nt * sizeof(float);
                    const size_t   need_out  = GGML_PAD(dst_bytes, 128)
                        + GGML_PAD(size_t(K->ne[0]) * union_n * n_head_kv * nt * sizeof(half), 128)
                        + GGML_PAD(size_t(V->ne[0]) * union_n * n_head_kv * nt * sizeof(half), 128);

                    if (need_cm > scratch.cmask_capacity) {
                        if (scratch.cmask) { CUDA_CHECK(cudaFree(scratch.cmask)); }
                        CUDA_CHECK(cudaMalloc(&scratch.cmask, need_cm));
                        scratch.cmask_capacity = need_cm;
                    }
                    if (need_out > scratch.out_capacity) {
                        if (scratch.out) { CUDA_CHECK(cudaFree(scratch.out)); }
                        CUDA_CHECK(cudaMalloc(&scratch.out, need_out));
                        scratch.out_capacity = need_out;
                    }

                    {
                        fattn_stage_timer t("prep_cmask", ggml_cuda_get_device(), rows, stream);
                        const dim3 grid(nt, mask_rows, 1);
                        fattn_sparse_compact_mask<<<grid, GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS, 0, stream>>>(
                            (const half *) mask->data, scratch.union_idx, scratch.union_len, scratch.cmask,
                            n_kv, t0*tile, tile, union_n, mask_rows, rows - t0*tile,
                            int(mask->ne[1]), s31);
                        fattn_sparse_prep_check("prep_cmask", stream, !fattn_stage_profile::enabled());
                    }

                    ggml_tensor Qc = *Q;
                    Qc.ne[1] = tile;    Qc.ne[3] = nt;
                    Qc.nb[3] = size_t(tile) * Q->nb[1];
                    Qc.data  = (char *) Q->data + size_t(t0) * tile * Q->nb[1];
                    Qc.view_src = nullptr; Qc.buffer = nullptr;

                    ggml_tensor Kc = *K;
                    Kc.ne[1] = union_n; Kc.ne[2] = n_head_kv; Kc.ne[3] = nt;
                    Kc.nb[1] = row_bytes;
                    Kc.nb[2] = size_t(union_cap) * row_bytes;
                    Kc.nb[3] = size_t(n_head_kv) * union_cap * row_bytes;
                    Kc.data  = gK; Kc.view_src = nullptr; Kc.buffer = nullptr;

                    ggml_tensor Vc = *V;
                    Vc.ne[1] = union_n; Vc.ne[2] = n_head_kv; Vc.ne[3] = nt;
                    Vc.nb[1] = row_bytes_v;
                    Vc.nb[2] = size_t(union_cap) * row_bytes_v;
                    Vc.nb[3] = size_t(n_head_kv) * union_cap * row_bytes_v;
                    Vc.data  = gV; Vc.view_src = nullptr; Vc.buffer = nullptr;

                    ggml_tensor Mc = *mask;
                    Mc.ne[0] = union_n; Mc.ne[1] = mask_rows; Mc.ne[2] = 1; Mc.ne[3] = nt;
                    Mc.nb[0] = sizeof(half);
                    Mc.nb[1] = size_t(union_n) * sizeof(half);
                    Mc.nb[2] = size_t(mask_rows) * Mc.nb[1];
                    Mc.nb[3] = Mc.nb[2];
                    Mc.data  = scratch.cmask; Mc.view_src = nullptr; Mc.buffer = nullptr;

                    ggml_tensor Dc = *dst;
                    Dc.ne[0] = V->ne[0]; Dc.ne[1] = n_head; Dc.ne[2] = tile; Dc.ne[3] = nt;
                    Dc.nb[0] = sizeof(float);
                    Dc.nb[1] = Dc.ne[0] * Dc.nb[0];
                    Dc.nb[2] = Dc.ne[1] * Dc.nb[1];
                    Dc.nb[3] = Dc.ne[2] * Dc.nb[2];
                    Dc.data  = scratch.out; Dc.view_src = nullptr; Dc.buffer = nullptr;
                    Dc.src[0] = &Qc; Dc.src[1] = &Kc; Dc.src[2] = &Vc; Dc.src[3] = &Mc; Dc.src[4] = nullptr;
                    // n_kv_max of zero keeps the inner dispatch on the dense path,
                    // so the prototype cannot recurse into its own preparation.
                    ggml_set_op_params_i32(&Dc, 4, 0);

                    {
                        fattn_stage_timer t("prep_attn", ggml_cuda_get_device(), rows, stream);
                        g_fattn_in_compact = true;
                        ggml_cuda_flash_attn_ext(ctx, &Dc);
                        g_fattn_in_compact = false;
                        fattn_sparse_prep_check("prep_attn", stream, !fattn_stage_profile::enabled());
                    }

                    // The decisive check.  A compact path can look fast for the
                    // wrong reason: flash attention skips a key block whose mask
                    // is entirely -inf, so a mask that wrongly masks too much buys
                    // speed by computing less.  Run the ordinary dense dispatch on
                    // the same queries against the full cache and compare outputs.
                    static const bool verify_attn = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_PREP_VERIFY");
                    if (verify_attn) {
                        static bool attn_announced = false;
                        for (int b = 0; b < nt; ++b) {
                            GGML_ASSERT(h_len[b] <= union_n);   // a truncated union would drop selected keys
                        }

                        const size_t ref_mask_bytes = size_t(mask_rows) * n_kv * sizeof(half);
                        const size_t ref_out_bytes  = GGML_PAD(size_t(V->ne[0]) * n_head * tile * sizeof(float), 128)
                            + GGML_PAD(size_t(K->ne[0]) * n_kv * n_head_kv * sizeof(half), 128)
                            + GGML_PAD(size_t(V->ne[0]) * n_kv * n_head_kv * sizeof(half), 128);
                        half  * ref_mask = nullptr;
                        float * ref_out  = nullptr;
                        CUDA_CHECK(cudaMalloc(&ref_mask, ref_mask_bytes));
                        CUDA_CHECK(cudaMalloc(&ref_out,  ref_out_bytes));

                        // Elementwise relative error is the wrong measure here: most of
                        // the output is near zero, where a difference of one ulp reads as
                        // a large relative error.  Normalised mean square error is what
                        // ggml's own backend tests use for this operator.
                        double sum_d2 = 0.0, sum_e2 = 0.0, max_abs = 0.0;
                        int worst_b = -1, worst_i = -1;
                        double worst = 0.0;
                        for (int b = 0; b < nt; ++b) {
                            const int row0 = (t0 + b) * tile;
                            if (row0 >= rows) {
                                break;
                            }
                            {
                                const dim3 g(64, mask_rows, 1);
                                fattn_sparse_ref_mask<<<g, 256, 0, stream>>>(
                                    (const half *) mask->data, ref_mask, n_kv, row0, tile,
                                    mask_rows, int(mask->ne[1]), s31);
                                fattn_sparse_prep_check("ref_mask", stream, true);
                            }

                            ggml_tensor Qd = *Q;
                            Qd.ne[1] = tile; Qd.ne[3] = 1;
                            Qd.data  = (char *) Q->data + size_t(row0) * Q->nb[1];
                            Qd.view_src = nullptr; Qd.buffer = nullptr;

                            ggml_tensor Md = *mask;
                            Md.ne[0] = n_kv; Md.ne[1] = mask_rows; Md.ne[2] = 1; Md.ne[3] = 1;
                            Md.nb[0] = sizeof(half);
                            Md.nb[1] = size_t(n_kv) * sizeof(half);
                            Md.nb[2] = size_t(mask_rows) * Md.nb[1];
                            Md.nb[3] = Md.nb[2];
                            Md.data  = ref_mask; Md.view_src = nullptr; Md.buffer = nullptr;

                            ggml_tensor Dd = *dst;
                            Dd.ne[0] = V->ne[0]; Dd.ne[1] = n_head; Dd.ne[2] = tile; Dd.ne[3] = 1;
                            Dd.nb[0] = sizeof(float);
                            Dd.nb[1] = Dd.ne[0] * Dd.nb[0];
                            Dd.nb[2] = Dd.ne[1] * Dd.nb[1];
                            Dd.nb[3] = Dd.ne[2] * Dd.nb[2];
                            Dd.data  = ref_out; Dd.view_src = nullptr; Dd.buffer = nullptr;
                            Dd.src[0] = &Qd; Dd.src[1] = (ggml_tensor *) K; Dd.src[2] = (ggml_tensor *) V;
                            Dd.src[3] = &Md; Dd.src[4] = nullptr;
                            ggml_set_op_params_i32(&Dd, 4, 0);

                            g_fattn_in_compact = true;
                            ggml_cuda_flash_attn_ext(ctx, &Dd);
                            g_fattn_in_compact = false;
                            fattn_sparse_prep_check("ref_attn", stream, true);

                            const size_t n_out = size_t(V->ne[0]) * n_head * tile;
                            std::vector<float> a(n_out), e(n_out);
                            CUDA_CHECK(cudaMemcpyAsync(a.data(), scratch.out + (size_t) b * n_out,
                                n_out*sizeof(float), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaMemcpyAsync(e.data(), ref_out, n_out*sizeof(float), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                            for (size_t i = 0; i < n_out; ++i) {
                                const double d = double(a[i]) - double(e[i]);
                                sum_d2 += d*d;
                                sum_e2 += double(e[i])*double(e[i]);
                                if (std::fabs(d) > max_abs) { max_abs = std::fabs(d); worst_b = b; worst_i = int(i); }
                            }
                        }
                        CUDA_CHECK(cudaFree(ref_mask));
                        CUDA_CHECK(cudaFree(ref_out));

                        const double nmse = sum_e2 > 0.0 ? sum_d2 / sum_e2 : 0.0;
                        worst = nmse;
                        // ggml's own backend test for this operator accepts 5e-4.
                        if (nmse > 5e-4) {
                            GGML_LOG_ERROR("prep_verify attention mismatch: nmse %.4g at tile=%d elem=%d (n_kv=%d union_n=%d n_head=%d tile=%d DV=%d)\n",
                                           worst, worst_b, worst_i, n_kv, union_n, int(n_head), tile, int(V->ne[0]));
                            // Where the disagreement sits says which index is wrong:
                            // one query means the query view, one head means the head
                            // stride, everything means the layout.
                            const int row0 = (t0 + worst_b) * tile;
                            const size_t n_out = size_t(V->ne[0]) * n_head * tile;
                            std::vector<float> a(n_out), e(n_out);
                            CUDA_CHECK(cudaMemcpyAsync(a.data(), scratch.out + (size_t) worst_b * n_out,
                                n_out*sizeof(float), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                            {
                                half * rm = nullptr; float * ro = nullptr;
                                CUDA_CHECK(cudaMalloc(&rm, size_t(mask_rows) * n_kv * sizeof(half)));
                                CUDA_CHECK(cudaMalloc(&ro, GGML_PAD(n_out*sizeof(float), 128)
                                    + GGML_PAD(size_t(K->ne[0]) * n_kv * n_head_kv * sizeof(half), 128)
                                    + GGML_PAD(size_t(V->ne[0]) * n_kv * n_head_kv * sizeof(half), 128)));
                                const dim3 g(64, mask_rows, 1);
                                fattn_sparse_ref_mask<<<g, 256, 0, stream>>>(
                                    (const half *) mask->data, rm, n_kv, row0, tile, mask_rows, int(mask->ne[1]), s31);
                                ggml_tensor Qd = *Q; Qd.ne[1] = tile; Qd.ne[3] = 1;
                                Qd.data = (char *) Q->data + size_t(row0) * Q->nb[1];
                                Qd.view_src = nullptr; Qd.buffer = nullptr;
                                ggml_tensor Md = *mask;
                                Md.ne[0] = n_kv; Md.ne[1] = mask_rows; Md.ne[2] = 1; Md.ne[3] = 1;
                                Md.nb[0] = sizeof(half); Md.nb[1] = size_t(n_kv)*sizeof(half);
                                Md.nb[2] = size_t(mask_rows)*Md.nb[1]; Md.nb[3] = Md.nb[2];
                                Md.data = rm; Md.view_src = nullptr; Md.buffer = nullptr;
                                ggml_tensor Dd = *dst;
                                Dd.ne[0] = V->ne[0]; Dd.ne[1] = n_head; Dd.ne[2] = tile; Dd.ne[3] = 1;
                                Dd.nb[0] = sizeof(float); Dd.nb[1] = Dd.ne[0]*Dd.nb[0];
                                Dd.nb[2] = Dd.ne[1]*Dd.nb[1]; Dd.nb[3] = Dd.ne[2]*Dd.nb[2];
                                Dd.data = ro; Dd.view_src = nullptr; Dd.buffer = nullptr;
                                Dd.src[0] = &Qd; Dd.src[1] = (ggml_tensor *) K; Dd.src[2] = (ggml_tensor *) V;
                                Dd.src[3] = &Md; Dd.src[4] = nullptr;
                                ggml_set_op_params_i32(&Dd, 4, 0);
                                g_fattn_in_compact = true; ggml_cuda_flash_attn_ext(ctx, &Dd); g_fattn_in_compact = false;
                                CUDA_CHECK(cudaMemcpyAsync(e.data(), ro, n_out*sizeof(float), cudaMemcpyDeviceToHost, stream));
                                CUDA_CHECK(cudaStreamSynchronize(stream));
                                CUDA_CHECK(cudaFree(rm)); CUDA_CHECK(cudaFree(ro));
                            }
                            const int DV = int(V->ne[0]);
                            for (int q = 0; q < tile; ++q) {
                                double mq = 0.0; int nbad = 0;
                                for (int h = 0; h < n_head; ++h) {
                                    for (int d = 0; d < DV; ++d) {
                                        const size_t i = size_t(d) + DV*(h + n_head*q);
                                        const double df = std::fabs(double(a[i]) - double(e[i]));
                                        const double sc = std::max(std::fabs(double(e[i])), 1e-3);
                                        if (df/sc > mq) { mq = df/sc; }
                                        if (df/sc > 2e-2) { nbad++; }
                                    }
                                }
                                GGML_LOG_ERROR("  query %2d: max rel %.4g, %d/%d elements off\n", q, mq, nbad, int(n_head)*DV);
                            }
                            for (int h = 0; h < n_head; ++h) {
                                double mh = 0.0;
                                for (int q = 0; q < tile; ++q) {
                                    for (int d = 0; d < DV; ++d) {
                                        const size_t i = size_t(d) + DV*(h + n_head*q);
                                        const double df = std::fabs(double(a[i]) - double(e[i]));
                                        const double sc = std::max(std::fabs(double(e[i])), 1e-3);
                                        if (df/sc > mh) { mh = df/sc; }
                                    }
                                }
                                GGML_LOG_ERROR("  head %2d: max rel %.4g\n", h, mh);
                            }
                            GGML_ABORT("compact attention disagrees with dense");
                        }
                        static int64_t nmse_next = 0;
                        if (!attn_announced || n_kv >= nmse_next) {
                            attn_announced = true;
                            nmse_next = int64_t(n_kv) * 2;
                            GGML_LOG_WARN("prep_verify: compact attention matches dense, nmse %.3g max_abs %.3g (n_kv=%d union_n=%d)\n",
                                          nmse, max_abs, n_kv, union_n);
                        }
                    }
                }
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // How much do neighbouring queries select in common?  A sparse attention that
    // gives every query its own index list cannot amortise a K or V row across a
    // tile of queries the way the dense kernel does, and at these shapes that costs
    // more traffic than visiting fewer positions saves.  The only way it pays is a
    // shared list per tile with per-query masking, and whether that is smaller than
    // the dense span is entirely a property of the model's selections.  Opt-in via
    // GGML_CUDA_FATTN_SPARSE_OVERLAP; reports once and is not cheap.
    static const bool overlap_report = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_OVERLAP");
    static int64_t overlap_next = 4096;   // report when the cache passes each power of two
    if (overlap_report && mask->ne[0] >= overlap_next) {
        overlap_next = mask->ne[0] * 2;

        const int64_t rows = mask->ne[1] * mask->ne[3];
        std::vector<int32_t> host(size_t(rows) * n_kv_max);
        CUDA_CHECK(cudaMemcpyAsync(host.data(), indices, host.size()*sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GGML_LOG_WARN("fattn_sparse_overlap n_kv=%d n_kv_max=%d rows=%" PRId64 "\n",
                      int(mask->ne[0]), n_kv_max, rows);
        for (int tile : {1, 2, 4, 8, 16, 32, 64}) {
            int64_t union_total = 0;
            int64_t tiles = 0;
            for (int64_t r0 = 0; r0 + tile <= rows; r0 += tile) {
                std::set<int32_t> u;
                for (int64_t r = r0; r < r0 + tile; ++r) {
                    for (int i = 0; i < n_kv_max; ++i) {
                        const int32_t v = host[size_t(r)*n_kv_max + i];
                        if (v >= 0) {
                            u.insert(v);
                        }
                    }
                }
                union_total += (int64_t) u.size();
                tiles++;
            }
            if (tiles) {
                const double mean_union = double(union_total) / tiles;
                // rows read per query, against one row per query in a dense tile pass
                GGML_LOG_WARN("fattn_sparse_overlap tile=%-3d mean_union=%8.1f  rows_read_per_query=%7.1f  (dense would read %d)\n",
                              tile, mean_union, mean_union / tile, int(mask->ne[0]));
            }
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // Compacting the mask costs one pass over the whole cache regardless of how many
    // queries there are, so it is repaid by the queries that then skip the unselected
    // positions.  At one or two queries -- decode, and speculative verification --
    // it is not repaid, so leave those to the dense kernels.
    if (Q->ne[1] < 16) {
        return false;
    }

    // 256/256 is off unless asked for.  Enabling it was neutral on prefill here --
    // only the two Blackwells can take the path and they hold about a sixth of the
    // GPU time -- and appeared to cost generation, but that measurement was taken
    // against binaries carrying an unconditional profiler synchronisation and has
    // to be redone.  A run-time switch keeps both arms in one binary.
    if (Q->ne[0] == 256 && K->ne[0] == 256) {
        static const bool sparse_256 = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_256");
        if (!sparse_256) {
            return false;
        }
    }

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// Reports what geometry was requested and what was actually used, once per
// distinct outcome.  A forced value that no instantiation exists for must be
// visible: silently using the default instead makes the measurement compare the
// default against itself.
static void ggml_cuda_fattn_geometry_report(int cc, int want1, int want2, int got1, int got2, bool taken) {
    static std::set<std::tuple<int,int,int,int,int,int>> seen;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (!seen.insert({cc, want1, want2, got1, got2, taken ? 1 : 0}).second) {
        return;
    }
    if (taken) {
        GGML_LOG_WARN("fattn_geometry cc=%d requested ncols1=%d ncols2=%d -> USED ncols1=%d ncols2=%d\n",
                      cc, want1, want2, got1, got2);
    } else {
        GGML_LOG_WARN("fattn_geometry cc=%d requested ncols1=%d ncols2=%d -> NOT AVAILABLE (needs ncols1*ncols2 <= 64), falling back to the default rule\n",
                      cc, want1, want2);
    }
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    static const bool only_volta = !ggml_env_flag_enabled("GGML_CUDA_FATTN_GEOMETRY_ALL_ARCH");
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    // Only instantiations with ncols1*ncols2 <= 64 exist, so a request outside that
    // is not available.  Say so rather than falling through to the default, which
    // silently turns the experiment into a comparison of the default with itself.
    static const int ncols1_forced = getenv("GGML_CUDA_FATTN_NCOLS1") ? atoi(getenv("GGML_CUDA_FATTN_NCOLS1")) : 0;
    if (ncols1_forced && (!only_volta || cc == GGML_CUDA_CC_VOLTA)) {
        bool taken = false;
        switch (ncols1_forced) {
            case  8: if constexpr (8*ncols2  <= 64) { taken = true; ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV,  8, ncols2>(ctx, dst); } break;
            case 16: if constexpr (16*ncols2 <= 64) { taken = true; ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16, ncols2>(ctx, dst); } break;
            case 32: if constexpr (32*ncols2 <= 64) { taken = true; ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32, ncols2>(ctx, dst); } break;
            default: break;
        }
        if (taken) {
            ggml_cuda_fattn_geometry_report(cc, ncols1_forced, ncols2, ncols1_forced, ncols2, true);
            return;
        }
        ggml_cuda_fattn_geometry_report(cc, ncols1_forced, ncols2, 0, ncols2, false);
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // Diagnostic override: the tile geometry is chosen by a fixed rule, and whether
    // that rule is right for these shapes on Volta has never been measured.  These
    // select among already-compiled instantiations, so the sweep costs no build.
    static const int ncols2_forced = getenv("GGML_CUDA_FATTN_NCOLS2") ? atoi(getenv("GGML_CUDA_FATTN_NCOLS2")) : 0;

    // The sparse path puts one query column in each block, so its grouping is the
    // only thing left to choose, and 8 is the only 256/256 grouping any tile table
    // configures.  Volta's rule below would pick 4, which has no configuration at
    // all.  Volta cannot reach the sparse path anyway -- its MMA fragments are
    // fixed at 32 columns -- but keeping the choice here means the grouping is
    // decided in one place if that ever changes.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, 8)) {
        if (use_gqa_opt && gqa_ratio > 4 && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(ctx, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && ncols2_forced) {
            switch (ncols2_forced) {
                case  1: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  1>(ctx, dst); return;
                case  2: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  2>(ctx, dst); return;
                case  4: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  4>(ctx, dst); return;
                case  8: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  8>(ctx, dst); return;
                default: break;
            }
        }
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && ncols2_forced) {
        switch (ncols2_forced) {
            case  1: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  1>(ctx, dst); return;
            case  2: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  2>(ctx, dst); return;
            case  4: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  4>(ctx, dst); return;
            case  8: ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV,  8>(ctx, dst); return;
            default: break;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        // The MMA kernels for the large head sizes do not fit on Volta:
        // cudaOccupancyMaxActiveBlocksPerMultiprocessor reports 0 blocks and the assert in
        // launch_fattn fires. Report the op as unsupported instead, which lets the scheduler
        // pick another backend.
        //
        // It is not (only) the 96 KB of shared memory per SM. DeepSeek V4-Flash runs MLA at
        // 512/512, Volta has an explicit config for that shape, and its widest step needs about
        // 81 KB - capping the ncols ladder to keep it under 96 KB was measured and the assert
        // still fires, so the binding limit is occupancy, i.e. registers. Shrinking the tile
        // does not help; a Volta-capable kernel would need a different register budget.
        if (Q->ne[0] > 256) {
            return BEST_FATTN_KERNEL_NONE;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// ---------------------------------------------------------------------------
// The compact attention path.  Builds the union of each tile's selections,
// gathers K and V at those positions, reads the real mask at them, and runs the
// ordinary dense dispatch on the result -- so the attention that executes is the
// production kernel, on a cache of a few thousand rows instead of the whole one.
//
// Opt-in via GGML_CUDA_FATTN_SPARSE_COMPACT; clearing it restores the dense path
// exactly, since nothing outside this function changes behaviour.
// GGML_CUDA_FATTN_SPARSE_COMPACT_MIN_KV is the cache length below which the
// dense path is left alone: the preparation is a fixed cost per tile and the
// union is not much smaller than a short cache, so below it this loses.
// ---------------------------------------------------------------------------
static bool ggml_cuda_flash_attn_ext_compact_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (mask == nullptr || dst->src[4] != nullptr) {
        return false;
    }
    // The sparse marker: llama sets it on the layers whose mask carries a selection.
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (n_kv_max <= 0) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (mask->ne[0] != K->ne[1] || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }

    const int tile  = 16;
    const int n_q   = int(Q->ne[1]);
    const int n_kv  = int(mask->ne[0]);
    // A partial tile would make the query view read past Q; those microbatches are
    // small and the dense path handles them.
    if (n_q < tile || n_q % tile != 0) {
        return false;
    }
    if (mask->ne[1] < n_q) {
        return false;
    }

    static const int min_kv = [] {
        const char * e = getenv("GGML_CUDA_FATTN_SPARSE_COMPACT_MIN_KV");
        return e ? atoi(e) : 8192;
    }();
    if (n_kv < min_kv) {
        return false;
    }
    // A capture cannot express a host read-back, and this path shapes its launches
    // from union lengths.  Decline rather than disabling graphs globally, which
    // costs generation the graphs it does use.
    {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return false;
        }
    }

    const int    n_head      = int(Q->ne[2]);
    const int    n_head_kv   = int(K->ne[2]);
    const int    DV          = int(V->ne[0]);
    const size_t row_bytes   = ggml_row_size(K->type, K->ne[0]);
    const size_t row_bytes_v = ggml_row_size(V->type, V->ne[0]);

    // The gather moves whole int4s and the output is copied as one flat run.
    if (row_bytes % sizeof(int4) || row_bytes_v % sizeof(int4) ||
        K->nb[1] % alignof(int4) || K->nb[2] % alignof(int4) ||
        V->nb[1] % alignof(int4) || V->nb[2] % alignof(int4) ||
        reinterpret_cast<uintptr_t>(K->data) % alignof(int4) ||
        reinterpret_cast<uintptr_t>(V->data) % alignof(int4)) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] != size_t(DV)*sizeof(float) ||
        dst->nb[2] != size_t(DV)*n_head*sizeof(float)) {
        return false;
    }

    const int words = (n_kv + 31) / 32;
    if (size_t(words) * sizeof(uint32_t) > 48*1024) {   // the union's bitmap is in shared memory
        return false;
    }

    const int          device = ctx.device;
    cudaStream_t       stream = ctx.stream();
    auto &             scratch = ctx.fattn_prep_scratch[device][ctx.curr_stream_no];
    const int          n_tiles = n_q / tile;
    const int64_t      s31 = mask->nb[1] / sizeof(half);
    const int64_t      s33 = mask->nb[3] / sizeof(half);

    // Selections, one padded list per mask row.
    ggml_cuda_pool_alloc<int32_t> indices(ctx.pool());
    indices.alloc(size_t(n_kv_max) * mask->ne[1] * mask->ne[3]);
    {
        const dim3 blocks_num(mask->ne[1], mask->ne[3], 1);
        const ggml_cuda_kernel_launch_params lp(blocks_num, dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, lp,
            (const half *) mask->data, indices.ptr, n_kv, n_kv_max, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    // Unions for every tile at once, so the lengths come back in one transfer
    // rather than one per group.
    const size_t need_idx = size_t(n_tiles) * n_kv * sizeof(int32_t);
    const size_t need_bm  = size_t(n_tiles) * words * sizeof(uint32_t);
    if (need_idx > scratch.union_idx_capacity) {
        if (scratch.union_idx) { CUDA_CHECK(cudaFree(scratch.union_idx)); }
        CUDA_CHECK(cudaMalloc(&scratch.union_idx, need_idx));
        scratch.union_idx_capacity = need_idx;
    }
    if (size_t(n_tiles) * sizeof(int32_t) > scratch.union_len_capacity) {
        if (scratch.union_len) { CUDA_CHECK(cudaFree(scratch.union_len)); }
        CUDA_CHECK(cudaMalloc(&scratch.union_len, size_t(n_tiles) * sizeof(int32_t)));
        scratch.union_len_capacity = size_t(n_tiles) * sizeof(int32_t);
    }
    if (need_bm > scratch.bitmap_capacity) {
        if (scratch.bitmap) { CUDA_CHECK(cudaFree(scratch.bitmap)); }
        if (scratch.prefix) { CUDA_CHECK(cudaFree(scratch.prefix)); }
        CUDA_CHECK(cudaMalloc(&scratch.bitmap, need_bm));
        CUDA_CHECK(cudaMalloc(&scratch.prefix, need_bm));
        scratch.bitmap_capacity = need_bm;
    }

    {
        fattn_stage_timer t("compact_union", device, n_q, stream);
        fattn_sparse_union<<<n_tiles, GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS,
                             size_t(words)*sizeof(uint32_t), stream>>>(
            indices.ptr, scratch.union_idx, scratch.union_len, scratch.bitmap, scratch.prefix,
            n_kv, n_kv_max, tile, n_q);
        CUDA_CHECK(cudaGetLastError());
    }

    // One synchronisation for the whole operation.  The launch geometry depends on
    // how many rows the widest union holds, and that is only known on the device.
    std::vector<int32_t> h_len(n_tiles);
    CUDA_CHECK(cudaMemcpyAsync(h_len.data(), scratch.union_len,
        h_len.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int union_max = 0;
    for (int b = 0; b < n_tiles; ++b) {
        if (h_len[b] <= 0 || h_len[b] > n_kv) {
            return false;   // nothing selected, or a length that cannot be right
        }
        union_max = std::max(union_max, h_len[b]);
    }
    const int union_n = std::min(GGML_PAD(union_max, 256), n_kv);
    // The only honest test of whether this is worth doing: the union that was
    // actually built.  An a-priori bound from the selection count per query is far
    // too pessimistic -- sixteen queries selecting 2051 positions each overlap so
    // heavily that their union is a third of that at long context.
    if (union_n * 2 >= n_kv) {
        return false;
    }

    // Tiles per launch, bounded by what the gathered cache may occupy.
    const int64_t budget_rows = 32768;
    const int group = std::max(1, std::min<int>(n_tiles, int(budget_rows / std::max(1, union_n))));

    const int    mask_rows = 64;   // kernels may read past the queries a tile holds
    const size_t need_gath = size_t(group) * union_n * n_head_kv * (row_bytes + row_bytes_v);
    const size_t need_cm   = size_t(group) * mask_rows * union_n * sizeof(half);
    const size_t need_out  = GGML_PAD(size_t(DV) * n_head * tile * group * sizeof(float), 128)
        + GGML_PAD(size_t(K->ne[0]) * union_n * n_head_kv * group * sizeof(half), 128)
        + GGML_PAD(size_t(DV)       * union_n * n_head_kv * group * sizeof(half), 128);

    if (need_gath > scratch.gathered_capacity) {
        if (scratch.gathered) { CUDA_CHECK(cudaFree(scratch.gathered)); }
        CUDA_CHECK(cudaMalloc(&scratch.gathered, need_gath));
        // Rows past a tile's union are masked out, but a stale quantisation scale
        // could still be a NaN that survives the mask.
        CUDA_CHECK(cudaMemsetAsync(scratch.gathered, 0, need_gath, stream));
        scratch.gathered_capacity = need_gath;
    }
    if (need_cm > scratch.cmask_capacity) {
        if (scratch.cmask) { CUDA_CHECK(cudaFree(scratch.cmask)); }
        CUDA_CHECK(cudaMalloc(&scratch.cmask, need_cm));
        scratch.cmask_capacity = need_cm;
    }
    if (need_out > scratch.out_capacity) {
        if (scratch.out) { CUDA_CHECK(cudaFree(scratch.out)); }
        CUDA_CHECK(cudaMalloc(&scratch.out, need_out));
        scratch.out_capacity = need_out;
    }

    char * gK = scratch.gathered;
    char * gV = gK + size_t(group) * union_n * n_head_kv * row_bytes;

    for (int t0 = 0; t0 < n_tiles; t0 += group) {
        const int nt = std::min(group, n_tiles - t0);

        {
            fattn_stage_timer t("compact_gather", device, n_q, stream);
            const dim3 grid(256, nt, 1);
            fattn_sparse_gather_rows<<<grid, 64, 0, stream>>>(
                (const char *) K->data, gK, (const char *) V->data, gV,
                scratch.union_idx + (size_t) t0 * n_kv, scratch.union_len + t0, n_kv,
                row_bytes, K->nb[1], K->nb[2], row_bytes_v, V->nb[1], V->nb[2],
                n_head_kv, union_n);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            fattn_stage_timer t("compact_mask", device, n_q, stream);
            const dim3 grid(nt, mask_rows, 1);
            fattn_sparse_compact_mask<<<grid, GGML_CUDA_FATTN_SPARSE_PREP_NTHREADS, 0, stream>>>(
                (const half *) mask->data, scratch.union_idx + (size_t) t0 * n_kv,
                scratch.union_len + t0, scratch.cmask, n_kv, t0*tile, tile, union_n,
                mask_rows, n_q - t0*tile, int(mask->ne[1]), s31);
            CUDA_CHECK(cudaGetLastError());
        }

        ggml_tensor Qc = *Q;
        Qc.ne[1] = tile; Qc.ne[3] = nt;
        Qc.nb[3] = size_t(tile) * Q->nb[1];
        Qc.data  = (char *) Q->data + size_t(t0) * tile * Q->nb[1];
        Qc.view_src = nullptr; Qc.buffer = nullptr;

        ggml_tensor Kc = *K;
        Kc.ne[1] = union_n; Kc.ne[2] = n_head_kv; Kc.ne[3] = nt;
        Kc.nb[1] = row_bytes;
        Kc.nb[2] = size_t(union_n) * row_bytes;
        Kc.nb[3] = size_t(n_head_kv) * union_n * row_bytes;
        Kc.data  = gK; Kc.view_src = nullptr; Kc.buffer = nullptr;

        ggml_tensor Vc = *V;
        Vc.ne[1] = union_n; Vc.ne[2] = n_head_kv; Vc.ne[3] = nt;
        Vc.nb[1] = row_bytes_v;
        Vc.nb[2] = size_t(union_n) * row_bytes_v;
        Vc.nb[3] = size_t(n_head_kv) * union_n * row_bytes_v;
        Vc.data  = gV; Vc.view_src = nullptr; Vc.buffer = nullptr;

        ggml_tensor Mc = *mask;
        Mc.ne[0] = union_n; Mc.ne[1] = mask_rows; Mc.ne[2] = 1; Mc.ne[3] = nt;
        Mc.nb[0] = sizeof(half);
        Mc.nb[1] = size_t(union_n) * sizeof(half);
        Mc.nb[2] = size_t(mask_rows) * Mc.nb[1];
        Mc.nb[3] = Mc.nb[2];
        Mc.data  = scratch.cmask; Mc.view_src = nullptr; Mc.buffer = nullptr;

        ggml_tensor Dc = *dst;
        Dc.ne[0] = DV; Dc.ne[1] = n_head; Dc.ne[2] = tile; Dc.ne[3] = nt;
        Dc.nb[0] = sizeof(float);
        Dc.nb[1] = Dc.ne[0] * Dc.nb[0];
        Dc.nb[2] = Dc.ne[1] * Dc.nb[1];
        Dc.nb[3] = Dc.ne[2] * Dc.nb[2];
        Dc.data  = scratch.out; Dc.view_src = nullptr; Dc.buffer = nullptr;
        Dc.src[0] = &Qc; Dc.src[1] = &Kc; Dc.src[2] = &Vc; Dc.src[3] = &Mc; Dc.src[4] = nullptr;
        ggml_set_op_params_i32(&Dc, 4, 0);   // keeps the inner dispatch dense

        {
            fattn_stage_timer t("compact_attn", device, n_q, stream);
            g_fattn_in_compact = true;
            ggml_cuda_flash_attn_ext(ctx, &Dc);
            g_fattn_in_compact = false;
            CUDA_CHECK(cudaGetLastError());
        }

        // The compact output is contiguous over (dim, head, query) for this group's
        // queries, and so is the destination, so one run copies it.
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size_t(t0) * tile * dst->nb[2],
            scratch.out, size_t(nt) * tile * n_head * DV * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    static bool announced = false;
    if (!announced) {
        announced = true;
        GGML_LOG_WARN("fattn compact path active: n_kv=%d union_n=%d tiles=%d group=%d min_kv=%d\n",
                      n_kv, union_n, n_tiles, group, min_kv);
    }
    return true;
#endif
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    static const bool compact = ggml_env_flag_enabled("GGML_CUDA_FATTN_SPARSE_COMPACT");
    if (compact && !g_fattn_in_compact && ggml_cuda_flash_attn_ext_compact_run(ctx, dst)) {
        return;
    }

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
