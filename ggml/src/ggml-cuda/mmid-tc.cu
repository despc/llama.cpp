#include "mmid-tc.cuh"
#include "mmid.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#endif

// Volta has no int8 MMA, so MMQ runs on DP4A there. With 512 experts a prompt microbatch gives each expert ~10
// tokens. This kernel stages each row's weights a whole block group at a time in shared memory, dequantizes 64 of
// them per step to f16 and multiplies on the tensor cores (WMMA, f32 accumulation). Not bit-identical to MMQ: the
// activations are f16 instead of q8_1. Per call at Flash-Next shapes (512 experts, 512 tokens): Q4_K 1600 -> 1540 us,
// Q5_1 2690 -> 1860, Q8_0 2770 -> 2290.

#define MMID_TC_BN     16   // tokens per wmma column tile
#define MMID_TC_NC     2    // column tiles per pass
#define MMID_TC_BK     64   // K per step
#define MMID_TC_PAD    8

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Integer-to-half without conversion instructions (I2F/F2F run at a quarter rate on Volta): a value q < 1024 in the
// low mantissa bits of 0x6400 is the half 1024 + q, and subtracting 1024 is exact.
static __device__ __forceinline__ half2 mmid_tc_u2h(const uint32_t q2) {   // q2: two 16-bit lanes, each < 1024
    const uint32_t h = q2 | 0x64006400u;
    return __hsub2(*(const half2 *) &h, __half2half2(__ushort_as_half(0x6400)));
}

// bytes b and b+1 of w into the two 16-bit lanes
static __device__ __forceinline__ uint32_t mmid_tc_pair(const uint32_t w, const int b) {
    return __byte_perm(w, 0, b == 0 ? 0x4140 : 0x4342);
}

// Weights come in two stages so the next step's loads can be in flight during the current step's MMA:
// load() reads the raw quantized bytes of 32 consecutive weights of one row (element k, a multiple of 32) into
// registers, store() dequantizes them to 16 half2 in shared memory.
template <ggml_type type> struct mmid_tc_w;

template <> struct mmid_tc_w<GGML_TYPE_Q8_0> {
    uint32_t w[9]; uint32_t d; int sh;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k) {
        const block_q8_0 * b = (const block_q8_0 *) row + k/QK8_0;
        // blocks are 34 bytes, so qs is only 2-byte aligned: read the 9 words around it and shift
        const uintptr_t addr = (uintptr_t) b->qs;
        const uint32_t * p = (const uint32_t *) (addr & ~uintptr_t(3));
        sh = 8*(addr & 3);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            w[i] = p[i];
        }
        d = *(const uint16_t *) &b->d;
    }
    __device__ __forceinline__ void store(half2 * __restrict__ dst) const {
        const half2 d2 = __half2half2(__ushort_as_half((unsigned short) d));
        const half2 off = __half2half2(__float2half(128.0f));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t v = __funnelshift_r(w[i], w[i + 1], sh) ^ 0x80808080u;   // qs[4i .. 4i+3] + 128
            dst[2*i + 0] = __hmul2(__hsub2(mmid_tc_u2h(mmid_tc_pair(v, 0)), off), d2);
            dst[2*i + 1] = __hmul2(__hsub2(mmid_tc_u2h(mmid_tc_pair(v, 2)), off), d2);
        }
    }
};

template <> struct mmid_tc_w<GGML_TYPE_Q5_1> {
    uint2 r0, qa, qb;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k) {
        const block_q5_1 * b = (const block_q5_1 *) row + k/QK5_1;
        r0 = *(const uint2 *) b;                  // dm, qh
        qa = *(const uint2 *) b->qs;              // qs[0..7]
        qb = *((const uint2 *) b->qs + 1);        // qs[8..15]
    }
    __device__ __forceinline__ void store(half2 * __restrict__ dst) const {
        half2 dmh; memcpy(&dmh, &r0.x, 4);
        const half2 d2 = __half2half2(__low2half(dmh));
        const half2 m2 = __half2half2(__high2half(dmh));
        const uint32_t qh = r0.y;
        const uint32_t qw[4] = {qa.x, qa.y, qb.x, qb.y};
        // values j and j+16 share byte j; the 5th bits of j and j+1 go to bit 4 of each 16-bit lane
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t p  = mmid_tc_pair(qw[i/2], 2*(i%2));
            const uint32_t hl = ((qh >> (2*i))      & 1) << 4 | ((qh >> (2*i + 1))      & 1) << 20;
            const uint32_t hh = ((qh >> (2*i + 16)) & 1) << 4 | ((qh >> (2*i + 17))     & 1) << 20;
            dst[i]     = __hfma2(mmid_tc_u2h(( p       & 0x000F000Fu) | hl), d2, m2);
            dst[8 + i] = __hfma2(mmid_tc_u2h(((p >> 4) & 0x000F000Fu) | hh), d2, m2);
        }
    }
};

// byte j (0..11) of the three scale words, without indexing a local array
static __device__ __forceinline__ int mmid_tc_sbyte(const uint32_t w0, const uint32_t w1, const uint32_t w2, const int j) {
    const uint32_t w = j < 4 ? w0 : (j < 8 ? w1 : w2);
    return (w >> (8*(j % 4))) & 0xFF;
}

static __device__ __forceinline__ void mmid_tc_scale_min_w(const int j, const uint32_t w0, const uint32_t w1, const uint32_t w2, int & d, int & m) {
    if (j < 4) {
        d = mmid_tc_sbyte(w0, w1, w2, j)     & 63;
        m = mmid_tc_sbyte(w0, w1, w2, j + 4) & 63;
    } else {
        d = (mmid_tc_sbyte(w0, w1, w2, j + 4) & 0xF) | ((mmid_tc_sbyte(w0, w1, w2, j - 4) >> 6) << 4);
        m = (mmid_tc_sbyte(w0, w1, w2, j + 4) >>  4) | ((mmid_tc_sbyte(w0, w1, w2, j)     >> 6) << 4);
    }
}

template <> struct mmid_tc_w<GGML_TYPE_Q4_K> {
    uint4 head, q0, q1; int s;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k) {
        const block_q4_K * b = (const block_q4_K *) row + k/QK_K;
        s = (k % QK_K)/32;                        // 64-value chunk s/2, low or high nibble
        head = *(const uint4 *) b;                // dm, then the 12 scale bytes
        const uint4 * q4 = (const uint4 *) (b->qs + 32*(s/2));
        q0 = q4[0];
        q1 = q4[1];
    }
    __device__ __forceinline__ void store(half2 * __restrict__ dst) const {
        half2 dmh; memcpy(&dmh, &head.x, 4);
        const float2 dm = __half22float2(dmh);
        int sc, mn;
        mmid_tc_scale_min_w(s, head.y, head.z, head.w, sc, mn);
        const half2 d2 = __half2half2(__float2half(dm.x*sc));
        const half2 m2 = __half2half2(__float2half(-dm.y*mn));
        const int shift = 4*(s % 2);
        const uint32_t ww[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t p = (mmid_tc_pair(ww[i/2], 2*(i%2)) >> shift) & 0x000F000Fu;
            dst[i] = __hfma2(mmid_tc_u2h(p), d2, m2);
        }
    }
};

template <> struct mmid_tc_w<GGML_TYPE_Q5_K> {
    uint4 head, q0, q1, h0, h1; int s;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k) {
        const block_q5_K * b = (const block_q5_K *) row + k/QK_K;
        s = (k % QK_K)/32;
        head = *(const uint4 *) b;
        const uint4 * q4 = (const uint4 *) (b->qs + 32*(s/2));
        const uint4 * h4 = (const uint4 *) b->qh;
        q0 = q4[0]; q1 = q4[1];
        h0 = h4[0]; h1 = h4[1];
    }
    __device__ __forceinline__ void store(half2 * __restrict__ dst) const {
        half2 dmh; memcpy(&dmh, &head.x, 4);
        const float2 dm = __half22float2(dmh);
        int sc, mn;
        mmid_tc_scale_min_w(s, head.y, head.z, head.w, sc, mn);
        const half2 d2 = __half2half2(__float2half(dm.x*sc));
        const half2 m2 = __half2half2(__float2half(-dm.y*mn));
        const int shift = 4*(s % 2);
        const uint32_t ww[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const uint32_t hh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t p  = (mmid_tc_pair(ww[i/2], 2*(i%2)) >> shift) & 0x000F000Fu;
            const uint32_t ph = ((mmid_tc_pair(hh[i/2], 2*(i%2)) >> s) & 0x00010001u) << 4;
            dst[i] = __hfma2(mmid_tc_u2h(p | ph), d2, m2);
        }
    }
};

// Raw chunk per row and step: whole super-blocks (k-quants) or 4 blocks, contiguous in the row, so a block reads
// 128 contiguous runs of 96-176 bytes per step instead of 128 scattered 32-byte pieces.
template <ggml_type type> struct mmid_tc_chunk;
template <> struct mmid_tc_chunk<GGML_TYPE_Q4_K> { static constexpr int rk = 256; static constexpr int rb = 144; };
template <> struct mmid_tc_chunk<GGML_TYPE_Q5_K> { static constexpr int rk = 256; static constexpr int rb = 176; };
template <> struct mmid_tc_chunk<GGML_TYPE_Q5_1> { static constexpr int rk = 128; static constexpr int rb =  96; };
template <> struct mmid_tc_chunk<GGML_TYPE_Q8_0> { static constexpr int rk = 128; static constexpr int rb = 136; };

template <ggml_type type, int nwarps>
__launch_bounds__(nwarps*WARP_SIZE, 4)
static __global__ void mul_mat_id_tc(
        const char * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const int K, const size_t nb01, const size_t nb02, const int64_t s11, const int64_t s1, const int64_t s2,
        const int n_expert_used) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMPERE_MMA_AVAILABLE)
    using namespace nvcuda;
    constexpr int NT   = nwarps*WARP_SIZE;
    constexpr int BM   = 16*nwarps;
    constexpr int LDA  = MMID_TC_BK + MMID_TC_PAD;
    constexpr int NCOL = MMID_TC_NC*MMID_TC_BN;
    constexpr int RK   = mmid_tc_chunk<type>::rk;
    constexpr int RB   = mmid_tc_chunk<type>::rb;
    constexpr int RU   = RB/8;                       // uint2 per row chunk
    constexpr int NU   = (BM*RU + NT - 1)/NT;        // uint2 per thread per chunk
    constexpr int NBT  = (NCOL*(MMID_TC_BK/4) + NT - 1)/NT;
    static_assert(BM*MMID_TC_BK/32 == NT, "one 32-value run per thread");
    static_assert(RK % MMID_TC_BK == 0 && RB % 8 == 0, "chunk layout");
    static_assert(nwarps*MMID_TC_BN*MMID_TC_BN*sizeof(float) <= BM*LDA*sizeof(half), "Cs aliases As");

    __shared__ __align__(16) char  raw[BM*RB];
    __shared__ __align__(32) half  As[BM*LDA];
    __shared__ __align__(32) half  Bs[NCOL*LDA];
    __shared__ int src_row[NCOL];
    float * Cs = (float *) As;   // output staging, after the K loop

    const int expert = blockIdx.y;
    const int row0   = blockIdx.x*BM;
    const int beg    = expert_bounds[expert];
    const int cnt    = expert_bounds[expert + 1] - beg;
    if (cnt == 0) {
        return;
    }

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const char * xe = x + expert*nb02 + row0*nb01;
    const char * rrow = raw + (tid/2)*RB;
    half2 * arun = (half2 *) (As + (tid/2)*LDA + (tid % 2)*32);

    uint2 ru[NU];
    auto load_chunk = [&](const int kc) {
        const size_t coff = size_t(kc/RK)*RB;
#pragma unroll
        for (int i = 0; i < NU; ++i) {
            const int u = tid + i*NT;
            if (u < BM*RU) {
                ru[i] = *(const uint2 *) (xe + (u/RU)*nb01 + coff + 8*(u % RU));
            }
        }
    };
    auto store_chunk = [&]() {
#pragma unroll
        for (int i = 0; i < NU; ++i) {
            const int u = tid + i*NT;
            if (u < BM*RU) {
                *(uint2 *) (raw + (u/RU)*RB + 8*(u % RU)) = ru[i];
            }
        }
    };

    for (int c0 = 0; c0 < cnt; c0 += NCOL) {
        const int ncol  = min(NCOL, cnt - c0);
        const int ntile = (ncol + MMID_TC_BN - 1)/MMID_TC_BN;
        __syncthreads();   // previous pass done with src_row, raw and Cs
        if (tid < NCOL) {
            src_row[tid] = tid < ncol ? ids_src1[beg + c0 + tid] : -1;
        }
        load_chunk(0);
        store_chunk();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MMID_TC_NC];
#pragma unroll
        for (int ct = 0; ct < MMID_TC_NC; ++ct) {
            wmma::fill_fragment(acc[ct], 0.0f);
        }
        __syncthreads();   // src_row and the first chunk visible

        for (int kc = 0; kc < K; kc += RK) {
            const bool more = kc + RK < K;
            if (more) {
                load_chunk(kc + RK);   // in flight during this chunk's steps
            }
#pragma unroll 1
            for (int ks = 0; ks < RK; ks += MMID_TC_BK) {
                const int k0 = kc + ks;
                float4 br[NBT];
#pragma unroll
                for (int i = 0; i < NBT; ++i) {
                    const int idx = tid + i*NT;
                    const int c   = idx / (MMID_TC_BK/4);
                    const int kk  = 4*(idx % (MMID_TC_BK/4));
                    br[i] = c < ncol ? *(const float4 *) (y + src_row[c]*s11 + k0 + kk) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                {
                    mmid_tc_w<type> wr;
                    wr.load(rrow, ks + (tid % 2)*32);
                    wr.store(arun);
                }
#pragma unroll
                for (int i = 0; i < NBT; ++i) {
                    const int idx = tid + i*NT;
                    const int c   = idx / (MMID_TC_BK/4);
                    const int kk  = 4*(idx % (MMID_TC_BK/4));
                    if (c < ntile*MMID_TC_BN) {
                        half2 * bp = (half2 *) (Bs + c*LDA + kk);
                        bp[0] = __floats2half2_rn(br[i].x, br[i].y);
                        bp[1] = __floats2half2_rn(br[i].z, br[i].w);
                    }
                }
                __syncthreads();

#pragma unroll
                for (int kk = 0; kk < MMID_TC_BK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                    wmma::load_matrix_sync(a, As + warp*16*LDA + kk, LDA);
#pragma unroll
                    for (int ct = 0; ct < MMID_TC_NC; ++ct) {
                        if (ct < ntile) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                            wmma::load_matrix_sync(b, Bs + ct*MMID_TC_BN*LDA + kk, LDA);
                            wmma::mma_sync(acc[ct], a, b, acc[ct]);
                        }
                    }
                }
                __syncthreads();   // As, Bs and raw reads done
            }
            if (more) {
                store_chunk();
                __syncthreads();
            }
        }

#pragma unroll
        for (int ct = 0; ct < MMID_TC_NC; ++ct) {
            if (ct >= ntile) {
                break;
            }
            float * cw = Cs + warp*MMID_TC_BN*MMID_TC_BN;
            wmma::store_matrix_sync(cw, acc[ct], MMID_TC_BN, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16*MMID_TC_BN; idx += WARP_SIZE) {
                const int i = idx % 16;
                const int j = idx / 16;
                const int c = ct*MMID_TC_BN + j;
                if (c < ncol) {
                    const int d  = ids_dst[beg + c0 + c];
                    const int it = d / n_expert_used;
                    const int ie = d % n_expert_used;
                    dst[it*s2 + ie*s1 + row0 + warp*16 + i] = cw[i*MMID_TC_BN + j];
                }
            }
            __syncwarp();
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ids_src1, ids_dst, expert_bounds, K, nb01, nb02, s11, s1, s2, n_expert_used);
    NO_DEVICE_CODE;
#endif
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

bool ggml_cuda_mmid_tc_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                 const ggml_tensor * dst, const int cc) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(src0, src1, ids, dst, cc);
    return false;
#else
    // GGML_CUDA_MMID_TC=0 keeps MMQ
    static const bool enabled = !getenv("GGML_CUDA_MMID_TC") || atoi(getenv("GGML_CUDA_MMID_TC")) != 0;
    if (!enabled || cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            break;
        default:
            return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
        src1->ne[2] >= 64 &&                                    // prompt batches; decode stays on MMVQ
        src0->ne[0] % (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K ? 256 : 128) == 0 && src0->ne[1] % 64 == 0 &&
        (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K ? true : src0->ne[0] % QK_K == 0) &&
        src0->ne[3] == 1 && src1->ne[3] == 1 &&
        src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src1->nb[1] % (2*sizeof(float)) == 0 && src1->nb[2] % src1->nb[1] == 0 && dst->nb[2] % dst->nb[1] == 0 &&
        ids->nb[0] == sizeof(int32_t);
#endif
}

void ggml_cuda_mul_mat_id_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    GGML_ABORT("not supported");
#else
    cudaStream_t stream = ctx.stream();

    const int64_t n_experts     = src0->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = src1->ne[2];
    const int64_t ne_get_rows   = n_tokens*n_expert_used;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = src1->nb[2] / src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    constexpr int nwarps = 4;
    const dim3 grid(src0->ne[1] / (16*nwarps), n_experts, 1);
    const dim3 block(nwarps*WARP_SIZE, 1, 1);

    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s1  = dst->nb[1]  / sizeof(float);
    const int64_t s2  = dst->nb[2]  / sizeof(float);

#define MMID_TC_LAUNCH(T) \
    mul_mat_id_tc<T, nwarps><<<grid, block, 0, stream>>>((const char *) src0->data, (const float *) src1->data, (float *) dst->data, \
        ids_src1.get(), ids_dst.get(), expert_bounds.get(), int(src0->ne[0]), src0->nb[1], src0->nb[2], s11, s1, s2, int(n_expert_used))

    switch (src0->type) {
        case GGML_TYPE_Q8_0: MMID_TC_LAUNCH(GGML_TYPE_Q8_0); break;
        case GGML_TYPE_Q5_1: MMID_TC_LAUNCH(GGML_TYPE_Q5_1); break;
        case GGML_TYPE_Q4_K: MMID_TC_LAUNCH(GGML_TYPE_Q4_K); break;
        case GGML_TYPE_Q5_K: MMID_TC_LAUNCH(GGML_TYPE_Q5_K); break;
        default: GGML_ABORT("unsupported type");
    }
#undef MMID_TC_LAUNCH
    CUDA_CHECK(cudaGetLastError());
#endif
}
