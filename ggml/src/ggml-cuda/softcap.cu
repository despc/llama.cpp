#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}

// scale -> sigmoid -> scale and scale -> silu, the same float operations in the same order as the three (two) kernels
template <bool silu>
static __global__ void scale_act_scale_f32(const float * x, float * dst, const float scale, const float scale2, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    const float t = scale * x[i];
    if constexpr (silu) {
        dst[i] = t / (1.0f + expf(-t));
    } else {
        dst[i] = scale2 * (1.0f / (1.0f + expf(-t)));
    }
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE (scale2 != nullptr) or GGML_OP_SCALE + GGML_UNARY_OP_SILU
void ggml_cuda_op_scale_act_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale, ggml_tensor * scale2) {
    const ggml_tensor * src0 = scale->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float s1;
    float s2 = 1.0f;
    memcpy(&s1, (float *) scale->op_params + 0, sizeof(float));
    if (scale2) {
        memcpy(&s2, (float *) scale2->op_params + 0, sizeof(float));
    }

    const int k = ggml_nelements(src0);
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    if (scale2) {
        ggml_cuda_kernel_launch(scale_act_scale_f32<false>, launch_params, (const float *) src0->data, (float *) dst->data, s1, s2, k);
    } else {
        ggml_cuda_kernel_launch(scale_act_scale_f32<true>,  launch_params, (const float *) src0->data, (float *) dst->data, s1, s2, k);
    }
}
