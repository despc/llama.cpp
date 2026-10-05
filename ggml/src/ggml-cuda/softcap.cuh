#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);

// scale -> sigmoid -> scale (scale2 set) or scale -> silu (scale2 == nullptr), bit-identical to the separate ops
void ggml_cuda_op_scale_act_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale, ggml_tensor * scale2);
