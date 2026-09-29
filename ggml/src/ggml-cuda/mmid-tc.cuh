#pragma once

#include "common.cuh"

// Prompt-batch MUL_MAT_ID on Volta's FP16 tensor cores: the expert weights are dequantized to f16 tile by tile,
// the activations are converted to f16, products are accumulated in f32. On by default, GGML_CUDA_MMID_TC=0
// keeps MMQ. See mmid-tc.cu.
bool ggml_cuda_mmid_tc_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                 const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst);
