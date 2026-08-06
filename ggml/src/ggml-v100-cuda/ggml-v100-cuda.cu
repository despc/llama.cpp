// V100 CUDA backend - the stock CUDA backend built against an isolated driver stack.
//
// The host carries two NVIDIA kernel modules: the open "nvidia" driver for the
// consumer GPUs, and a renamed proprietary "nvidia_v100" driver for the Tesla V100.
// Both ship a libcuda with the same SONAME, so ld.so would keep only one copy.
// tools/rename_soname.py makes libcv* copies with distinct SONAMEs, and this
// backend links against those. Both stacks then live in one process.
//
// loader/v100_redirect.c must be LD_PRELOADed: it points open() calls coming from
// the libcv* libraries at /dev/nvidia-v100-*, and gives their UVM socket a private
// name. See loader/README.md.
//
// This file only renames the backend and its entry points, then includes the
// original ggml-cuda.cu unchanged.

// Override backend names BEFORE including ggml-cuda.h (which has #ifndef guards).
#undef  GGML_CUDA_NAME
#undef  GGML_CUBLAS_NAME
#define GGML_CUDA_NAME   "V100_CUDA"
#define GGML_CUBLAS_NAME "V100_cuBLAS"

// The registration entry points would clash with the system backend at link time.
#define ggml_backend_cuda_reg              ggml_backend_v100_cuda_reg
#define ggml_backend_cuda_init             ggml_backend_v100_cuda_init
#define ggml_backend_cuda_get_device_count ggml_backend_v100_cuda_get_device_count

// GGML_BACKEND_DL comes from a -D flag. Undefining it stops the included
// ggml-cuda.cu from emitting its own ggml_backend_init; ours is below.
#undef GGML_BACKEND_DL

#include <unistd.h>

#define V100_LIBCUDA_PATH "/opt/nvidia-v100/lib/libcvda.so.1"

#include "../ggml-cuda/ggml-cuda.cu"

#undef  GGML_BACKEND_DL
#define GGML_BACKEND_DL 1

extern "C" GGML_BACKEND_API int ggml_backend_score(void);
int ggml_backend_score(void) {
    return access(V100_LIBCUDA_PATH, F_OK) == 0 ? 50 : 0;
}

extern "C" GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);
ggml_backend_reg_t ggml_backend_init(void) {
    return ggml_backend_v100_cuda_reg();
}
