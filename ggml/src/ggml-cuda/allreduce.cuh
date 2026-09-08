#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Native-type mixed-runtime wire. Version 5 cannot load the old compressed ABI.
static constexpr uint32_t GGML_CUDA_MIXED_AR_ABI_VERSION = 5;
static constexpr size_t GGML_CUDA_MIXED_AR_SLOTS = 2;
static constexpr size_t GGML_CUDA_MIXED_AR_RANK_BYTES = 64 * 1024 * 1024;
static constexpr size_t GGML_CUDA_MIXED_AR_BLOCKS = 8;
// Ranks a single mixed AllReduce group can hold; sizes the per-element scratch
// the butterfly tree is reduced in.
static constexpr int GGML_CUDA_MIXED_AR_MAX_RANKS = GGML_CUDA_MAX_DEVICES;
static constexpr size_t GGML_CUDA_MIXED_AR_SIGNAL_STRIDE = 64;

struct ggml_cuda_mixed_ar_group_config {
    uint32_t abi_version;
    ggml_backend_t * backends;
    const int * ranks;
    size_t n_backends;
    size_t n_ranks;
    void * shared_host;
    size_t shared_bytes;
    size_t data_bytes;
    size_t slots;
    size_t rank_bytes;
    size_t blocks;
    size_t signal_stride;
};

using ggml_cuda_mixed_ar_group_init_t = void * (*)(const ggml_cuda_mixed_ar_group_config *);
using ggml_cuda_mixed_ar_group_free_t = void (*)(void *);
using ggml_cuda_mixed_ar_group_prepare_t = bool (*)(void *, size_t);
using ggml_cuda_mixed_ar_group_enqueue_t = bool (*)(void *, ggml_tensor **, size_t, uint32_t);

void * ggml_cuda_mixed_ar_group_init(const ggml_cuda_mixed_ar_group_config * config);
void ggml_cuda_mixed_ar_group_free(void * context);
bool ggml_cuda_mixed_ar_group_prepare(void * context, size_t slot);
bool ggml_cuda_mixed_ar_group_enqueue(void * context, ggml_tensor ** tensors, size_t slot, uint32_t token);

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);
