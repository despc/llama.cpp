#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Mixed-runtime AllReduce.  Unlike ggml_cuda_ar_pipeline, one group contains
// only devices owned by a single backend registry / CUDA runtime.  Multiple
// groups register the same ordinary host allocation and launch their local
// ranks through function pointers resolved from their own backend DSO.
static constexpr uint32_t GGML_CUDA_MIXED_AR_ABI_VERSION = 4;
static constexpr size_t   GGML_CUDA_MIXED_AR_SLOTS       = 2;
// The tensor-parallel Qwen prefill graph reduces hidden-state buffers of
// roughly ubatch * n_embd * sizeof(BF16).  A 1 MiB lane only covered decode
// and forced normal 2K-token prefill (about 20 MiB for n_embd=5120) through
// the meta-backend fallback.  Keep enough pinned staging space for that path.
static constexpr size_t   GGML_CUDA_MIXED_AR_RANK_BYTES  = 64 * 1024 * 1024;
// The hierarchical kernels process large prefill buffers in stripes. More
// blocks reduce the serial qblock loop in the fused local+cross path; the
// mapped-host token arrays are sized from this constant as well. 32 was the
// best value while publication and consumption were separated by a barrier;
// with the streamed duplex kernel 64 wins, because each block runs its own
// independent overlap of the two link directions.
static constexpr size_t   GGML_CUDA_MIXED_AR_BLOCKS      = 64;
// Grid width for the small flat path, which decode runs on.  Those tensors are
// a few tens of KiB, so the kernel is latency-bound: every extra block adds one
// more arrival token to publish and poll across the link.  It is kept separate
// from the striped hierarchical width above, which wants the opposite.
static constexpr size_t   GGML_CUDA_MIXED_AR_FLAT_BLOCKS = 4;
// Values per INT8 scale on the cross-runtime wire.  The transfer tile stays at
// 4096 because a 256-thread block moves it as one 16-byte-per-thread vector,
// but the scale can be finer: a shorter run of values under one scale means
// less quantization error, and each extra FP32 scale costs 4 bytes per group.
// Measured on a fixed corpus, perplexity against the BF16 wire's 4.0647:
// 4096 -> 4.0948, 1024 -> 4.0720, 512 -> 4.0688, 256 -> 4.0661.  Prefill stays
// within noise across all of them (1155-1165 tok/s), so 256 is chosen: it costs
// 1.6% more wire bytes and brings the INT8 path back to +0.03% perplexity.
// Only the streamed kernel implements this; the coarse fused and plain
// hierarchical kernels still carry one scale per 4096-value tile.
static constexpr int      GGML_CUDA_MIXED_AR_SCALE_VALUES = 256;

static constexpr size_t   GGML_CUDA_MIXED_AR_SIGNAL_STRIDE = 64;

// Optional profiling records written by the hierarchical kernel into the
// shared mapped allocation.  globaltimer is expressed in nanoseconds on the
// supported CUDA architectures, so the host does not need to guess the
// effective SM clock while the cards are boosting/throttling independently.
struct ggml_cuda_mixed_ar_trace_record {
    uint64_t begin_ns;
    uint64_t publish_ns;
    uint64_t peer_ready_ns;
    uint64_t done_ns;
};

struct ggml_cuda_mixed_ar_rank_profile {
    int   rank;
    float local_ms;
    float total_ms;
    float publish_ms;
    float wait_ms;
    float reduce_ms;
};

struct ggml_cuda_mixed_ar_group_profile {
    size_t n_entries;
    ggml_cuda_mixed_ar_rank_profile entries[GGML_CUDA_MAX_DEVICES];
};

struct ggml_cuda_mixed_ar_group_config {
    uint32_t         abi_version;
    ggml_backend_t * backends;       // local backends, all from this registry
    const int      * ranks;          // corresponding global ranks
    size_t           n_backends;
    size_t           n_ranks;
    void           * shared_host;
    size_t           shared_bytes;
    size_t           data_bytes;
    size_t           arrival_offset;
    size_t           departure_offset;
    size_t           trace_offset;
    size_t           bf16_threshold;
    bool             profile;
    bool             device_slots;
    bool             hierarchical;
    int              leader_rank;
    int              peer_leader_rank;
};

using ggml_cuda_mixed_ar_group_init_t = void * (*)(const ggml_cuda_mixed_ar_group_config * config);
using ggml_cuda_mixed_ar_group_free_t = void   (*)(void * context);
using ggml_cuda_mixed_ar_group_prepare_t = bool (*)(void * context, size_t slot);
using ggml_cuda_mixed_ar_group_enqueue_t = bool (*)(
    void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16);
using ggml_cuda_mixed_ar_group_enqueue_hier_t = bool (*)(
    void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16);
using ggml_cuda_mixed_ar_group_profile_collect_t = bool (*)(
    void * context, ggml_cuda_mixed_ar_group_profile * profile);

void * ggml_cuda_mixed_ar_group_init(const ggml_cuda_mixed_ar_group_config * config);
void   ggml_cuda_mixed_ar_group_free(void * context);
bool   ggml_cuda_mixed_ar_group_prepare(void * context, size_t slot);
bool   ggml_cuda_mixed_ar_group_enqueue(
    void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16);
bool   ggml_cuda_mixed_ar_group_enqueue_hier(
    void * context, ggml_tensor ** tensors, size_t slot, int token, bool use_bf16);
bool   ggml_cuda_mixed_ar_group_profile_collect(
    void * context, ggml_cuda_mixed_ar_group_profile * profile);

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
