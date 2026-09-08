#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Native-type mixed-runtime wire.  Version 6 adds streaming publication and
// reduce-scatter, which use signal words version 5 did not, and moves the choice
// of algorithm into the config so both runtimes cannot pick different ones.
// Matching buffer sizes never implied a matching protocol; a rank running flat
// while its peer runs reduce-scatter would wait forever for a phase the other
// never signals.  Version 7 adds the probe rendezvous: the two runtimes measure
// the link in separate libraries, so without a meeting point each of them times
// a link the other is not using, which is not the link the collective runs on.
static constexpr uint32_t GGML_CUDA_MIXED_AR_ABI_VERSION = 8;
static constexpr size_t GGML_CUDA_MIXED_AR_SLOTS = 2;
static constexpr size_t GGML_CUDA_MIXED_AR_RANK_BYTES = 64 * 1024 * 1024;
// Signal-slot stride and the largest grid a group may launch.  The grid itself
// is negotiated (config->blocks) and defaults to 8, which is what every figure in
// the journal up to now was measured at.  It is a stride, not a count: peers
// index each other's signals by it, so it must be identical in both runtimes
// whatever grid they end up running.
static constexpr size_t GGML_CUDA_MIXED_AR_BLOCKS = 64;
// Ranks a single mixed AllReduce group can hold; sizes the per-element scratch
// the butterfly tree is reduced in.
static constexpr int GGML_CUDA_MIXED_AR_MAX_RANKS = GGML_CUDA_MAX_DEVICES;
static constexpr size_t GGML_CUDA_MIXED_AR_SIGNAL_STRIDE = 64;

// Words within one (rank, block) signal slot.  Every algorithm and phase gets its
// own: they carry different things -- a token here, a step count there -- and a
// step count of 3 left by one algorithm is indistinguishable from another's token
// 3 when a slot is reused.  Sixteen words fit in the stride; five are used.
enum ggml_cuda_mixed_ar_signal_word {
    GGML_CUDA_MIXED_AR_SIG_FLAT_ARRIVAL = 0,
    GGML_CUDA_MIXED_AR_SIG_STREAM_STEPS = 1,   // how many steps published
    GGML_CUDA_MIXED_AR_SIG_STREAM_TOKEN = 2,   // validates the count above
    GGML_CUDA_MIXED_AR_SIG_RS_PUBLISHED = 3,
    GGML_CUDA_MIXED_AR_SIG_RS_REDUCED = 4,
    GGML_CUDA_MIXED_AR_SIG_PIPE_TOKEN = 5,   // validates the two counts below
    GGML_CUDA_MIXED_AR_SIG_PIPE_PUB   = 6,   // chunks published, per block
    GGML_CUDA_MIXED_AR_SIG_PIPE_RED   = 7,   // chunks reduced, per block
};

// Which kernel a group runs.  Chosen once, on the host, and handed to every rank
// through the config: each runtime deciding for itself from its own environment
// and its own build is how two of them end up in different protocols.
enum ggml_cuda_mixed_ar_algo {
    GGML_CUDA_MIXED_AR_ALGO_FLAT = 0,
    GGML_CUDA_MIXED_AR_ALGO_STREAM = 1,
    GGML_CUDA_MIXED_AR_ALGO_RS = 2,
};

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
    size_t blocks;          // the grid to launch; <= GGML_CUDA_MIXED_AR_BLOCKS
    size_t signal_stride;
    // Negotiated once for the whole communicator, not re-derived per runtime.
    // Every rank picks the algorithm from these and the tensor's size, so the
    // choice is a pure function of values all of them were handed.
    uint64_t stream_min_bytes;  // 0 disables streaming
    uint64_t rs_min_bytes;      // 0 disables reduce-scatter
    uint32_t stream_chunk;
    uint32_t pipe_chunks;        // 0 disables the pipelined reduce-scatter
    // Relative share of the reduction each rank owns, in rank order.  Integers,
    // so every rank derives identical boundaries from them; all ones is an even
    // split.  This is not the weight split: it decides who reduces an element,
    // never which values are summed, so it cannot change a result.
    uint32_t shard_weight[GGML_CUDA_MIXED_AR_MAX_RANKS];
    // Where the runtimes meet to start a timed region together, and how many of
    // them to expect.  Probing only: the collective never touches this.
    void * probe_host;
    size_t probe_bytes;
    uint32_t probe_peers;
};

// Cumulative shard weights, so the kernel can derive lo[r]..lo[r+1) itself with
// integer arithmetic.  Every rank is handed the same numbers and divides the same
// vector count, so all of them land on the same boundaries: a gap would leave an
// element unreduced and an overlap would reduce it twice.
struct ggml_cuda_ar_shards {
    uint32_t cum[GGML_CUDA_MIXED_AR_MAX_RANKS + 1];   // cum[0] = 0, cum[n] = total
};

using ggml_cuda_mixed_ar_group_init_t = void * (*)(const ggml_cuda_mixed_ar_group_config *);
// The duplex probe is entered per registry from its own thread.  It cannot run
// from group_init: the runtimes are initialised one after another on one thread,
// so a rendezvous inside that loop is a rendezvous with nobody.
// (backends, count, rendezvous page, its size, how many runtimes, which one this
// is, and the read-over-write ratio of each local device).  The ratio is per
// device because it is per share: a rank publishes N and reads (1+2w)N, so
// 35/35/17/13 asks four different questions of the link, not one.
using ggml_cuda_probe_duplex_t = void (*)(ggml_backend_t *, size_t, void *, size_t,
                                          uint32_t, uint32_t, const double *);
using ggml_cuda_mixed_ar_group_free_t = void (*)(void *);
using ggml_cuda_mixed_ar_group_prepare_t = bool (*)(void *, size_t);
using ggml_cuda_mixed_ar_group_enqueue_t = bool (*)(void *, ggml_tensor **, size_t, uint32_t);

void ggml_cuda_probe_p2p(ggml_backend_t * backends, size_t n);
void ggml_cuda_probe_duplex(ggml_backend_t * backends, size_t n,
                            void * probe_host, size_t probe_bytes,
                            uint32_t probe_peers, uint32_t my_index, const double * ratios);
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
