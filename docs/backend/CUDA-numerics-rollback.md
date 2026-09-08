# CUDA numerical optimization rollback

Date: 2026-09-08. Working-tree changes based on `79f26bba9`, using its merged upstream `67672dc5b` as the CUDA arithmetic reference.

## Removed from source

- INT8 local/cross-runtime packing, F32-to-BF16 mixed wire, fused/streamed compressed modes, flat-group and two-stage reductions, and total requantization. These remain removed from source. At the user's subsequent request, only native-type flat mixed AllReduce has been restored, opt-in and off by default.
- The Volta MMVQ parameter table, the fork's Volta MMQ tile-table selection, per-expert J_FIT, and the forced grouped-MMQ prefill dispatch. MMQ and MMVQ now use the merged upstream implementation.
- The optional 256/256 sparse MMA attention variant, the fork's sparse dispatch overrides, and the Qwen4Exp sparse execution hint. The model's own top-k selection and dense attention mask remain unchanged.
- The optional sort chunk-size override, to retain upstream chunking and avoid introducing a separate tie-order assumption.

Compact prefill/decode attention was already removed in `91a4d956e`.

## Preserved

The V100 backend and driver bridge, model/converter/MTP support, model files and KV-cache settings, pinned staging copies, shape-keyed CUDA graph caching, and general opt-in profiling are not rolled back. The Volta large-head attention guard remains: it rejects an unsupported launch rather than substituting a faster approximate kernel.

This rollback addresses the CUDA numerical optimizations, not all feature differences between the fork and upstream. Older expert-cache and speculative-sampling features are not removed here. It does not establish numerical identity for those features.

## Multi-runtime tensor parallelism

Different backend registries use the meta backend's existing reduction by default. Set `GGML_CUDA_ALLREDUCE=mixed` before starting the process to opt into native-type mixed AllReduce, or `GGML_CUDA_ALLREDUCE=none` to force the meta-backend reduction. Registry identity is checked explicitly because CUDA and V100_CUDA share the CUDA GUID. Cross-registry fallback copies use pinned host staging without changing the tensor representation.

The restored mixed kernel uses one type for its input, wire, and output: F32 stays F32, F16 stays F16, and BF16 stays BF16. It accumulates in FP32 in global rank order on every GPU. There is no compressed local reduction or conversion of F32 to BF16. Old compression flags cannot change this dispatch. This does not promise bit identity with the meta backend's different reduction tree.

All runtime groups retire a slot through completion events before any rank reuses it. The grid is fixed at eight blocks, with two 64 MiB-per-rank slots. Oversize or unsupported input falls back before dispatch. Equality-only unsigned tokens permit wraparound after retirement. ABI version 5 and explicit layout fields reject the old compressed ABI and differing slot/grid layouts. Both CUDA backends must be rebuilt together.

Old `GGML_CUDA_MIXED_AR_*`, `GGML_CUDA_AR_LOCAL_INT8_THRESHOLD`, `GGML_CUDA_MMID_MMQ_PREFILL`, `GGML_CUDA_MMQ_MMID_J_FIT`, `GGML_CUDA_FATTN_SPARSE_256`, and `GGML_CUDA_SORT_PREFILL_CHUNK_MIB` settings no longer enable their removed paths. `GGML_CUDA_AR_BF16_THRESHOLD` does not affect native mixed AllReduce. The 27B four-GPU launcher now defaults `GGML_CUDA_ALLREDUCE` to `none`; its other user settings were preserved.

Single-runtime NCCL/internal AllReduce is upstream code, including its pre-existing BF16 conversion behavior. This rollback is not a global FP32-only mode and does not remove the model's quantization or quantized KV cache.

## Verification and deployment

Only source inspection, upstream comparisons, removed-symbol searches, and `git diff --check` were performed. No build, executable test, GPU workload, deployment, commit, or push was performed. Installed binaries still contain their previous code.

Next, with separate authorization, build both CUDA backends from the same source revision and compare quality on the exact inputs that exposed the regression, keeping model, cache, prompt, sampling, split, and context settings fixed. Establish quality before measuring speed. Previous throughput and small perplexity deltas are historical, not claims about this rollback.
