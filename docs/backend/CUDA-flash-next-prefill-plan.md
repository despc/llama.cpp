# Flash-Next four-GPU prefill and generation: status and optimization plan

Date: 2026-09-06. Source baseline: `f3aacb2f7`, branch `master`, plus local changes listed below. This is a local deployment plan, not a claim about upstream performance.

Reading guide: [prefill ranking](#optimization-options), [detailed prefill code review](#deeper-review-ranked-candidates-and-implementation-details), [actual hardware audit](#hardware-audit-and-generation-extension), [generation ranking](#generation-ranking-for-this-machine), and [combined execution plan](#combined-execution-plan-and-measurement-contract). The hardware/generation extension was added after the user requested optimization of both prefill and generation on the existing equipment.

## Scope and stop point

Improve prefill without changing the model files, hardware, model semantics, or reducing generation performance. Keep the usable context requirement of 160072 tokens. The already running 150k experiment is complete. No further GPU experiments were started after it; the test server was stopped. Implementation beyond the current opt-in scratch change is pending user review of this plan.

The current deployment is **Flash-Next with layer splitting**, not the earlier dense 27B tensor-parallel configuration. The launcher supplies `-ts` but not `-sm tensor`; the default is layer split. Earlier mixed-runtime AllReduce results cannot be used as a diagnosis of this workload.

Fixed reference configuration:

| Setting | Reference |
| --- | --- |
| Launcher | `/home/despc/llama.cpp/start_qwen_flash-4gpu-mtp.sh` |
| Runtime | `/home/despc/llama.cpp/fork_v100/llama-server` |
| Target | `/mnt/data/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` |
| Draft | `/mnt/data/models/qwen3.8-flash-next/mtp/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` |
| Devices | CUDA0: RTX 5080; CUDA1: RTX 5070 Ti; V100_CUDA0: V100 SXM2 32 GB; V100_CUDA1: V100 PCIe 32 GB |
| Split weights | `146,175,345,334` in layer mode |
| Context | Requested 160072; runtime slot padded to 160256 |
| Batch / microbatch | 4048 / 512 in the unchanged launcher |
| KV types | K and V both q8_0 |
| MTP | draft-mtp, depth 1, draft on V100_CUDA1 |
| Offload | All expert layers on GPU, no CPU MoE |
| CPU | 8 threads, affinity mask 5555, strict affinity |
| Driver bridge | `/opt/nvidia-v100/lib/v100_redirect.so` |

Model metadata was read from the actual GGUF: 48 layers, embedding width 2560, 512 experts with 10 selected per token, 24 attention heads, 2 KV heads, K/V head dimensions 256, indexer head count 4 and dimension 128, indexer top-k 2048. There are 12 QSA attention layers with compression ratio 4 and 36 recurrent layers. The large per-layer embedding table stays in host RAM by design.

## Measurements completed

Rates are tokens/s. Cold and warm are the first and second cycles of the existing `bench/cycle_flash.sh` harness, not confidence intervals. The 5k comparison has only two cycles per configuration.

| Configuration | 5k prefill, cold / warm | Generation, cold / warm | Long-prompt result |
| --- | ---: | ---: | --- |
| Original libraries, ubatch 512 | 452.9 / 477.2 | 57.00 / 64.05 | Historical: 30k at 411; 100k at 309 |
| Original under Nsight | 453.8 / 477.1 | 56.20 / 62.79 | Not run |
| Sort cap 8 MiB, ubatch 640 | 480.3 / 504.0 | 57.87 / 65.00 | 30001 tokens: 440.0155; 150001 tokens: 269.8467 |

The candidate gains about 5.6% warm 5k prefill. No generation slowdown appeared in these short samples; this is not yet proof of non-regression. The generated 128-token text matched a saved baseline response exactly, which is only a smoke check.

The 30k request took 68181.691 ms. The 150k request took 555874.799 ms, about 9 minutes 16 seconds. Both report `cache_n=0` and `truncated=false`. Both request only one output token, so neither measures sustained generation with a long context. The 150k pass demonstrates this particular candidate can get past the former 12-30k OOM point; it does not establish all-input stability or full 160k occupancy.

The historical long-prompt baseline is not a fresh paired A/B test. Do not claim a measured 150k speedup: no equivalent baseline 150k result was collected.

The long-run progress also shows context-dependent cost: early batches run above 500 tokens/s, whereas the 4048-token interval ending at 145728 takes about 22.5 seconds, roughly 180 tokens/s locally. The final 269.85 is the cumulative average. This motivates attention/indexer profiling but does not by itself identify which operation dominates.

## What is established by the code and traces

1. **Sort scratch was a memory problem.** With the installed CUDA 12.9/CCCL 2.8.2 build, large top-k uses argsort followed by copying selected indices. The old chunk cap limits the input matrix to 64 MiB, not total workspace. Index arrays, temporary keys, output indices, and CUB workspace can together require roughly five times that amount, depending on shape and CUB behavior. An 8 MiB input cap makes room for ubatch 640 at long context.
2. **Sort is not established as the main 5k time bottleneck.** Direct event instrumentation measured 21.974 ms over 96 top-k calls in one initial 5k run. That instrumentation did not distinguish backend names in its log. Saving this time alone cannot explain a large multi-fold prefill improvement. The main demonstrated benefit is enabling a larger microbatch.
3. **The Nsight view is incomplete.** The captured trace exposes the Blackwell runtime, not complete kernel timelines for both V100s. Over the entire trace, including loading, two prefills, and generation, visible kernel time totals 1.7643 s; `mul_mat_q` contributes 1.1415 s. These are not isolated 5k timings and must not be extrapolated to all four cards. Whole-trace memcpy totals also include loading.
4. **Pipeline parallelism falls back off at startup.** The original log reports a failed 1882.99 MiB CUDA0 allocation followed by `sched_reserve: compute buffer allocation failed, retrying without pipeline parallelism`. This is a real memory limitation, not evidence that four layer stages are already overlapped efficiently.
5. **QSA still builds a dense mask and does not select the sparse attention path.** `build_attn_qsa` supplies `n_kv_max=0`. Existing sparse CUDA eligibility requires Turing-style MMA, and sparse template configurations cover 512/512 or 576/512 head dimensions, not this model's 256/256. Uncommenting the sparse call is not a complete implementation, especially on Volta.
6. **Foreign-runtime copies already use pinned staging.** `ggml_backend_cuda_cpy_tensor_foreign` has a pinned ring and destination completion events. Its source side still synchronizes. Replacing pageable staging is not a new optimization here, and old AllReduce assumptions do not locate the current bottleneck.

## Local changes and deployment state

Current source changes made during this investigation:

- `ggml/src/ggml-cuda/argsort.cu`: opt-in `GGML_CUDA_SORT_PREFILL_CHUNK_MIB`, integer 1..64, default 64; invalid input warns and falls back. The cap applies only when the sort invocation has more than 8 rows. Short decode/verification invocations retain the old cap. `GGML_CUDA_SORT_PROFILE` prints scratch sizes.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: opt-in `GGML_CUDA_OP_PROFILE` event timing with backend, device, operation, and tensor name. It disables CUDA graph use and synchronizes each measured operation. A fused group is labeled with its first node. Presence of the variable enables it, including a value of `0`; unset it for performance tests.
- `tests/test-backend-ops.cpp`: two additional top-k cases, shape 33024 x 64 with k=2051, with and without ties. At an 8 MiB cap the row partition has a partial final chunk. Tests compiled but have **not been executed**.

Both CUDA backend targets and the test target compiled. The generic operation profiler has not been exercised. Its timings would be diagnostic, not valid production throughput: CPU time includes waits and overlaps GPU time, so the two must not be added. Transfers outside the graph, CPU operations, and unrelated streams require separate instrumentation.

The benchmarked library revision also contains an earlier optional top-k-only event profiler. That redundant profiler was subsequently removed from source and the final CUDA targets rebuilt, but those final artifacts were not benchmarked. With profiling unset, both revisions use the same new sort cap; nonetheless they are distinct binaries and must not be presented as identical tested artifacts.

At the stop point:

- The launcher is unchanged, including its ubatch 512 default.
- The original two runtime CUDA libraries were restored from `/home/despc/llama.cpp/sort-backup-jek8ww/`, with matching SHA-256 checksums.
- The exact experimental libraries used for the completed runs are preserved in `/home/despc/llama.cpp/sep06-sort-candidate-qBc2yK/`.
- Final-source builds remain under `build-v100/bin/`; promoting them requires another validation pass.
- The test server PID 14526 was stopped. No new GPU experiments are running from this task.
- No commit or push was made. Pre-existing model-name-check changes in `common/`, `tools/server/`, and server tests were left untouched.

## Acceptance protocol for every future change

Keep one immutable reference build and a separate candidate directory. Record source revision, dirty patch, binary hashes, launch environment, model paths, actual layer placement, driver/backend versions, and any startup fallback. Change one factor at a time.

Use identical tokenized prompts, sampling settings, seeds, and cache policy. Run interleaved A/B/B/A comparisons with at least five measured samples per candidate for short tests, separating startup/cold behavior from steady state. Record prompt tokens actually evaluated, server prompt time, wall-clock time to first token, generation tokens/s, inter-token latency, MTP acceptance, draft cost, and per-device memory peak. Report noise rather than rounding it into an improvement.

Cover 5k, 30k, 100k, and 150k prefixes, plus a boundary test close to the required context capacity that leaves enough output room. Measure sustained generation after both short and long prefixes. Repeat mixed-length requests in one server to catch allocator high-water marks, fragmentation, graph reuse, and state reuse. Use representative text and routing patterns as well as the synthetic long prompt.

Accept only changes with no repeatable generation regression relative to the reference distribution. A small difference inside measurement noise is inconclusive, not permission to slow generation by that amount. For arithmetic or graph changes, compare backend outputs/logits under appropriate numerical tolerances, token selection where exactness is required, and MTP acceptance; a matching single response is insufficient. Test sm_70 and sm_120 independently and then the combined deployment.

## Optimization options

The ranking below was revised after the deeper source review on 2026-09-06. It ranks plausible end-to-end benefit, not ease of implementation. The 30k-150k workload is the primary ordering; the short-prefill column distinguishes the 5k objective. Exact ordering within a tier is provisional until four-device phase measurements exist. No new GPU runs were made for this revision.

High means a candidate can remove a substantial repeated phase or enable overlap of whole stages; medium means it removes a bounded subphase or unlocks a useful memory tradeoff; low means current evidence points to secondary overhead. These are qualitative engineering judgments, not measured speedups. Items overlap, so their gains cannot be added.

| Rank / ID | Optimization | Long-prefix potential | 5k potential | Evidence and implementation cost |
| --- | --- | --- | --- | --- |
| R1 | Index-native sparse QSA attention, including selected Q8 KV loads | High | Medium | Dense route confirmed; large kernel/interface work |
| R2 | Real pipeline overlap with smaller live buffers and safe input slots | High ceiling, low confidence of reaching it | High ceiling | Allocation fallback and input barrier confirmed; large scheduler/state work |
| R3 | V100 MoE dispatch first, then grouped kernel tuning | High if fallback dominates | High; best first compute investigation | Concrete dispatch mismatch found; small diagnostic/dispatch experiment before large kernel work |
| R4 | Generate QSA bias/masks from compact metadata on GPU | Medium-high | Low-medium | Host nested loops and dense transfers confirmed; medium graph/input work |
| R5 | Select from compressed QSA scores without full token expansion | Medium-high | Low-medium | Repeated expanded matrices confirmed; difficult exact boundary semantics |
| R6 | Chunked Gated DeltaNet prefill with rollback support | Medium-high | High if recurrent phase dominates | Serial token loop in fused kernel; substantial algorithm/state work |
| R7 | Tile-local Q8-to-FP16 KV conversion in dense attention | Medium, higher if conversion/memory dominates | Low-medium | Full-view conversion confirmed on MMA path; medium-large kernel work |
| R8 | Tune dense attention and its MTP catch-up path | Medium, possible post-R1 bottleneck | Low-medium | Draft is explicitly dense; medium tuning work |
| R9 | Fuse hyper-connection mixing/combine and repeated pointwise operations | Medium | Medium | Four-wide residual and repeated graph sequences confirmed; medium fusion work |
| R10 | Incrementally maintain pooled indexer keys and cache mappings | Medium | Low | Old prefix is reprocessed; additional persistent memory and invalidation complexity |
| R11 | Bounded sort workspace plus larger safe microbatch | Medium indirect; stability already demonstrated once | About 5.6% observed for the combined candidate | Lowest-cost validation/promote candidate; not fully validated |
| R12 | Generic exact batched top-k instead of full argsort | Medium-low alone; useful fallback/enabler | Low direct, based on initial timing | Full-sort path confirmed; medium implementation, tie contract unresolved |
| R13 | Device-local target-to-MTP hidden-state handoff | Low-medium | Low-medium | Host handoff confirmed, 10240-wide hidden state; medium interface/lifetime work |
| R14 | Refine integer layer placement using new stage timings | Conditional medium | Conditional medium | Existing memory cliff; easy to vary, expensive to validate |
| R15 | Overlap remaining foreign-runtime pinned transfers | Low-medium unless waits dominate | Low-medium | Pinned ring already present; medium-large bridge/lifetime work |
| R16 | Graph reuse, launch scheduling, logical batch alignment | Low-medium; interacts with R2/R3 | Low-medium | Exact-prefix shape changes and fallback restrictions confirmed; medium work |
| R17 | CPU PLE preparation and exact gather/upload improvements | Low on present evidence | Low on present evidence | PLE metadata selects one layer, not all 48; small-medium work |

For 5k alone, investigate R3 first, then compare measured R6 and R9 costs; R2 has a larger theoretical overlap ceiling but is much harder to realize. R11 remains the first candidate to finish validating because it has an actual positive result. For long prefixes, R1/R4/R5 address work that grows with prefix length. R7/R8 must be accounted for before assuming R1 solves that growth completely.

The sections below retain their original work-package labels for continuity. The new [deeper review](#deeper-review-ranked-candidates-and-implementation-details) maps them to R1-R17 and adds concrete entry points, byte estimates, and prototype decisions.

### 0. Account for the full prefill, then choose the expensive branch

Start with the opt-in operation profiler for diagnostic runs on both runtime backends. Group by backend/device, layer, operation, shape, prefix length, and target versus draft graph. Separate MoE routing, activation quantization, expert matmuls, attention/indexer/mask operations, recurrent layers, and draft prefix catch-up. Check the profiler itself against a small known workload before trusting its output.

Add coarse wall-clock ranges around scheduler reserve, graph/input construction, host PLE gather, input upload, backend copies/waits, target graph execution, hidden-state export, and draft processing. These ranges must identify requests and microbatches. CUDA event values from separate driver runtimes cannot be treated as one global timestamp clock. Use host-correlated ranges and each runtime's local event intervals; do not sum overlapping work into a false critical path.

Measure uninstrumented throughput separately. If per-op synchronization changes dispatch/overlap materially, add an event-ring mode with deferred collection at existing completion points, bounded event count, and no new per-op synchronization. First retain the simpler profiler as a correctness reference.

Deliverable: a per-phase time/memory table at 5k and long-prefix positions, plus unexplained wall time. Use Amdahl's law before a kernel rewrite: if a component takes fraction f of the critical path, removing it entirely cannot exceed speedup 1/(1-f).

### 1. Finish the bounded-workspace candidate

Files: `argsort.cu`, `top-k.cu`, existing backend operation tests.

Run the new top-k cases with 8 MiB and the default, on both backends. Include ties, partial chunks, k near row width, short rows, infinities, supported NaN behavior, and non-contiguous inputs where the operation supports them. Verify CUDA graph capture/reuse and the <=8-row path. Confirm that the sort scratch report covers the actual top-k output allocation as well as CUB scratch when calculating total memory.

After correctness, compare caps 4, 8, 16, and 32 MiB with ubatches 512 and 640. Smaller scratch can cause more launches, so the smallest cap is not necessarily fastest. Do not immediately repeat known-failing 704/768 configurations; first show a memory budget that makes them plausible. Benchmark the final-source binaries rather than assuming the earlier instrumented build is equivalent.

Longer term, expose a bounded total sort workspace policy accounting for all live arrays, not merely input bytes. Keep a conservative per-device reserve for graph buffers and KV growth. Avoid `cudaMemGetInfo` in every operation and avoid forcing global pool flushes. Publish a separate opt-in launcher only after validation, retaining an easy fallback.

### 2A. Optimize the actual V100 MoE path

Files: `mmq.cu`, `mmq.cuh`, the `MUL_MAT_ID` dispatch in `ggml-cuda.cu`, and associated quantization helpers; resolve the exact selected kernel in the trace first.

With 512 experts and 10 active per token, the mean expert receives about 10 token assignments at ubatch 512 and 12.5 at 640. Real routing is nonuniform. Benchmark the observed per-expert token histogram, tensor widths, and quantization types, not only a dense large GEMM.

Compare the current grouped MMQ path against bounded block dequantization followed by FP16 Tensor Core GEMM on Volta. Choose by device, quantization type, and token count; keep decode dispatch unchanged. Volta must not be treated as having the same native integer Tensor Core path as Blackwell. Include the cost of sorting assignments, quantization, dequantization, scratch, and scatter in end-to-end measurements.

Audit the existing `dedup_bcast`/quantize-scatter path before adding an equivalent optimization. Potential later fusions are routing with activation packing, paired gate/up processing where formats permit, and activation/down preparation. Avoid dequantizing or caching the whole model: the memory budget cannot absorb it. No weight requantization or lower-precision substitutions.

Exit criterion: lower measured V100 expert-stage time at real shapes, acceptable scratch at 150k, and unchanged decode/acceptance under paired testing. If profiling finds another phase dominant, defer this branch.

### 2B. Replace full sorting with exact bounded top-k

The active large-row path sorts all elements then keeps roughly 2051 indices. Prototype batched radix selection or hierarchical block top-k with a bounded merge workspace, retaining the current path as a fallback. Avoid an algorithm that launches one host-side operation for every query row without measuring launch overhead.

Keep the current ordering and index tie-break rules, including equal repeated compressed scores and masked values. Compare both selected indices and resulting attention masks. Measure work across prefix lengths: a method that wins at 150k can lose at 5k. Do not update CUDA/CCCL blindly to obtain `DeviceTopK`; toolkit compatibility with Volta and per-row implementation overhead both need checking.

### 2C. Avoid expanding compressed scores before selection

File: `src/models/qwen4exp.cpp`, particularly `build_qsa_top_k`.

The indexer scores compressed blocks and then expands scores to token positions before top-k. Investigate selecting compressed candidates first, then expanding only the required token indices. Compression ratio 4 does not imply that selecting exactly 2048/4 blocks is equivalent: the current token-level width can be 2051, causal boundaries split blocks, and tied scores require the same token-level tie rule.

First write a reference mapping from compressed scores plus query position to the exact existing selected token indices. Test every query position modulo 4, the newest partial block, short contexts, equal scores, and padding. Only replace the graph after equivalence is established. This can save expanded score buffers and work; it cannot remove the cost of scoring all compressed blocks.

### 3. Make sparse attention actually consume sparse indices

Files: `qwen4exp.cpp:build_attn_qsa`, `fattn.cu`, `fattn-mma-f16.cuh` and dispatch instantiations.

Stage A: implement and test sparse 256/256 support with 24 query heads and 2 KV heads on Blackwell. Preserve q8_0 KV access, masks, scaling, rotations, and supported attention semantics. Use the current dense-mask result as the reference. Retain a dense fallback where short prefixes or the selected fraction make sparse execution unprofitable.

Stage B: implement a Volta-compatible path and benchmark its register/shared-memory usage and occupancy. Existing sparse eligibility excludes Volta, so this is more than changing a flag or adding a template instantiation.

Stage C: pass compact selected indices directly to attention, removing dense mask fill/scatter/add and a possible later dense-to-sparse compaction. Define the graph operation contract and buffer ownership before changing interfaces. Keep a reference path and tests for causal rows, empty/masked candidates, and multiple streams.

This is a strong architectural candidate for long context because the model already selects a small subset of KV positions. The size of the end-to-end gain remains unknown until actual dense attention/indexer time is measured. Do not reduce model top-k to manufacture a faster result.

### 4A. Reduce live graph buffers and reserve memory

Files: `src/llama-context.cpp`, graph allocator/scheduler code, QSA graph construction, CUDA temporary pools.

Collect per-device planned graph allocation, temporary-pool high-water mark, KV/indexer storage, draft storage, and failed reserve requests. Distinguish allocator-reserved memory from tensors concurrently live. Inspect expanded scores, masks, repeated inputs, and intermediate FFN outputs for exact lifetime-based reuse or chunking.

Fuse or stream intermediates only where consumers permit it. Prove that aliased buffers are no longer needed by another stream, microbatch, recurrent state update, or draft pass. Prefer bounded scratch and shorter lifetimes over memory overcommit, unified-memory paging, or allocation retries in the hot path.

The goal is usable headroom, not simply a lower startup reading: validate late-prefix allocations and repeated requests. This work can enable both larger microbatches and pipeline buffers; those two consumers compete for the same memory.

### 4B. Restore useful pipeline overlap

Files: `src/llama-context.cpp` pipeline eligibility/reserve/process_ubatch paths and backend scheduler copy slots.

First make the existing pipeline allocation succeed within the fixed memory budget and log whether it remains enabled. Then trace actual overlap of two microbatches across layer stages. Reserve double-buffered inputs/outputs only for needed lifetimes and reuse existing scheduler mechanisms where possible.

Audit recurrent state, KV writes, graph reuse, and host input mutation before loosening synchronization. A stage cannot read a new microbatch from storage still used by the prior one. MTP catch-up must see the correct hidden states and prefix ordering. Do not remove `synchronize` calls merely because they appear expensive.

Tune stage balance using measured stage times, not equal VRAM fractions. Keep single-token generation on its efficient path. A pipeline that speeds prefill but increases decode synchronization or memory pressure fails the objective.

### 5A. Remove avoidable target-to-MTP host round trips

Files: `common/speculative.cpp` draft-MTP `process`, `src/llama-context.cpp` hidden-state output/export.

The current path exports target hidden states, obtains host embeddings, copies shifted rows into the draft batch, and runs draft prefix catch-up. Determine the real producer device: only if target hidden output and draft input are on the same V100 device can a device-local handoff remove the host copy outright.

Add an internal device-buffer handoff with explicit ownership and completion signaling, preserving the one-token row shift, pending hidden row, draft KV position, rollback, and accepted-token accounting. Otherwise use bounded pinned asynchronous staging between runtimes. Do not skip draft prefix processing or disable MTP to improve prefill numbers: existing generation performance depends on it.

### 5B. Improve only the measured inter-runtime boundaries

File: `ggml-cuda.cu:ggml_backend_cuda_cpy_tensor_foreign` and bridge interfaces.

Count bytes and waiting time at actual layer boundaries and MTP transfers. The current pinned ring already avoids ordinary pageable staging. The remaining candidate is asynchronous source-side transfer plus overlapped destination upload using bounded chunks and per-runtime events.

A cross-runtime ABI must manage source completion, staging-slot reuse, errors, and destination completion without passing incompatible event/stream objects between driver namespaces. Test small and large transfers: extra chunking and callbacks can hurt small decode messages. Keep the existing path for those messages unless measurements justify replacement.

Shared NCCL or direct cross-driver P2P is not assumed available. Reopening that infrastructure project is low priority for this layer-split workload unless transfer accounting shows a large critical-path fraction.

### 5C. Optimize host PLE preparation and input copies

The roughly 26.8 GiB per-layer embedding table remains in RAM. Measure token-dependent gather, CPU memory bandwidth, projection preparation, pinning, upload, and any repeated copies to devices that do not consume the input.

Possible changes: batch/prefetch exact row gathers, reuse input buffers through the microbatch, use bounded pinned upload buffers, and avoid redundant copies where the graph permits. Preserve token hashing/indexing and recurrent request state. Adjust thread count/affinity only after showing a host bottleneck; do not assume more threads helps. Moving the whole table to GPU or removing the feature is outside scope.

### 5D. Audit launch overhead, graph reuse, and existing fusions

Measure graph rebuild/update/capture frequency as prefix length changes, host kernel-launch gaps, and actual activation/normalization/routing fusions. Confirm whether the model reaches already implemented optimized paths before adding new ones.

Potential improvements are shape-bucketed graph reuse with exact mask lengths, bounded reusable input storage, and fusing proven short pointwise sequences. Keep dynamic KV length and recurrent-state semantics correct. Do not use the synchronized diagnostic profiler to claim reduced production launch overhead.

Treat logical batch size 4048 versus aligned alternatives such as 4096 as a later small controlled experiment, with fixed ubatch and sufficient memory. A different batch changes graph/input and draft-processing boundaries; it is not automatically an optimization.

### 5E. Refine layer placement after kernel and memory changes

Keep the same four devices, model, and draft. Read actual integer layer assignments; nearby `-ts` values can produce no change or abruptly move a large expert layer. Use per-stage compute time and late-context headroom rather than percentage weights alone.

If sparse attention becomes faster only on Blackwell, investigate placing appropriate attention-heavy stages there within memory limits. Do not assume arbitrary non-contiguous placement is supported by the current launcher. Preserve draft headroom on V100_CUDA1 and rerun both decode and long-context checks for every accepted placement.

## Excluded shortcuts and lower-priority ideas

- Different GGUF quantization, model, hardware, expert count, routing top-k, compression ratio, KV precision, or smaller usable context violate the fixed reference unless the user changes scope.
- CPU MoE offload has already regressed this deployment and is not the default memory remedy.
- Turning MTP off, reducing its context preparation, or changing draft depth to inflate prefill numbers does not preserve the generation requirement.
- Prefix-cache hits must not be reported as raw prefill throughput.
- Global force-MMQ/force-cuBLAS flags are experiments, not a substitute for shape/device-specific dispatch.
- Blind toolkit upgrades, one NCCL across separate driver runtimes, full-model dequantized caches, and oversubscribed GPU memory are not immediate solutions.
- Replacing the entire execution engine or converting back to dense-27B tensor-parallel assumptions would be a separate project, not an incremental fix to this launcher.

## Exact next sequence after approval

1. Validate the final-source scratch-cap build with backend correctness tests on both architectures; rerun the short paired benchmark and long-context generation checks before promoting it.
2. Collect complete four-device/host phase accounting with the new profiler, then validate conclusions against uninstrumented wall time.
3. Specifically check the V100 `MUL_MAT_ID` fallback and its graph-disable condition (R3). If active and expensive, compare a prefill-only grouped-MMQ dispatch before writing new kernels. Update both execution and graph-compatibility predicates together.
4. For prefix-dependent cost, separate host QSA bias, indexer preparation/selection, target attention including KV conversion, and dense MTP attention. Prefer R4 or a bounded R5 prototype if they remove large time/memory at modest complexity; pursue R1 for the larger structural opportunity. For short-prefill recurrent cost, assess R6 and R9 independently.
5. Reassess live memory and actual overlap for R2; then work on the remaining R7-R17 candidates in descending order of their measured critical-path cost. The potential ranking is not a requirement to build the hardest item first.
6. Promote only validated candidates, update launch comments with reproducible results, and ask separately before commit/push.

## Deeper review: ranked candidates and implementation details

This section is source/metadata research, not another benchmark report. Current source was inspected directly; the restored production libraries can differ from it. Dispatch deductions below require confirmation against the actual candidate binary. Newly read GGUF fields: hyper-connection count 4, low rank 320, expert FFN width 640, recurrent state dimension 128, 16 key heads and 48 value heads, PLE layers `[1]`, 16 PLE hash heads, and PLE n-gram size 3. Expert tensors are not uniformly Q4: inspected shards include Q4_K, Q5_1, Q5_K, and Q8_0. Layer 0 gate/up experts have shape `[2560,640,512]` in GGUF dimension order and Q4_K; down experts are `[640,2560,512]` Q5_1.

### Scale estimates that change the priorities

The following are logical payloads at N=150000 cached positions, T=640 query tokens, one stream, without allocator alignment/padding. They are not measured VRAM peaks or transfer counts; graph reuse can make tensors share storage and copies can multiply traffic.

| Object | Formula | Approximate size |
| --- | --- | ---: |
| Expanded F32 token scores | N * T * 4 bytes | 366.2 MiB |
| Four-head F32 compressed scores | (N/4) * 4 * T * 4 | 366.2 MiB |
| F32 compressed block bias | (N/4) * T * 4 | 91.6 MiB |
| One F16 dense attention mask | N * T * 2 | 183.1 MiB |
| Selected I32 token indices | 2051 * T * 4 | 5.0 MiB |
| Full FP16 K plus V conversion, one attention layer | 2 * N * 2 KV heads * 256 * 2 bytes | 293.0 MiB |
| Wide F32 hidden microbatch | T * (2560 * 4) * 4 | 25.0 MiB |
| F32 pooled indexer keys, one layer | (N/4) * 128 * 4 | 18.3 MiB |
| One recurrent state snapshot, one recurrent layer | 128 * 128 * 48 * 4 | 3.0 MiB |

At a 150k prefix, selected attention positions are about 1.37% of N. The ratio N/2051 is about 73, but this is a candidate-position ratio, **not** an attention or model speedup prediction: indexer scoring, irregular loads, GQA reuse, conversion, draft attention, and other layers remain. Likewise, 21.974 ms of top-k in an approximately 11-second 5k run suggests only about 0.2% direct opportunity in that particular diagnostic sample, not a large short-prefill win from sorting alone.

Use measured critical-path fraction f and subphase speedup r: end-to-end speedup is `1 / (1 - f + f/r)` if the rest is unchanged. Doubling a phase that occupies 10%, 30%, or 50% gives only about 1.05x, 1.18x, or 1.33x overall. Memory-enabled overlap is a separate effect and needs an actual timeline.

### R1. Sparse attention: three separate costs must be removed

Source anchors: `qwen4exp.cpp:build_attn_qsa`, `fattn.cu:ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse`, `fattn-mma-f16.cuh:ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse`, `fattn-common.cuh:launch_fattn`.

The complete chain is currently selected indices -> dense fill/scatter/add mask -> dense attention. The existing optional sparse wrapper would additionally scan the dense mask to compact indices. Its MMA launch also requests FP16 K and V, causing conversion of the whole KV view before attention. Thus merely activating the existing sparse kernel leaves two large O(N) preparations intact.

Minimal staged design:

1. Build a correctness prototype for 256/256 and GQA ratio 12, using the current dense mask and converted KV. This isolates sparse arithmetic from representation changes.
2. Introduce an index input with explicit per-query valid counts, causal filtering, strides, and sequence ownership. Do not put GPU pointers into scalar op parameters. Preserve the generic graph/backend fallback.
3. Load selected Q8_0 blocks into shared/register FP16 tiles inside the sparse kernel. Reuse a KV load across query heads sharing it when profitable. Do not allocate `[T,2051,heads,256]` gathered K and V: naive per-query materialization would itself require multiple GiB.
4. Remove dense mask materialization for the validated single-sequence causal path; retain the general path for unsupported masks or layouts.

Current sparse templates use `ncols1=1`; this matters because different query tokens select different positions. Increasing token columns to get dense-style reuse can mix incompatible index lists. Volta needs its own legal MMA/load configuration; the current sparse mask loader disallows its cp.async path and sparse multistage loading is not implemented. Start with one stage and measure gathers/occupancy, then optimize.

Measure selected-index compaction, KV conversion, attention, and full request time separately. Require equivalence of masks/selected membership before attributing logit differences merely to floating-point accumulation. Keep the original decode path initially. R4/R5/R7 remove overlapping costs, so credit each byte/time only once.

### R2. Pipeline: memory slots are necessary, but not sufficient

Source anchors: `llama-context.cpp` constructor and `process_ubatch`; `ggml-backend.cpp:ggml_backend_sched_new` and split-copy/event handling. `GGML_SCHED_MAX_COPIES` defaults to 4, whereas the sequential scheduler uses 1. A two-slot experimental scheduler is a narrower first design than assuming four slots are mandatory. Its memory reduction is not automatically 50%: only duplicated live inputs/copies are affected.

For four ideal equal-duration stages and m microbatches, serial stage time is `4*m*t`, pipeline time is `(m+3)*t`. A logical batch 4048 with ubatch 640 has about seven microbatches, giving an ideal 2.8x stage-only ceiling. Actual unequal stages, copies, CPU work, draft catch-up, barriers, and partial batches reduce that ceiling. This is not a forecast for the server.

First measure reserve with 1/2/4 copy slots after memory accounting. Then inspect input reuse: `process_ubatch` explicitly synchronizes the scheduler before overwriting reused graph inputs when pipeline is enabled. QSA input shapes also change with N, creating a different obstacle to reuse. Removing the allocation fallback alone therefore does not prove overlap.

The implementation needs immutable in-flight inputs, bounded graph-result ownership, and per-stage state ordering. Recurrent updates for consecutive microbatches may overlap different stages, but must remain ordered within the same layer/sequence. Record start/end of stage i for microbatch j and prove overlap directly. Changing tensor overrides for placement can also disable pipeline eligibility; account for that interaction.

### R3. MoE: investigate dispatch before replacing math

Source chain: `ggml-cuda.cu:ggml_cuda_mul_mat_id` -> `mmq.cu:ggml_cuda_should_use_mmq` -> `mmf.cu:ggml_cuda_should_use_mmf` -> fallback in `ggml_cuda_mul_mat_id`. The graph checker uses the parallel predicate `ggml_cuda_mul_mat_id_needs_sync`.

For sm_70 with the current normal build flags, FP16 MMA hardware is available but Turing MMA is not. The NVIDIA MMQ predicate then uses `ne11 < MMQ_DP4A_MAX_BATCH_SIZE`, where the threshold is 64. The caller passes total microbatch tokens (`src1->ne[2]`), not the per-expert token population. At 512/640 it rejects MMQ. MMF rejects quantized weights. Subject to the actual binary and tensor path matching this source, quantized expert projections therefore enter the fallback.

That fallback copies IDs to CPU, synchronizes, scans expert x token x selected-expert assignments, uploads regrouped IDs, synchronizes again, and calls `ggml_cuda_mul_mat` once for each nonempty expert. The inner call may itself choose MMQ for a small expert slice; it is inaccurate to label every inner call cuBLAS. There can be up to 512 such calls for each gate/up/down projection, plus packing. The fallback also disqualifies its graph from CUDA capture. This is a concrete mechanism for CPU overhead and launch fragmentation, not yet a measured attribution of the V100 time.

First prototype: an opt-in **Volta, quantized MUL_MAT_ID, prefill-only** grouped-MMQ choice, using the existing `ggml_cuda_mul_mat_q` and `mmid` helper. Keep single-token/MMVQ verification behavior untouched. Update execution and `needs_sync` logic together. Log selected route, quant type, token count, expert count, and actual per-expert histogram. Compare full operation time including packing against fallback; do not use a global FORCE_MMQ as the proposed production change.

Second prototype only if needed: device/shape-specific grouped tile selection or bounded dequantized tiles plus FP16 MMA. Volta currently shares the function named `ggml_cuda_mmq_get_config_ampere` for configuration selection; that name is not proof of native Ampere instructions or optimal Volta occupancy. Inspect instantiated instructions and resource use before retuning J/rows/warps. Tiny expert populations may favor DP4A over a dequantized GEMM despite lower peak compute.

The present grouped path already has GPU assignment grouping and `dedup_bcast` quantize/scatter for gate/up. Reuse them. Do not predict a speedup from changing arithmetic precision; dequantization and activation rounding need numerical comparison. NVIDIA documents Volta Tensor Core FP16 inputs and FP16/FP32 accumulation, which is the applicable hardware constraint, not Blackwell integer/FP4 capability. [Volta tuning guide](https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html#tensor-core-operations).

### R4. Eliminate dense host QSA bias uploads

Source: `llama-memory-hybrid-idx.cpp:set_input_qsa` and `qwen4exp.cpp:build_qsa_top_k`. The current `blk_bias` optimization already reduces the host bias width from N to N/4 and shares one graph input set across layers with the same ratio. Do not count a new 4x compression or twelve independent CPU bias constructions as future gains.

There is still a loop over every query and every block writing F32 values: 0, -infinity, or finite `1e9f` for the tail. At N=150k/T=640 this is about 91.6 MiB of host output per graph input set. Scheduler copies to consumer devices must be measured, not assumed once per layer or once total.

Prototype for single-stream text: upload block position/validity/ownership plus query positions, then generate the bias where it is consumed, preferably fused into score reduction/selection. Keep the general CPU path for mixed sequences, noncausal attention, ALiBi, duplicated multimodal positions, and ranked-cell layouts until covered. Preserve the finite tail sentinel: replacing it with positive infinity can create NaNs when combined with a negative-infinite causal mask.

This removes O(T*N/4) host filling and payload, not necessarily O(T*N/4) GPU scoring. Extend the same principle to a causal mask only when the attention/indexer interface accepts compact metadata. Measure CPU set_inputs time and copy bytes; a GPU kernel that merely fills another full temporary may save transfer but not memory.

### R5. Compressed-domain selection and fused score reduction

Source: `build_qsa_top_k` produces a four-head score tensor, applies ReLU before head summation, adds block bias, permutes/gathers to token space, casts/adds the causal mask, and takes width 2051. Because four indexer heads equal compression ratio four, the unsummed head score tensor is as large as the expanded token-score tensor; removing only expansion still leaves roughly 366 MiB in the example above.

Prototype A: fuse ReLU-per-head, ordered head summation, bias application, and compressed candidate selection. A tiled score-matmul epilogue or tiled partial selection can avoid storing all head scores, but reuse of the existing GEMM may be more valuable than saving a buffer at short N. Do not sum raw head scores before ReLU; that changes the model.

Prototype B: expand only candidate blocks, then apply an exact final token-level selection. Establish a provable candidate count using full valid blocks, forced tail, masks, and boundary ties; do not assume simply 512 blocks or assume physical cache cells are contiguous token positions. Cache metadata can remap cells and introduce a spare dead block.

The tie problem is stronger than the first plan implied: the current noncaptured CUB path calls `DeviceSegmentedSort::SortPairsDescending`, not a stable-sort API, while capture uses segmented radix sort. Existing behavior must be characterized before claiming stable low-index ties. For QSA, output permutation alone is irrelevant to scatter-mask membership; which tokens win a boundary tie is not irrelevant. Define and test the required reference contract, including captured versus uncaptured runs. A deliberate tie-rule correction must be identified as a separate correctness change, not hidden in a speed patch.

### R6. Chunked recurrence while retaining speculative rollback

Source: `gated_delta_net.cu:gated_delta_net_cuda` has `for (t=0; t<n_tokens; ++t)` inside each CUDA block; its launcher contains a chunked-prefill TODO. The model uses state dimension 128, 48 value heads, and 16 key heads. At one sequence the present launch has 48 x 32 = 1536 blocks, each doing the token loop; this is not the same as no GPU parallelism, but tokens within a block are serialized.

There is already a graph-based `build_delta_net_chunking` in `src/models/delta-net-base.cpp`. Use it as an algorithm/reference starting point before adding another subsystem. More importantly, `build_recurrent_attn` takes the snapshot-preserving fused operator directly when `n_rs_seq > 0`; toggling `fused_gdn_ch` alone does not replace that path under rollback.

Prototype: process most of a prefill microbatch with an exact chunked recurrence, then run a short sequential suffix to produce all required last K state snapshots. Establish K from runtime rollback configuration, not from an assumed relation to draft depth. Alternatively extend the chunk kernel to emit the required suffix snapshots. Validate state at every rollback slot, output at every token, different microbatch partitions, and accepted/rejected draft sequences. Keep the existing one-token kernel.

Start with bounded chunks such as 32/64/128 as experimental values and measure temporary matrices, numerical drift, register pressure, and total recurrent time. The algorithm changes accumulation order, so long-sequence state/logit comparisons are essential. Gated DeltaNet's paper and the authors' associated implementation family establish a chunkwise approach, not drop-in compatibility of a Triton kernel with this mixed-driver C++ backend. [Gated Delta Networks](https://arxiv.org/abs/2412.06464), [FLA chunk implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py).

### R7. Avoid converting an entire Q8 KV view before each MMA call

Source: `fattn-common.cuh:launch_fattn` calls `to_fp16` over `ggml_nelements(K)` and V, or the corresponding noncontiguous conversion. `fattn-mma-f16.cuh` requests both converted operands. The extra buffers are accounted for by the attention backend; they are not necessarily CUDA-pool allocations. No incremental validity cache is maintained by this conversion code.

Prototype A for dense attention: dequantize Q8_0 tiles while loading shared memory. Compare saved conversion writes/reads and graph-buffer memory against extra repeated dequantization across query tiles. Full conversion can be faster when many queries reuse the same cache, so direct Q8 loads are not an unconditional win.

Prototype B: reuse converted tiles within a bounded persistent kernel or within a microbatch; do not retain a full FP16 mirror for every layer at long context. A 150k layer's full K+V converted view is about 293 MiB, so twelve permanent mirrors alone approach 3.4 GiB before draft/other buffers. Preserve exactly the same Q8 decoded values; do not bypass the quantized representation by caching original unquantized K/V.

R1 should use selected tile conversion from the outset of its production design. R7 remains independently relevant to dense MTP attention. Measure bandwidth, register use, occupancy, and changed allocation headroom, not just the dequantization kernel duration.

### R8. Dense attention and MTP catch-up remain after sparse trunk work

Source: `qwen4exp.cpp:graph_mtp` explicitly builds dense attention and has a QSA TODO. `common/speculative.cpp:process` runs draft catch-up for the prefix when memory is not shared. This means there is another context-growing attention phase on V100_CUDA1 even if target QSA becomes efficient.

Profile target and draft separately; inspect graph pruning when catch-up requests no logits before assuming every draft FFN/LM-head row executes. Optimize the dense attention kernel for actual 256/256, GQA ratio 12, microbatch/prefix shapes. Current Volta dispatch uses GQA grouping 4 for ratio 12, whereas the non-Volta optimized branch chooses 8; compare existing legal alternatives before adding a new layout. A group of 8 leaves a partial second group for ratio 12, but that alone does not prove grouping 4 is faster.

Also inspect `launch_fattn`'s KV_max optimization: the mask scan is currently gated on query count >=1024 or multiple streams. Our 512/640 single-stream prefill does not satisfy that gate. Testing a cheaper per-query causal bound may avoid the future/padded end of a microbatch, but cannot skip arbitrary sparse holes and is likely a small late-prefix improvement.

Keep the draft's current dense semantics for this optimization. Switching its attention to QSA is a separate correctness/acceptance project, not a free acceleration under the fixed reference. R7 can help without changing draft semantics.

### R9. Four-wide hyper-connections and pointwise fusion

Source: `qwen4exp.cpp:build_hc_mix` and `build_hc_combine`. HC count 4 makes the live residual width 10240, not 2560. Each layer has attention and FFN mixing/combine stages. The combine sequence scales and sigmoids four injection weights, repeats the block output, multiplies, then adds the residual.

First prototype a fused combine implementing `residual + block_out * (2 * sigmoid(inject / 4))` with broadcast indexing, without materializing REPEAT and its multiplied output. At T=640, each wide F32 intermediate is 25 MiB. Count actual allocated nodes and existing backend fusions before claiming all of that is saved. Protect graph input/output aliasing and match rounding where possible.

Then consider fused gating plus four-stream mean in `build_hc_mix`, preserving the existing sum order, and grouped norm/scale fusion. The low-rank HC projections are real matmuls (10240 <-> 320), not pointwise overhead; profile them separately and do not expect elementwise fusion to remove that compute. This is a useful short-prefill candidate and memory enabler, not just a launch-count cleanup.

### R10. Cache pooled keys only with explicit invalidation

Source: `build_qsa_top_k` gathers all raw cached keys, makes four slice copies, sums/scales, applies norm and RoPE to all pooled blocks every microbatch. Source raw keys are quantized in the indexer cache; pooled results must be derived from those same decoded values, not from pre-cache F32 projections.

First reduce redundant materialization with a gather/pool/norm/rotation kernel for current inputs; then consider caching final pooled keys for completed unchanged blocks. The example F32 cache costs about 18.3 MiB per QSA layer, around 220 MiB across twelve layers, plus metadata. This persistent allocation competes with pipeline and sort headroom; do not reserve it globally without a per-device budget.

Cache identity must include layer, sequence ownership, physical cell mapping, positions/rotation, and validity generation. Invalidate for rollback, partial-block completion, cell reuse, sequence copy/remove/shift, and multimodal ranked positions. For append-only single-sequence text, an incremental fast path can recompute newly completed blocks and the tail while retaining the general fallback. Cache block metadata separately from layer-specific values. This removes old-key preparation, not query-to-all-block score matmul.

### R11. Finish the small proven memory enabler

At N=150000 one F32 row is 600000 bytes: 8 MiB fits only 13 rows per sort chunk, so a T=640 call takes about 50 chunks versus roughly 6 at 64 MiB. The observed benefit therefore comes despite extra chunks, not because fewer sort launches are guaranteed. Prefix padding changes these counts slightly.

The nrows>8 condition is a shape heuristic, not a semantic prefill flag. It protects current short verification shapes but not an arbitrary future large speculative batch. Test the final partial chunk and captured/uncaptured CUB paths, whose workspace can differ. Sweep only after correctness and retain the default/backup. The 5.6% result belongs to the combined cap+ubatch candidate; isolate cap at fixed ubatch before attributing direct speed to the cap.

### R12. Generic batched top-k is a fallback, not the first 5k target

Use a bounded per-row threshold/radix selector plus stable boundary resolution if that is the chosen contract, or hierarchical local candidates with exact merge. With k=2051, a naive hierarchy retaining k per small input tile may keep most data and cost more than argsort. Model workspace and traffic at real k/N rather than choosing a top-k algorithm tuned for k=1..32.

Current `top-k.cu` would enable per-row `DeviceTopK` on the newer compile-time branch and explicitly request `not_guaranteed` / `unsorted`. The newer CCCL documentation distinguishes selected-set determinism from output order and imposes architecture-specific batched support restrictions; it is documentation of a newer/unstable API, not installed CUDA 12.9 capability. Do not upgrade the toolkit as a substitute for implementing and validating our contract. [CCCL top-k requirements](https://nvidia.github.io/cccl/unstable/cub/device_topk_requirements.html).

Test generic top-k separately from QSA-specific selection: the former can have callers that require sorted output, whereas QSA mask scatter cares primarily about membership. Keep a generic reference fallback and do not silently weaken its API.

### R13. Hidden-state handoff is wider than previously assumed

MTP uses `n_embd_out=hc*n_embd=10240`; at 150k, one complete F32 hidden stream is 6.144 GB decimal per direction, before host memcpy and verification storage. This is a logical payload, not proof all bytes cross a particular boundary in one transfer. A 640-row piece is 25 MiB. Log the actual producer backend and transfer sizes; the final layer/output placement must be verified before assuming same-device handoff.

Prototype a graph-owned device view for rows 0..T-2 plus a small device pending row for each sequence, consumed only after the producer event. Audit `verify_h`, `pending_h`, row reordering, and CPU API consumers: removing the initial memcpy alone does not remove later host retrieval. Prefer a narrow internal MTP path with host API fallback over changing public embedding ownership globally. Keep decoding latency/acceptance tests and long-prefix catch-up tests separate.

### R14. Placement changes only help measured stage imbalance

`llama-model.cpp` assigns layer devices by an `upper_bound` on cumulative split fractions over active layer slots, which can include additional/output slots. Derive actual layer boundaries from the model loader rather than rounding 48 times each percentage. No-pipeline time is approximately the sum of stage times; a balanced schedule matters differently once R2 creates overlap, when the slowest stage becomes central.

Build a small offline candidate list using measured layer cost and per-device late-prefix memory. Preserve the draft reserve. Evaluate one integer boundary move at a time, avoid unsupported tensor overrides that disable pipeline, and keep a decode constraint: moving work to accelerate prefill can make bandwidth-bound generation worse. Re-rank placement after R1/R3/R6, since those changes alter relative device strengths.

### R15. Foreign-runtime overlap has bounded scope

The existing foreign-copy helper rotates pinned slots, waits for destination slot completion, synchronizes the source backend, performs a blocking source read, and queues destination H2D plus an event. Only contiguous device-to-device foreign tensors enter it; host or noncontiguous tensors can take another fallback. Count both paths.

A minimal prototype should replace whole-source synchronization with completion of the specific producing stream/event and queue D2H in the source namespace, then upload completed bounded chunks in the destination namespace. A bridge must coordinate host-visible completion without treating an event object from one runtime as valid in the other. Respect callback restrictions; do not submit arbitrary CUDA API work inside a CUDA host callback.

Test chunking across 1/4/16 MiB as experimental sizes with one and two in-flight slots only after measuring transfer sizes and lifetime pressure. Whole hidden boundary payloads can be 25 MiB, so old tiny-AllReduce-message intuition is not enough. Pinned memory and independent streams are prerequisites for useful copy/compute overlap, not guarantees of overlap in this dependency chain. [CUDA 12.9 transfer guidance](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).

### R16. Graph reuse must respect prefix-dependent inputs

`llm_graph_input_qsa::can_reuse` requires exact n_kv/block/input dimensions. `llama_kv_cache::get_n_kv` pads the active view to at least 256-position granularity, so the shape changes repeatedly during prefill. In addition, R3's fallback disables CUDA graphs for affected backend graphs. Fix dispatch barriers before treating capture overhead as the independent primary bottleneck.

Prototype bounded shape buckets only if measured graph-build/launch time warrants it: supply actual valid length and generate exact masks so extra bucket positions are excluded. Bucket padding increases score/mask memory and can trigger OOM, defeating R11. Graph reuse also interacts with R2 input lifetime barriers. Keep a limited cache and measure hit rate, memory retention, and avoided CPU time. Batch alignment alone is a low-confidence finishing experiment, not a route to several-fold speedup.

### R17. PLE is not a 26.8 GiB transfer per batch

Metadata lists PLE layer `[1]`; hash heads are 16 and each head dimension is 160, giving a gathered vector width 2560. The large table remains resident in host RAM, while a 640-token F32 gathered output is about 6.25 MiB. The model graph builds one shared gather result; confirm actual scheduler copies, but do not assume PLE projections run in all 48 layers.

`llm_graph_input_ple::set_input` constructs a small per-token context vector and computes hashes, obtaining predecessors through `llama_kv_cache::get_prev_tokens`. For contiguous single-sequence text, reuse microbatch predecessor tokens directly and consult KV only at the first n-gram boundary, with a general fallback for EOS, gaps, rollback, and multimodal positions. Reuse small scratch vectors, batch/prefetch exact table gathers, and measure CPU gather versus hashing versus upload before deeper work. Avoid a new GPU hash subsystem unless that measured cost justifies it.

### Dependency and validation summary

- R11 validation and four-device accounting are prerequisites for credible comparisons, not the highest theoretical gain.
- R3 can unlock both grouped execution and CUDA graphs; do not separately add its graph benefit to R16.
- R4/R5/R7/R9 free different intermediates, potentially enabling R2 or a larger ubatch; only an actual allocation/liveness report establishes cumulative savings.
- R1 overlaps R4/R5/R7 and leaves dense MTP work (R8); evaluate the whole target+draft request.
- R6 changes state arithmetic/lifetimes and must preserve rollback before R2 can safely overlap recurrent microbatches.
- R10 adds persistent memory; accept it only if its saved compute is worth the lost workspace headroom.
- R13/R15 can remove waits but require actual byte/time accounting; neither establishes cross-driver P2P support.

This revision changes documentation only. It adds no runtime toggles, launches no GPU workload, and does not promote or commit a candidate.

## Hardware audit and generation extension

Audit time: 2026-09-06, approximately 16:30-16:38 Europe/Moscow. Read-only checks used both driver namespaces, PCI sysfs and configuration space, CPU topology, SMBIOS memory information, GGUF metadata, and the current source tree. No model server, compute/copy benchmark, clock change, PCI reconfiguration, BIOS update, or driver reload was performed. Existing idle telemetry is not a load profile.

### Actual GPU and host configuration

| Backend | Device / PCI address | Reported usable memory | Driver | Observed host path |
| --- | --- | ---: | --- | --- |
| CUDA0 | RTX 5080, `0000:01:00.0` | 16303 MiB | 580.178.04, open kernel module | CPU root port `00:01.0`, x8; upstream reports Gen3 maximum/target |
| CUDA1 | RTX 5070 Ti, `0000:02:00.0` | 16303 MiB | 580.178.04 | CPU root port `00:01.1`, x8; upstream reports Gen3 maximum/target |
| V100_CUDA0 | V100 SXM2 32 GB, `0000:03:00.0` | 32768 MiB | 580.159.03, isolated proprietary module | CPU root port `00:06.0`, negotiated Gen3 x4 |
| V100_CUDA1 | V100 PCIe 32 GB, `0000:09:00.0` | 32768 MiB | 580.159.03 | PCH root port `00:1d.0`, negotiated Gen3 x4 |

The Blackwell endpoints advertise 32 GT/s x16 capability, but their current upstream ports advertise only 8 GT/s x8, with target speed 8 GT/s. At idle the links are at 2.5 GT/s x8. Low idle speed is normal power management; the separately observed upstream capability/target is why this plan budgets Gen3, not Gen5. Do not diagnose permanent bandwidth from idle negotiated speed alone. The origin of the exposed Gen3 limit (firmware/platform configuration, bifurcation/riser arrangement, or another constraint) has not been established.

```text
i7-13700K / CPU root complex
  00:01.0 -> RTX 5080       [x8, exposed upstream max Gen3]
  00:01.1 -> RTX 5070 Ti    [x8, exposed upstream max Gen3]
  00:06.0 -> V100 SXM2     [Gen3 x4]
  chipset link -> Z690 PCH
    00:1d.0 -> V100 PCIe   [Gen3 x4, draft device]
    NVMe controllers / network / other chipset devices
```

Theoretical one-direction PCIe payload ceilings before packet/protocol overhead are approximately 7.88 GB/s for Gen3 x8 and 3.94 GB/s for Gen3 x4. These are calculated from link rate/encoding, not measured copy rates. A 25 MiB wide-hidden microbatch needs at least about 6.7 ms for one transfer over a Gen3 x4 bottleneck, before staging, synchronization, and contention. A single 40 KiB hidden row has a much smaller bandwidth floor; latency/waits can dominate that decode transfer.

Host:

- CPU: Intel Core i7-13700K, 8 SMT P-cores plus 8 E-cores, 24 logical CPUs, one NUMA node; 30 MiB shared L3. P-core siblings are `(0,1), (2,3), ... (14,15)`; E-cores are CPUs 16-23. Read CPU topology rather than the misleading aggregate threads-per-core value on a hybrid CPU.
- The launch mask `5555` selects `0,2,4,6,8,10,12,14`, one thread per P-core. It already avoids SMT siblings and E-cores for the configured ggml pool. It does not establish affinity of every server, sampler, driver, or helper thread.
- Motherboard: Gigabyte Z690 UD DDR4, BIOS F34a dated 2026-07-29 according to SMBIOS. Do not infer actual wiring or safe PCI settings solely from a generic motherboard diagram; the enumerated device tree above is the current authority. [Board specifications](https://www.gigabyte.com/se/Motherboard/Z690-UD-DDR4-rev-1x/sp).
- RAM: 80 GiB installed as 8+32 GiB in channel A and 8+32 GiB in channel B. SMBIOS configured speed is DDR4-3200, with 1-rank 8 GiB DIMMs and 2-rank 32 GiB DIMMs. The separate SMBIOS nominal `Speed: 2400` field is not the configured operating-speed field. OS-visible RAM is about 78 GiB.
- Balanced channel capacities and two 64-bit DDR4-3200 channels imply a theoretical 51.2 GB/s host-memory ceiling, not the CPU's often quoted DDR5-based 89.6 GB/s limit. Sustained gather/copy bandwidth has not been measured.
- At the idle audit, about 74 GiB was available, most occupied memory was reclaimable cache, and swap use was negligible. This does not prove the loaded-model request has no faults or memory pressure.
- CPU0 reported `powersave` governor and `balance_performance` energy preference. On this CPU that is not proof it stays at 800 MHz; inspect actual busy-core frequency and scheduling latency before changing a policy.

Nominal GPU memory bandwidth is 960 GB/s for RTX 5080, 896 GB/s for RTX 5070 Ti, and about 900 GB/s for the two V100 models. These figures are vendor peaks, not this application's effective bandwidth. They explain why V100 can remain useful for weight-streaming decode despite weaker modern low-precision compute. [NVIDIA GeForce comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/), [NVIDIA V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf).

### Corrected P2P assumptions

The current NVML read/write capability reports are:

| Pair | Read / write capability report | Interpretation |
| --- | --- | --- |
| RTX 5080 <-> RTX 5070 Ti | `CNS` / `CNS` | Reported chipset support limitation; direct working P2P must not be assumed |
| V100 SXM2 <-> V100 PCIe | `OK` / `OK` | Candidate for same-runtime PCIe peer copies; actual CUDA transfer route/rate still unmeasured |
| Blackwell <-> V100 | Not exposed as a common pair by either namespace | Current foreign-runtime backend explicitly stages through host memory |

Both within-runtime topology reports show PHB, not NVLink. The V100 SXM2 NVLink query reports all links inactive. SXM2 packaging does not establish an active NVLink connection in this machine. No assertion of cross-runtime P2P follows from V100-to-V100 capability.

In `ggml_backend_cuda_cpy_tensor_async`, same-runtime copies between physical devices call `cudaMemcpyPeerAsync`, while foreign-runtime contiguous GPU copies use the pinned helper. Calling a peer-copy API does not prove that hardware transferred the data without host staging. `GGML_CUDA_NO_PEER_COPY` is OFF in the current build cache. Explicit peer enablement under `GGML_CUDA_P2P` checks `cudaDeviceCanAccessPeer`; this is not a knob that overrides the physical/driver limitation.

First future topology test should query `cudaDeviceCanAccessPeer` in each namespace, then time isolated bidirectional and simultaneous transfers with integrity checks. Compare small 40-80 KiB decode payloads and 25 MiB prefill payloads. Run no bridge-wide all-pairs test using incompatible runtime handles. Capture actual route-related diagnostics and host traffic where supported. Do not disable ACS/IOMMU, force link retraining, or change firmware in this task: those are separate security/stability decisions. Read ACS redirect bits were enabled on inspected CPU root ports; this is a fact, not proof that disabling them fixes P2P.

### Power, clocks, and thermal budget

Read power limits were 360 W, 300 W, 300 W, and 250 W respectively. V100 application clocks were 1290/877 MHz on SXM2 and 1230/877 MHz on PCIe (graphics/memory); reported maximum graphics clocks were 1530 and 1380 MHz. These application/max values do not establish sustained loaded clocks. Both V100s have ECC enabled. Blackwell display-active state was Disabled; there is no measured desktop VRAM saving to promise.

On the next benchmark, record loaded graphics/memory clocks, power, temperature, clock-event reasons, and CPU package behavior throughout both prefill and decode. Only if a real limit appears should we consider a supported application-clock/power policy within existing equipment limits, with separate approval. Do not disable ECC, overclock, undervolt, raise power limits, or alter fans as an automatic optimization. Combined configured GPU power limits alone total 1210 W; PSU/cooling capability was not inspected and must not be assumed from that sum.

### A decode bandwidth model based on the actual GGUF

Read tensor payloads from all four target shards, excluding the separate multimodal projector:

| Tensor class | Stored bytes | Why it matters |
| --- | ---: | --- |
| Routed expert tensors | 77017907200 | Only selected experts are read per token; a large prefill can touch nearly all |
| Other tensors excluding the three rows below | 4154723840 | Includes dense/shared/HC and auxiliary tensors; not all necessarily execute each token |
| Output projection, Q8_0 `[2560,248320]` | 675430400 | About 644.1 MiB; full vocabulary scoring can be a material decode phase |
| Token embedding, Q8_0 `[2560,248320]` | 675430400 | A token lookup does not read the whole matrix |
| Host PLE table, IQ4_NL `[160,320001536]` | 28800138240 | Gather only selected rows; not 26.8 GiB of traffic per token |

For equal-sized expert slots with 10 of 512 used, the routed-weight payload scale is about 1.504 GB per token if every corresponding expert projection executes once. This is a structural estimate, not measured HBM traffic: reuse across verification rows, caches, fusion, duplicated loads, draft layers, and routing distribution all change it. Add the **actually executed** dense/HC/output weights and activation/KV traffic; do not divide the complete 76.9 GiB resident model by bandwidth and call that decode time.

Conversely, do not add the four GPU bandwidth figures and divide by total bytes for a single layer-split token. The dependent token path traverses stages serially. A useful first model is `sum_i(bytes_i / effective_bandwidth_i + compute_i + launch_i) + exposed transfers/waits + sampling/draft overhead`, with overlap credited only when observed. The output head alone has an ideal approximately 0.75 ms weight-read time at 900 GB/s if fully streamed; that is a lower bound, not a measured head latency.

The old 128-token decode fixture uses temperature 0, seed 4242, no prompt cache, and nonstreaming output. Its 64-65 tokens/s is not a benchmark of stochastic sampling, grammar, streaming latency, or generation after a 150k prefix. The new generation plan must cover all those distinctions.

## Generation ranking for this machine

Ranks below target single-request accepted-token speed at fixed model/precision and fixed reference MTP depth 1. Long-context effects are marked separately. R1-R17 remain the prefill ranking; G items reuse them where relevant. These are hypotheses ordered by plausible benefit and source evidence, not measured improvement percentages.

| Rank / ID | Candidate | Short-prefix generation | Long-prefix generation | Prefill interaction |
| --- | --- | --- | --- | --- |
| G1 | Tune/fuse quantized decode and two-row verification kernels | High if weight/launch phase dominates | Medium-high | Keep separate from R3 prefill dispatch |
| G2 | Reduce MTP/sampling synchronization and device-host handoffs | Medium-high if host gaps dominate | Medium | R13/R15 help both; preserve prefix catch-up |
| G3 | Sparse QSA decode and compact/incremental indexer | Medium | High architectural opportunity | Reuse R1/R4/R5/R10, with decode-specific dispatch |
| G4 | Tune one/two-query dense attention for V100 and Blackwell | Low-medium | Medium-high, especially draft | R7/R8; do not substitute prefill kernels blindly |
| G5 | Output-head execution and target backend sampling | Medium, cheap first experiment where supported | Medium | Usually little effect on raw prefill |
| G6 | HC/normalization/routing fusion and graph launch overhead | Medium | Medium | R9/R16 can help both |
| G7 | Recurrent one-step update and rollback-state traffic | Medium if state phase dominates | Medium | Preserve R6 chunked-prefill state contract |
| G8 | Placement based on decode costs and verified copy routes | Conditional medium | Conditional medium | Can conflict with optimal prefill placement |
| G9 | Host affinity, spin/wait policy, buffer allocation, streaming | Low-medium unless CPU gaps dominate | Low-medium | CPU contention can hurt prefill |
| G10 | Remove observed power/frequency throttling within safe policy | Conditional | Conditional | Requires loaded telemetry and separate approval |
| G11 | Adaptive draft depth/threshold after faster primitives | Conditional; not default scope | Conditional; likely sensitive to dense draft cost | Preserve depth-1 reference and report TTFT |
| G12 | Selective expert/dense-branch parallelism or exact replication | High theoretical ceiling, low feasibility confidence | Conditional | Major memory/scheduler project, defer |

### G1. Optimize the paths actually used by one and two tokens

Source anchors: `mmvq.cu`, `mmvq.cuh`, `ggml-cuda.cu:ggml_cuda_mul_mat_id`, and the gate/up fusion matcher. `MMVQ_MAX_BATCH_SIZE` is 8. Both Volta and Ada-or-newer return that bound for quantized expert MMVQ; thus current one-token draft and commonly two-row target verification do not have the same large-microbatch fallback as R3. A prefill dispatch fix must not be advertised as a direct decode kernel fix.

Volta-specific MMVQ tuning already exists: some single-row quant types use two warps rather than four. Q4_K is among the tuned cases, while Q5_1 uses the default four-warps branch in that table. The real model has Q4_K gate/up and many Q5_1 down projections, so benchmark those separately with their actual input/output widths. The multi-token `mul_mat_vec_q` expert kernel maps a query token to `threadIdx.y` and reads that token's expert ID; do not assume adjacent verification tokens reuse all weights.

First inspect whether existing gate/up/bias/GLU fusion matches this graph for both one and two rows. Then tune rows per block, warps, vectorized weight loads, and activation-quantization reuse for sm_70 and sm_120a independently. A two-row verification microkernel can reuse weights only for experts actually shared by the rows; measure the routed expert intersection and include regrouping cost. A resident grouped queue may help skewed expert populations but can be launch overkill for ten active experts.

Preserve formats and decode numerics. No conversion to FP4, INT8 weight requantization, or reduced active-expert count. Compare target cycle time, draft cycle time, accepted tokens/cycle, and memory transactions where available. Promote a tuning change only if it improves the entire request and does not regress the prefilling dispatch.

### G2. MTP orchestration: optimize accepted tokens per complete cycle

Measure a timeline with `draft decode -> draft candidate selection -> target verification -> accept/reject -> state update/rollback -> next draft`, including hidden-row copies and CPU waits. Existing `common_sampler_sample` begins with `llama_synchronize`, and hidden-output access also has completion requirements. Some waits are redundant completion checks, others are unavoidable data dependencies; remove only the former or reduce their scope.

For depth 1 and a consistently proposed single draft token, a simplified expected output is `1 + acceptance_probability` tokens per cycle. With a historical acceptance near 0.789 that is about 1.789, but use measured counters because rejected proposals, early stopping, confidence thresholds, and bookkeeping alter actual cycles. Optimize `cycle_time / accepted_output_tokens`, not the target-only tokens/s number. The historical depth sweep already found depth 2/3/5 slower, so increasing depth is not the first action.

Prototype device-local pending/verification hidden rows (R13), compact sampling results, and batched host-side metadata updates. At NMAX=1 the draft loop still obtains its hidden output row before it decides no further draft step is needed; inspect whether that retrieval/copy has consumers later in the iteration and can be avoided or delayed. Do not delete exports required by accept/process or the next iteration. Reuse vectors and batch storage after lifetime analysis.

At long context, draft dense attention may dominate even if copies are removed. Count prefix catch-up separately from steady-state draft calls. Do not fabricate speculative acceptance, skip draft KV preparation, or change rejection semantics to create a better benchmark.

### G3. Long-context generation has an indexer problem too

R1/R4/R5/R10 apply to each generated token as well: a small query count does not make scanning/repooling a large prefix disappear. At 150k, a single dense F16 mask is only about 293 KiB, so its memory-capacity importance falls sharply versus prefill; the repeated scans, launches, indexer scoring, and actual attention remain relevant. Rank decode opportunities by time, not the prefill buffer table alone.

Build a one/two-query sparse attention specialization with direct selected Q8 loads, rather than using a high-query-count kernel. Cache immutable completed pooled keys with rollback-safe invalidation; keep the incomplete block exact. A compact per-query causal/tail description can replace host bias filling. Short-prefix fallback is essential: selection/gather overhead can exceed the work saved when N is small. Preserve current top-k membership semantics and do not assume stable ties that the current CUB path has not guaranteed.

### G4. Decode attention dispatch differs across architectures

`fattn.cu:ggml_cuda_get_best_fattn_kernel` routes eligible Blackwell quantized KV attention with <=2 queries to the vector kernel. For Volta with GQA ratio 12, the effective grouping is 4; one/two queries pass the <=16 test for the tile kernel rather than the MMA path. Confirm these conditions for actual tensors, including padding and masks. Therefore, the full-KV FP16 conversion finding from R7's MMA path must not automatically be charged to every decode attention call.

Compare existing vector/tile/MMA kernels at query counts 1 and 2, head dimension 256, two KV heads, and actual prefix lengths. Tune KV partition count, warp count, reduction/fixup traffic, and load reuse across grouped query heads. Preserve q8_0 and the target/draft masks. A targeted shape/device dispatch table with a safe fallback is preferable to one global attention choice.

The draft's dense attention is a distinct target for G4 after sparse trunk attention improves. A cached FP16 KV mirror could avoid conversion only where that conversion exists, and its substantial memory cost would compete with all other improvements; selected/tiled dequantization is the safer initial avenue.

### G5. Target sampling and the large vocabulary

`common_params_sampling.backend_sampling` defaults to false, whereas draft backend sampling defaults to true. The MTP initializer already attempts a GPU top-k(10) chain and falls back with a warning if unsupported. The target has experimental `--backend-sampling`; the launcher does not currently request it.

First future experiment: target backend sampling for the existing temperature-0 fixture, comparing exact token IDs, per-cycle latency, and actual D2H payload. The target logits row is 248320 * 4 = 993280 bytes, about 0.95 MiB; fetching two verification rows can be meaningful over the actual host links. GPU argmax can reduce output traffic, but it does not remove the 644 MiB output projection or its full-vocabulary arithmetic.

For stochastic sampling, preserve the requested sampler order, penalties, RNG behavior/distribution, and speculative verification contract. `common_sampler_sample_and_accept_n` has both token-match and distribution-based variants; the latter needs constrained target probabilities and can sample a residual distribution on rejection. Do not send only argmax/top-k if the verifier still needs the full distribution. Grammar, reasoning-budget, and pre-sampling probability requests can disable or constrain backend sampling. Validate the actually installed backend chain and server behavior, not just flag parsing.

Further prototype: tile the output projection and reduce exact greedy maxima or required candidate statistics on-device, avoiding intermediate logits materialization only when the requested API and verification permit it. All vocabulary rows must still be scored; vocabulary pruning changes the model distribution. Inspect whether target/draft output weights already share storage or are correctly resident before proposing a duplicate. Sharing an allocation does not guarantee cross-call HBM cache residency.

### G6. Decode fusion, CUDA graphs, and dependent launches

Reuse R9 HC combine/gating fusion for one/two tokens and confirm existing norm/scale/GLU fusions first. Small graphs can be dominated by launches even when the tensors are small. Keep graph capture/reuse per stable decode shape and sequence state, with explicit invalidation for graph topology changes.

Current build cache enables CUDA graphs. The normal CUDA target has sm_70 and sm_120a code; the isolated V100 target has sm_70. The source already has Programmatic Dependent Launch helpers, enabled by default **if compiled in**, and checks kernel PTX eligibility. Do not promise that setting `GGML_CUDA_PDL=1` adds a feature already active, or that it can work on sm_70. Audit actual compile definitions/runtime behavior and compare on/off only on eligible Blackwell kernels.

A larger persistent decode kernel or graph-level fusion is a later option if launch gaps remain after existing fusions. It increases state ownership and cancellation complexity and cannot combine separate CUDA runtimes into a single CUDA graph. Avoid changing arithmetic or request scheduling to make an artificial fully captured loop look faster than the server.

### G7. One-step recurrence and rollback-state copies

Each recurrent layer's F32 state snapshot is 3 MiB, across 36 layers before extra rollback slots. The current backend already matches GATED_DELTA_NET followed by snapshot copy and has `ggml_cuda_op_gated_delta_net_fused_cache` to write cache snapshots directly. Verify this matcher hits the actual graph before proposing that optimization again.

Tune the existing single-token update for state layout/coalescing, reduction, registers, and snapshot writes. Keep output/state accumulation precision and last-K rollback snapshots. The model uses sigmoid gated normalization, not the SiLU gate used by some related architectures; importing a generic GDN fusion without that distinction changes results.

An index/ring-based rollback representation could avoid copying state to rearrange slots, but only if the actual state-memory API and consumers allow it; the mandatory new-state computation is not removable. Check numerical equality/tolerance after every accept/reject and sequence reset, not just final generated text. Keep chunked-prefill R6 as a separate dispatch path with the same state contract.

### G8. Placement and same-runtime boundaries

Use the actual topology rather than assuming a fast Blackwell peer link. Current ordering groups the two Blackwells and two V100s, limiting the main layer chain to one foreign-runtime boundary; changing to alternating runtimes can create many host-staged boundaries. V100-to-V100 P2P support is promising but must be verified under the CPU/PCH route and concurrent chipset traffic.

For decode, choose layer boundaries using measured per-layer weight bytes, kernel latency, QSA/recurrent cost, and memory headroom. V100 bandwidth being near Blackwell bandwidth means moving all possible work away from V100 is not automatically better. Moving the draft between V100s can trade stronger SXM2 compute/CPU-attached PCIe against trunk memory pressure and handoff locality; preserve the current default until a complete paired test wins both modes.

Output-head/draft co-location can save copies or improve reuse but costs memory and changes the final critical stage. Do not move a 644 MiB head or an expert layer based only on idle free VRAM: reserve at 160k, workspaces, long prompt, and decode all must fit. Use integer layer mapping, not just TS percentages.

### G9. CPU and service latency on this hybrid desktop

The current compute-pool P-core mask is sensible and already present. First sample thread CPU assignment, migrations, CPU run queues, active frequencies, sampling time, and exposed CUDA wait time during generation. Inspect main inference/sampling thread affinity separately; dedicate a physical P-core only if it removes observed contention, leaving the sibling idle for this latency-sensitive role if beneficial.

Test existing poll/spin and CUDA scheduling policies only as bounded, opt-in experiments. Spinning can reduce wakeup latency but consume cores needed by QSA preparation, drivers, or the server and hurt prefill. The source's explicit spin workaround targets cc121 integrated GPUs, not this machine's cc120 or sm_70; do not copy it blindly.

Remove repeated host allocations, formatting, and unnecessary data conversion only where profiler evidence identifies them. Test production streaming separately: report time to first token, accepted-token interarrival p50/p95, wall time, and server compute metrics. Excessive per-token debug logging, event synchronization profiling, tokenization/detokenization, JSON serialization, and client backpressure can distort observed generation speed. Keep raw model throughput and network-visible latency as distinct measurements.

Keep model data page-resident where feasible without locking more memory than the 80 GiB host can sustain. Audit major/minor faults and memory pressure during real runs, especially PLE and pinned staging. No recommendation for CPU experts, storage streaming, huge pinning pools, or NUMA interleaving follows from a single-node topology. The PCH-attached V100 also shares upstream resources with NVMe/network, so avoid benchmarking during unrelated heavy I/O and record interference.

### G10. Hardware policy opportunities require evidence

Audit the exposed Gen3 upstream limit and possible firmware link-speed configuration without changing it. A higher reliable link mode on the same physical equipment could help prefill transfers, but support through the actual risers/bifurcation is unknown; it is not part of the predicted software gain. V100 remains a Gen3 endpoint regardless, and x4 cannot become x16 by a runtime flag. Moving cards/cables is outside the fixed-layout experiment and needs explicit scope approval.

Clock tuning is conditional on sustained throttling, not idle P8/P0 or application-clock fields. Never present advertised max clocks as free performance. Supported policy changes need measured thermals, stability, power headroom, repeatability, and a rollback record. Disabling ECC/ACS/IOMMU or changing model precision is excluded, not a low-priority performance tweak.

### G11. Revisit speculative policy only after primitive costs change

Keep NMAX=1 as the reference. Historical deeper drafting was slower and cannot be undone by optimism about acceptance. Once target verification, draft attention, and CPU overhead change, remeasure depth-1 cost and acceptance by prefix length before considering any new policy.

An optional later controller could choose depth or skip a low-benefit proposal using observed expected accepted tokens divided by predicted cycle cost, with hysteresis and bounds. This must retain exact target sampling/verification, must not skip required draft-state catch-up, and must report both prefill and generation. It is a separate scheduling-policy experiment, not a change already authorized for deployment. Any long-context draft attention correction aimed at acceptance is likewise a separate numerical/quality validation task.

### G12. Larger parallelism options on the same four cards

These preserve model files but are major execution-design changes and rank below the measured local optimizations:

- **Parallel experts within a layer:** distribute selected experts over devices and combine their weighted outputs while preserving routing. Small decode activations make communication feasible in principle, but frequent per-layer barriers over Gen3 x4/host staging may outweigh parallel weight reads. Start with a cost model using the verified V100 peer pair; do not assume the Blackwell pair has direct P2P.
- **Hybrid parallelism:** retain layer groups across runtime domains and use expert/tensor work sharing inside a verified same-runtime group. Avoid global AllReduce per sublayer across the two driver namespaces. Existing graph and memory ownership would need extension, and recurrent/HC operations remain ordered.
- **Exact hot-weight replication:** keep a bounded replica of measured frequently used experts or a dense branch on a less busy device, without approximation or CPU fallback. Replication consumes scarce VRAM and must cover invalidation/ownership; loading experts dynamically over Gen3 x4 can be worse than local execution. The current model already keeps experts resident, so older hot/cold CPU-store features are not an automatic benefit here.
- **Independent branch overlap:** shared-expert/routed-expert or gate/up computations may run concurrently if inputs are ready and resource use permits. Same-device concurrency can reduce bandwidth/occupancy rather than improve it; dependency and buffer ownership must be explicit. Do not change weighted-sum semantics.
- **Multiple independent requests:** continuous batching can improve aggregate throughput through weight reuse, but `-np 1` single-request latency is the current objective. This is a distinct service benchmark with extra KV/recurrent memory and cannot be reported as faster individual generation.

No multi-GPU bandwidth-sum speedup is promised. Advance one of these designs only if profiling demonstrates that serial device execution, rather than local kernels/host overhead, remains the main limit after simpler work.

## Combined execution plan and measurement contract

1. **Freeze evidence:** archive hardware queries, source patch/build hashes, model tensor metadata, script environment, and actual integer placement. Before using `bench/cycle_flash.sh` again, replace its broad `pgrep -x llama-server` kill and first-PID selection with ownership of the exact launched process and a dedicated port. The current harness can affect unrelated servers; do not use it unchanged in a shared session.
2. **Validate existing candidate:** finish R11 correctness tests and paired throughput tests in a separate runtime directory. Do not silently promote patched libraries or change the working launcher.
3. **Measure the right cycles:** complete R0 four-device/host profiling; collect one-token draft, two-row verification, accepted/rejected cycles, pure decode reference as a diagnostic, and short/long prefill. Diagnostic profiling that synchronizes operations cannot establish production latency or overlap.
4. **Cheap generation probes:** verify existing gate/up/state-copy fusions, graph reuse and draft sampling; test target backend sampling only for supported exact chains; validate peer capabilities/routes with scoped integrity/copy tests. No driver/BIOS/security-setting changes.
5. **First compute changes:** investigate R3 for prefill and G1/G2/G5 for short-prefix generation, based on measured fractions. Keep separate shape/device guards. Then assess G3/G4 plus R4/R5/R1 for long prefixes and R6/G7 for recurrence.
6. **Memory and overlap:** use proven buffer savings to assess R2; evaluate placement G8 and transfer overlap only against actual topology. Revisit persistent caches and large parallelism last.

For each candidate record `(PP_5k, PP_30k, PP_100k, PP_150k, TG_short, TG_long, TTFT, ITL_p95, peak_memory_per_device, acceptance, output_correctness)`. Preserve model files, KV precision, context requirement, routing, and sampling constraints. Compare at least five interleaved short runs plus appropriately repeated long runs; monitor loaded clocks and host interference. Use real stochastic/grammar workloads in addition to the existing greedy fixture.

A PP improvement with a repeatable TG regression, or a TG improvement with an unapproved PP/context regression, is not the combined default. Keep such a result as a labeled Pareto tradeoff for user choice. No fixed percentage of regression is automatically acceptable. Sparse/recurrence/sampling changes need stronger correctness checks than one matching 128-token response, including state rollback and exact required candidate membership.

Hardware query reproduction (read-only; future copy/compute benchmarks are deliberately not included):

```sh
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
env LD_PRELOAD=/opt/nvidia-v100/lib/nvidia_v100_redirect.so LD_LIBRARY_PATH=/opt/nvidia-v100/lib /opt/nvidia-v100/nvidia-smi topo -m
env LD_PRELOAD=/opt/nvidia-v100/lib/nvidia_v100_redirect.so LD_LIBRARY_PATH=/opt/nvidia-v100/lib /opt/nvidia-v100/nvidia-smi topo -p2p r
env LD_PRELOAD=/opt/nvidia-v100/lib/nvidia_v100_redirect.so LD_LIBRARY_PATH=/opt/nvidia-v100/lib /opt/nvidia-v100/nvidia-smi topo -p2p w
lspci -Dtv
lscpu -e=CPU,CORE,SOCKET,NODE,MAXMHZ
sudo -n dmidecode -t 17
sudo -n lspci -Dvv -s 00:01.0
sudo -n lspci -Dvv -s 00:01.1
sudo -n lspci -Dvv -s 00:06.0
sudo -n lspci -Dvv -s 00:1d.0
```

This extension is documentation/research only. It does not claim all conceivable optimizations have been exhausted; it covers the identified hardware constraints and executable paths, with explicit tests for the remaining unknowns. No runtime, launcher, hardware setting, or user source change was modified, and no commit/push was made during this extension.

## Evidence files

Local artifacts may be temporary; paths below identify the actual runs and should be archived before cleanup of `/tmp`.

- Original run: `/tmp/flash-sep06-baseline.log`.
- Nsight run: `/tmp/flash-sep06-nsys.log`, `/tmp/sep06-flash-baseline.nsys-rep`, `/tmp/sep06-flash-baseline.sqlite`.
- Initial sort diagnostic: `/tmp/sep06-sort-profile.log`, `/tmp/sep06-profile-pp.json`.
- Candidate short run: `/tmp/flash-sep06-sort8-ub640.log`, `/tmp/sep06-sort8-prefill.json`, `/tmp/sep06-sort8-decode.json`.
- Saved reference responses from the Nsight run: `/tmp/sep06-baseline-prefill.json`, `/tmp/sep06-baseline-decode.json`.
- Candidate long runs: `/tmp/sep06-sort8-long.log`, `/tmp/sep06-sort8-long30k.json`, `/tmp/sep06-sort8-long150k.json`.
- Final CUDA build: `/tmp/sep06-final-build.log`; test-target build: `/tmp/sep06-test-build2.log`.
- Request fixtures: `/home/despc/llama.cpp/bench/prompt30k.json`, `prompt100k.json`, `prompt150k.json`.

Candidate launch used for the completed long tests, recorded for reproduction only (do not run against restored original libraries and expect the new environment variable to work):

```sh
env HOST=127.0.0.1 GGML_CUDA_SORT_PREFILL_CHUNK_MIB=8 UBATCH_SIZE=640 bash /home/despc/llama.cpp/start_qwen_flash-4gpu-mtp.sh
```

## Execution log: 2026-09-06 evening, R11 validated and re-attributed

This section records measurements, not intentions. Everything below was run on
the fixed reference configuration with the harness fix described first.

### The harness was unsafe and is fixed

`bench/cycle.sh`, `bench/cycle_flash.sh` and `bench/prof.sh` each began with
`pgrep -x llama-server` followed by `kill`, then selected their own server with
`pgrep ... | head -1` on a fixed port 8080. In a shared session that reaches
unrelated servers and can measure the wrong process.

Launch and teardown now live in `bench/harness.sh`. It refuses to start when the
chosen port already answers `/health`, starts the launcher with `bash script &`
so that `$!` is the server itself (every launcher ends in `exec`), stops exactly
that PID from an `EXIT` trap, and reads memory from both driver namespaces. The
port defaults to 18080 so a benchmark never collides with the deployment. The
three harnesses are now thin wrappers over it.

Two gaps in coverage were also closed. `bench/longctx.sh` runs the long half of
the acceptance protocol in one pass, and `bench/decode-after-100k.json` and
`bench/decode-after-150k.json` generate 128 tokens after a long prefix — the
existing long fixtures ask for a single token, so nothing measured sustained
generation at a full context.

The launchers gained a `RUNTIME_DIR` variable, defaulting to the deployed
`fork_v100`, so a candidate build can be measured without disturbing the
deployment.

### R11 correctness

`test-backend-ops` on all four devices (V100 SXM2, V100 PCIe, RTX 5080,
RTX 5070 Ti), at caps default/16/8/4:

| Operation | Cases per backend | Result |
| --- | ---: | --- |
| TOP_K | 527 | pass at every cap |
| ARGSORT | 98 | pass at cap 8 |

`GGML_CUDA_SORT_PROFILE` confirms the new 33024x64 cases reach the chunked path
and produce a partial final chunk: 63+1 rows at cap 8 MiB, 31+31+2 at cap 4 MiB.

The profile output also settles what the cap actually bounds. For a 63-row chunk
of 33024 columns the input is 8.3 MiB while CUB workspace and the index/key
arrays are about 15.9 MiB each — roughly four times the capped quantity, which
is why an input cap is a weak lever on total memory even though it works.

### R11 speed: the cap contributes nothing, the microbatch contributes everything

Four configurations, interleaved forward then reversed, five cycles each, eight
warm samples per configuration. `ref` is the deployment; `cand` is the rebuilt
CUDA libraries with the cap left off, which isolates the rebuild itself.

| Configuration | 5k prefill, warm | Generation, warm | vs reference |
| --- | ---: | ---: | --- |
| ref-512, deployed libraries | 478.0 +- 0.6 | 63.9 +- 0.2 | — |
| cand-512, rebuilt, cap off | 478.0 +- 0.6 | 64.1 +- 0.2 | pp -0.0%, tg +0.4% |
| cand-512, cap 8 MiB | 477.6 +- 0.4 | 64.2 +- 0.2 | pp -0.1%, tg +0.4% |
| cand-640, cap 8 MiB | 504.7 +- 0.6 | 64.9 +- 0.2 | pp +5.6%, tg +1.5% |

Reversing the order reproduced every figure, so this is not an ordering artefact.

**The earlier 5.6% was attributed to the wrong factor.** At a fixed ubatch the
cap changes nothing measurable: 477.6 against 478.0, with a run-to-run spread of
0.6. The whole gain comes from ubatch 640. The rebuild alone is also neutral,
which is the control that makes the comparison meaningful.

That does not make the cap useless, it makes it a different kind of change. For
the real QSA top-k shape the cap bounds peak sort scratch to about 30 MiB
regardless of microbatch, where the 64 MiB default reaches roughly 254 MiB:

| Prefix | Microbatch | Cap 64 MiB | Cap 8 MiB |
| ---: | ---: | --- | --- |
| 150272 | 640 | 111 rows/chunk, 6 chunks, ~254 MiB | 13 rows/chunk, 50 chunks, ~30 MiB |
| 160256 | 640 | 104 rows/chunk, 7 chunks, ~254 MiB | 13 rows/chunk, 50 chunks, ~32 MiB |

At a 5k prompt the sort is small enough that neither cap binds, which is exactly
why the two are indistinguishable above. The cap is a memory enabler for ubatch
640 at long context, not a speed optimization, and it should be promoted only
together with the larger microbatch and only if the long-context run holds.

Steady-state memory after the 5k prompt, all four devices: 15328/14556/29669/29772
MiB at ubatch 512 against 15612/14842/29957/30122 MiB at 640.

Evidence, including source revision and binary hashes for both runtime
directories, is in `/home/despc/llama.cpp/evidence/2026-09-06-r11.md`.

## Execution log: 2026-09-06 night, where the ranking was wrong

Prefill on the fixed reference configuration went from 478 to 811 tokens/s at a
5k prompt over this session, 412 to 661 at 30k and 310 to 441 at 100k, with
generation unchanged throughout and greedy output byte-identical at every step.
None of it came from the candidates the ranking put first.

### What actually paid, and why the ranking missed it

Both wins were dispatch mistakes on Volta, not missing capability.

The first (R3) was already described above: the MMQ batch-size rule rejects any
batch of 64 or more on hardware with FP16 tensor cores but no Turing MMA, which
sent every prefill expert projection into a host-synchronising fallback --
284199 individual kernel launches on the two Teslas over a single 5000-token
prefill.

The second was not in the ranking at all. The MMQ tile table is chosen by
architecture, and the first branch a compute capability of 700 satisfies is
`>= VOLTA`, which hands it the Ampere table: 128-wide tiles, occupancy one,
stream-k on. Volta does not run that layout -- without Turing MMA the kernel
takes the DP4A path, the one `pascal_dp4a` was written for, with 64-wide tiles
and occupancy two. It never reached that table because the Volta branch is tested
before the DP4A branch. Raising the Ampere branch to `>= TURING` was worth 32%.

The lesson is not about these two lines. It is that the ranking was built from
source reading plus one 5k profile, and the largest items in it were structural
projects while the actual defects were in three-line predicates that no amount of
ranking would surface. A cheap route census -- which route each dispatch took,
how many launches it spent -- found both in minutes.

### R1, sparse attention: blocked by fragment shapes, not by a flag

The chain is fully implemented in the backend. Three things kept this model off
it. The graph passed `n_kv_max = 0` behind a TODO: fixed. The eligibility list
lacked 256/256: added, at the grouping of 8 that is the only one configured for
that head size.

The third cannot be fixed here. Volta's MMA fragments exist only at `I == 32`;
`tile::supported()` traps for anything narrower, surfacing as an unspecified
launch failure. The sparse path needs one query column per block, so at this
model's GQA ratio of 12 it can never reach 32 columns. The `ncols1*ncols2 < 32`
guard is a correctness bound, not the compile-time prune it resembles. Sparse
attention on Volta needs new fragment shapes; that is the whole project, and its
ceiling is attention's 17% share at 30k.

Enabling it for the two Blackwells that can take it is neutral end to end --
754.3 against 752.8 tokens/s at 5k -- because they hold about a sixth of the GPU
time and attention is 8.4% of theirs. Committed as correct, not as faster.

### R3 second stage: the premise was wrong, the fix was elsewhere

The plan proposed dequantising to FP16 and using Volta's tensor cores for roughly
twice the arithmetic. Measurement says arithmetic was never the constraint: over
the expert shapes, an eightfold increase in tokens costs 1.25x the time. What it
did reveal is that the expert path achieved 136 GB/s where a dense matmul reading
the same 450 MiB with the same ten columns reaches 374.

The cause was padding, not bandwidth. `mul_mat_q_switch_J` sizes the column tile
from `ncols_max`, which for MUL_MAT_ID is the batch, but the tile is spent per
expert -- and 512 experts with 10 chosen per token leave each expert about ten
columns of a 64-wide tile. Fitting the tile to the expert's share: Q4_K gate/up
3644 -> 1627 us at 512 tokens, Q5_1 down 4305 -> 2708.

### The critical path is not where op profiles put it

That change removes 13% of the Teslas' GPU time and gains 1.3% of prefill. The
discrepancy is the most useful result of the session, and the first explanation
tried -- that a fifth of prefill was host overhead -- was wrong, arrived at by
summing per-operation GPU times, which is exactly what this document warns
against: with a synchronise after every operation each measurement carries its
own launch latency, and on thousands of small nodes that dominates the sum.

`LLAMA_UBATCH_PROFILE` measures the host timeline instead. On the same 30k
prefill: build 417 ms, set_inputs 252 ms, apply 9 ms -- under 1.5% together --
against 90.5% inside `graph_compute` and 8% in the trailing synchronise.

Two consequences. R4 is dead: preparing the indexer's block bias on the GPU can
save at most the 0.6% that all input preparation costs. And the graph was reused
once in 125 ubatches, because the indexer input shape follows the cache length,
so CUDA graphs are unavailable for prefill without bucketing it (R16).

### R2, pipeline parallelism: the slots were the easy half

The reserve fails at every startup -- 1882.99 MiB on device 0 -- and the
scheduler then retries with pipelining off entirely, so this deployment has never
pipelined. `GGML_SCHED_N_COPIES` makes the count a runtime choice; at two slots
the reserve succeeds and pipelining stays on.

It buys 0.5% of prefill and 1.2% of generation. As this document predicted,
removing the allocation fallback does not produce overlap: `process_ubatch` still
synchronises the scheduler before overwriting reused graph inputs. Three slots
configure and start, then fail during inference, so the override accepts only
powers of two.

### Revised standing

At 30k on the Teslas, after both dispatch fixes: MUL_MAT_ID 48.0%, FLASH_ATTN_EXT
17.0%, MUL_MAT 16.9%, GATED_DELTA_NET 4.0%, TOP_K 2.7%. Scaling from 5k to 30k,
experts grow 6.4x with the token count while attention grows 30x and top-k 42x,
so the two regimes need different work: experts up to about 50k, attention near
the context limit.

Ranked by what the measurements now support:

1. Volta MMA fragment shapes narrower than 32 columns. Unblocks R1, and R1 is the
   only candidate whose value grows with context, which is where this deployment
   is weakest (441 tokens/s at 100k against 811 at 5k).
2. Graph reuse across ubatches (R16), which would also make CUDA graphs available
   for prefill. Needs the indexer input bucketed by cache length; this document's
   warning about bucket padding and memory stands.
3. Real pipeline overlap (R2 second half), now that the slots fit.

Not worth pursuing on this evidence: R4 (host bias, 0.6% ceiling), R5 (the
indexer score matmul is 94.9 ms, 0.3%), R12 (top-k is 2.7%), and dequantised FP16
experts (the expert path is not arithmetic-bound).

Unrelated defect found and not fixed: FLASH_ATTN_EXT with max_bias 8.0 and a
sparse mask hint aborts the Tesla backend with an unspecified launch failure. It
reproduces on the libraries deployed before this session. This model uses no
ALiBi.
