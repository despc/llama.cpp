# Flash-Next four-GPU optimization plan

> 2026-09-08 source rollback: the retained Volta dispatch/tile optimizations, sparse execution overrides, and sort-workspace override described below have now been removed at the user's request. Their dispositions and throughput figures below are historical. See [CUDA numerical optimization rollback](CUDA-numerics-rollback.md) for the current source state and verification limits. No binaries have been rebuilt or deployed.

Updated: 2026-09-07. Source reviewed: `79f26bba9`, after merging upstream `67672dc5b` and retaining `91a4d956e` (removal of compact attention).

This document contains current constraints, a short record of completed work, and the remaining execution order. Earlier rankings, implementation diaries, and removed-path instructions are superseded. Their full record is available in this file at commit `91a4d956e` and its predecessors.

## Current status and constraints

- Improve prefill and single-request generation on the existing four GPUs, with the same model files, Q8 KV, usable context, and MTP depth.
- Compact attention was removed from source and launchers, both prefill and decode. Do not restore it as part of this plan. The model's own sparse selection remains; the removed optimization was a different way to execute selected attention.
- New optimizations must not introduce an accuracy tradeoff. Prefer removal of redundant copies, exact selection, and reuse of unchanged data. Reduced precision, changed reductions/routing, and tolerance-based quality claims do not meet this requirement.
- The existing Volta dispatch fixes remain the accepted starting point. They can change arithmetic ordering in principle; historical matching greedy responses do not prove universal numerical equivalence. Keeping them is an existing decision, not permission to accept further numerical changes. Removing them requires a separate decision.
- The upstream merge completed without textual conflicts. The merged source has not been built or benchmarked in this work. Historical rates and installed libraries must not be labeled as measurements of `79f26bba9`.
- This update changes documentation only. It does not authorize replacing deployed binaries, stopping a user-owned server, changing hardware settings, or committing/pushing subsequent work without approval.

## Reference configuration

The workload is Flash-Next with **layer splitting**, not the earlier dense 27B tensor-parallel workload. The launcher supplies `-ts` but no `-sm tensor`. Mixed-runtime AllReduce is not its primary execution model.

| Setting | Reference |
| --- | --- |
| Repository | `/home/despc/sources/fork/llama.cpp-v100` |
| Launcher | `/home/despc/llama.cpp/start_qwen_flash-4gpu-mtp.sh` |
| Default runtime | `/home/despc/llama.cpp/fork_v100` |
| Target | `/mnt/data/models/qwen3.8-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` |
| Draft | `/mnt/data/models/qwen3.8-flash-next/mtp/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` |
| Devices | RTX 5080, RTX 5070 Ti, V100 SXM2 32 GB, V100 PCIe 32 GB |
| Backend order | `CUDA0,CUDA1,V100_CUDA0,V100_CUDA1` |
| Layer split weights | `146,175,345,334` |
| Context | 160072 requested; historical runtime slot padded to 160256 |
| Batch / microbatch | 4048 / 512 |
| KV types | K and V both `q8_0` |
| MTP | `draft-mtp`, depth 1, draft on `V100_CUDA1` |
| Expert placement | All expert layers on GPU; no CPU MoE |
| CPU / concurrency | 8 threads, affinity `5555`, strict affinity; one request slot |
| Driver bridge | `/opt/nvidia-v100/lib/v100_redirect.so` |

Model facts from the actual GGUF: 48 layers, embedding width 2560, 512 experts with 10 selected per token, 24 attention heads, 2 KV heads, K/V head dimension 256. There are 12 QSA layers with compression ratio 4 and 36 recurrent layers. The indexer has 4 heads of dimension 128 and top-k 2048; expanded selection width can reach 2051 including the tail. Hyper-connections have width 4, making exported hidden rows 10240 floats. The large PLE table remains in host RAM by design.

The launcher enables grouped Volta MMQ with `GGML_CUDA_MMID_MMQ_PREFILL=1`. DP4A-only J_FIT defaults on in source; `GGML_CUDA_MMQ_MMID_J_FIT=0` disables it. The optional sort-workspace cap is not enabled by the reference launcher. Record effective inherited environment variables as well as script defaults.

## Hardware constraints relevant to the plan

These are findings from the 2026-09-06 hardware audit, not a new live audit:

| Device | Memory | Observed host route |
| --- | ---: | --- |
| RTX 5080 | 16 GB | CPU root port, x8; upstream capability/target reported Gen3 |
| RTX 5070 Ti | 16 GB | CPU root port, x8; upstream capability/target reported Gen3 |
| V100 SXM2 | 32 GB | CPU-attached Gen3 x4 |
| V100 PCIe, also draft device | 32 GB | PCH-attached Gen3 x4, sharing chipset resources |

- Host: i7-13700K, 24 logical CPUs, 80 GiB DDR4-3200, one NUMA node. Inference affinity selects one thread per P-core. Build with all available CPU threads; while building, do no other task work.
- Blackwells and V100s use separate driver namespaces. Foreign-runtime transfers already use pinned host staging; replacing pageable staging is not a new optimization.
- Blackwell-to-Blackwell P2P was reported unsupported by the audited topology. V100-to-V100 read/write capability was reported available, but actual route and bandwidth were not benchmarked. The SXM2 card had no active NVLink links.
- Idle link speeds/clocks are not sustained-load measurements. Driver, ACS/IOMMU, ECC, power, firmware, affinity, or physical-placement changes require a measured reason and separate approval.
- Check memory headroom at long context. Successful startup or a 5k request does not establish that another pipeline slot, cache, or larger microbatch fits.

## Completed work: short summary

All rates below are historical measurements from before the latest upstream merge.

| Work | Result | Current disposition |
| --- | --- | --- |
| Grouped quantized `MUL_MAT_ID` on Volta (`2628dc82d`) | Removed the per-expert host-synchronizing fallback; historical 5k prefill 478 -> 613 t/s | Kept |
| Correct Volta DP4A tile table (`efb8d872e`) | Fixed selection of a table for a different execution layout; 613 -> 811 t/s at 5k | Kept |
| Per-expert column tile, DP4A-only J_FIT (`de3dbad40`, `9e8b48b5f`) | 811 -> 907 t/s in the isolated 5k comparison; applying the rule to Blackwell MMA had erased much of the gain | Kept; attribution investigation closed |
| Profiling and flag parsing | Removed unconditional per-ubatch synchronization costing about 1.2% generation; `FLAG=0` now means disabled | Kept; profiling is opt-in |
| Owned-process harnesses | Replaced broad process killing and fixed-port PID adoption | Reuse these, not obsolete launch recipes |
| Sort-workspace cap (R11) | Correctness checked; cap alone did not speed prefill at fixed microbatch. It enabled ubatch 640 at long context | Optional memory tool, not a default speed optimization |
| Ubatch 640 with bounded sort scratch | About 7% more prefill in the later comparison, about 4% worse generation after 150k, plus 284-350 MiB/device | Rejected for the reference; keep 512 |
| Two pipeline copy slots | Reserve could succeed, but gain was about 0.5% and a 30k request OOMed | Off; revisit after memory/dependency work |
| Existing Blackwell sparse-256 path | Controlled repeat found neutral throughput; the earlier 2.2% generation penalty was a profiling artifact | Not a deployment candidate; regression theory retired |
| FP16 expert experiment | No deployment win established; padding/dispatch fixes addressed the demonstrated issue | Deferred; arithmetic changes are outside current policy |
| Compact attention, tiled and per-query | Tiled prefill gained about 23% at 100k and 34% at 150k; decode also accelerated, but numerical results and generated output changed | Removed in `91a4d956e`; not pending work |

The compact prototype exposed real bugs in allocation size, reference-mask shape, index stride, captured scratch lifetime, and non-finite verification. These were fixed before removal. The reusable lesson is to verify intermediate data and capture/replay behavior, not trust plausible output or speed alone.

Its historical quality evidence did not establish absence of regression: tiled PPL was 2.8943 versus 2.8929 on one documentation corpus; decode top-1 agreed at 21/24 positions, and the logprob comparison retained only 14/24 because of top-40 truncation. The 0.048% PPL difference is not an upper quality-loss bound, and the selected decode sample cannot bound all predictions. These results do not justify restoring the path.

### Historical dense-attention reference

These are the non-compact arm of a recorded comparison with retained dispatch fixes, not post-merge measurements. The isolated 907 t/s J_FIT result came from a different run.

| Prompt tokens | Prefill, t/s |
| ---: | ---: |
| 5000 | 865.9 |
| 30001 | 741.7 |
| 49999 | 640.3 |
| 100001 | 475.0 |
| 150001 | 377.1 |

The old uninstrumented dense decode profile reference was about 21.43 t/s after 150k. Short-prefix generation was roughly 60-65 t/s across historical runs. Neither rate has been re-established for the merged source.

### Bottleneck evidence still useful for prioritization

- At 100k prefill, Teslas accounted for 84.3% of summed GPU time. Their split was attention 41.3%, expert matmuls 25.9%, other matmuls 12.7%, TOP_K 7.4%, recurrence 3.0%.
- At 150k decode, summed GPU shares across both backends were attention 25.6%, TOP_K 17.7%, other matmuls 17.5%, GET_ROWS 12.2%, and expert matmuls 5.3%.
- TOP_K here is the QSA indexer selecting up to 2051 cache positions. Expert routing is a separate, much smaller ARGSORT operation.
- These are diagnostic shares, not wall-time ceilings or speedup promises. The per-operation profiler disables CUDA graphs and synchronizes operations; that decode run fell from 21.43 to 12.47 t/s. Do not infer a rigorous lower bound from these shares or add them to CPU time.
- Attention remains larger than TOP_K after compact removal. TOP_K is first among currently admissible compute candidates, not the largest operation overall.

## Current execution order

P0-P5 below replace all earlier rankings. Parenthesized R/G identifiers only identify the historical idea.

| Priority | Work | Expected value and gate |
| --- | --- | --- |
| P0 | Validate and freeze the merged-source baseline | Required before attributing new changes; bounded validation, not another instrumentation project |
| P1 | Exact TOP_K contract, tests, then selector (R12) | Best evidenced admissible long-context compute opportunity; preserve boundary-tie membership |
| P2 | Remove redundant MTP handoffs and waits (G2/R13/G5) | Small targeted changes first; proceed on measured exposed cost |
| P3 | Reduce peak-live temporary buffers and redundant copies (R4/R7/R9/R15/R16) | Memory/launch enabler; preserve arithmetic, ownership, and dependencies |
| P4 | Reuse unchanged pooled indexer keys (R10) | Conditional long-context opportunity; invalidation and persistent memory are substantial costs |
| P5 | Retry two-slot pipeline (R2) | Only after P3 demonstrates full-context headroom and a workable dependency schedule |

### P0. Freeze the new baseline

Upstream introduced changes relevant to this model:

- `5fdfa6282`: GDN Q/K normalization now uses `build_gdn_l2_norm`, with epsilon inside the root, instead of the previous clamped L2 norm. It changes the formula, not just speed. Source: `src/models/models.h` and `src/models/qwen4exp.cpp`.
- `73ab7599b`: branchless Q4_K/Q5_K unpack in MMVQ, relevant to multi-row verification. Its L2-prefetch addition is gated to DGX Spark, not our GPUs. Do not implement the unpack optimization again.
- `73a43d1f6` and `b74f590ea`: CUDA race/barrier fixes in MMID/MMF and flash attention.

Build a separate candidate runtime using all available CPU threads, with no other task work while building. Keep the installed runtime unchanged. Verify executable and loaded-library identities for both CUDA namespaces.

Run existing backend correctness coverage for changed paths on Volta and Blackwell. Establish uninstrumented prefill and sustained generation at representative short and long prefixes, including a full-context memory check. Reuse valid archived fixtures rather than rebuilding the historical experiment matrix.

Record differences against the pre-merge build separately from future optimization comparisons. A changed response can now come from the upstream GDN correction; blindly demanding old token output would conflate a model correctness fix with an optimization regression. Future candidates compare against the validated merged baseline.

Deliverable: an identified baseline build, effective configuration, raw timing/correctness results, and memory headroom. No promotion based only on an automatic textual merge.

### P1. Exact TOP_K before new attention work

Source: `ggml/src/ggml-cuda/top-k.cu`, `argsort.cu`, `src/models/qwen4exp.cpp:build_qsa_top_k`, and `tests/test-backend-ops.cpp`.

The CUDA 12.9-era CUB path sorts all cache positions and copies the first k. At long context k is at most 2051 while N reaches about 150k. Avoiding full sorting and full-sized index/key temporaries can reduce time and memory without changing attention arithmetic.

Implementation order:

1. Establish the exact consumer contract and actual build dispatch. QSA scatters indices into a mask, so membership matters; other TOP_K consumers may need output order. Do not weaken the generic contract to optimize QSA.
2. Resolve ties before choosing an algorithm. One-row radix sort, captured segmented radix sort, and uncaptured segmented sort take different branches. Equal scores are common because the indexer expands compressed blocks. Different tokens at the kth boundary change the model even if scores match.
3. Add regression cases to the existing backend test file: n_q=1/2 and prefill shapes, k=2051, short/long N, partial chunks, equal-score boundaries, forced tail/causal masks, signed zero, and unsupported/non-finite input behavior. Cover both architectures, direct execution, and capture/replay. Verify that the candidate branch actually runs.
4. Prototype a bounded exact selector, such as radix threshold selection followed by candidate extraction and required ordering. No approximate thresholds or reduced k. Preserve the reference tie contract, or fall back where it cannot be guaranteed. Establish how often fallback is needed before investing in a large kernel.
5. Compare indices and resulting QSA masks exactly. Keep scores and dense attention inputs unchanged. Then measure complete operation time including scratch/preparation, request latency, and peak memory.

Do not blindly switch to newer CUB `DeviceTopK`: the current source requests `not_guaranteed` determinism and `unsorted` output there. The HIP radix implementation uses atomic boundary-tie selection and is not an exact drop-in CUDA solution.

Start with decode shapes: the measured long-context cost is material and small row count bounds the prototype. Extend to prefill after the contract holds. A general sort rewrite or compressed-domain score rewrite is not required first.

Deliverable: exactness tests, an opt-in candidate with safe fallback, eligible/fallback counts, and an uninstrumented paired end-to-end result. An isolated kernel win with frequent fallback or worse request time is not a deployment win.

### P2. MTP handoffs and sampling

Source: `common/speculative.cpp`, `common/sampling.cpp`, and hidden-output handling in `src/llama-context.cpp`.

- Inspect the draft loop's `llama_get_embeddings_nextn_ith` call before the `n_max` termination check. At depth 1 its result may not be needed for another draft step. Establish whether moving/skipping it avoids an actual copy or wait; a getter call alone is not proof of expensive work.
- Attribute `pending_h`, `verify_h`, shifted hidden rows, and prefix catch-up transfers. An exported row is 10240 floats, not 2560. Determine producer/consumer devices before proposing a device-local handoff.
- If material, use an internal device-buffer handoff or bounded pinned staging with explicit completion and lifetime. Preserve the one-token shift, sequence ownership, accepted/rejected paths, pending row, rollback, and public CPU embedding consumers.
- Consider target backend sampling only for a supported chain with an exact contract. The launcher does not enable it. Greedy GPU argmax may avoid full-logit download, but probabilities, grammar, stochastic sampling, and MTP verification can require more than one token ID. Preserve sampler order, RNG behavior, and rejection semantics.

Keep depth 1 and full draft prefix processing. Optimize measured time per accepted output token, not target-only kernel rate. Stop a neutral small probe rather than expanding it into a sampler rewrite.

### P3. Memory and execution overhead without changed math

Identify the peak-live allocation and exposed waits after P1. Do not sum logical tensor sizes as if all were simultaneously live.

Candidates, ordered by the measured blocker:

- Avoid full-sized sort temporaries through P1 before introducing another workspace knob.
- Remove redundant copies/materialized broadcasts and reuse storage after its final consumer. HC pointwise fusion is eligible only if it preserves operation order and intermediate rounding; FMA contraction is not automatically equivalent.
- Generate existing mask/bias values from exact metadata where this removes a real buffer/upload. Preserve finite tail bias, causality, cell mapping, and sequence ownership. Host input preparation was only about 0.6% at 30k; this is primarily a memory opportunity.
- Consider tile-local Q8-to-F16 conversion only if it produces the same decoded values and retains attention arithmetic. Historical conversion was 0.7% of the profiled 100k attention operation; it is not a large standalone compute opportunity.
- Reuse stable CUDA subgraphs and remove proven redundant waits without changing tensor shapes or numerical dispatch. Retained addresses, in-flight readers, and allocation generations must remain valid. Defer shape bucketing that changes reduction geometry or consumes unbudgeted memory.
- Improve foreign-runtime transfers only when exposed time is material. Pinned staging already exists; source completion and safe slot reuse are remaining issues. Never pass stream/event handles between incompatible namespaces.

Deliverable: a specified reduction in peak-live bytes or exposed request time, exact affected data, and full-context stability. Buffer savings alone do not establish pipeline overlap.

### P4. Cache unchanged pooled indexer keys

The indexer repeatedly prepares old pooled keys. Reuse is admissible only if it avoids recomputing the same values, not if it substitutes different normalization/reduction arithmetic.

- Derive cached values from the same decoded quantized indexer keys, not pre-quantization activations.
- Define validity by layer, sequence, physical cells, positions, and cache generation. Invalidate for rollback, partial-block completion, cell reuse, sequence copy/remove/shift, and changed positions.
- Start with append-only single-sequence text and a general fallback. Keep the incomplete block current; verify that results do not depend on changed batch-shape dispatch.
- Budget memory first: the old estimate was about 18.3 MiB per QSA layer at 150k, about 220 MiB across twelve layers before metadata. This competes with pipeline headroom.

Advance only if refreshed GET_ROWS/pooling/norm/rotation attribution supports it. This does not remove query-to-block scoring or TOP_K.

### P5. Two-slot pipeline, conditional on P3

Keep ubatch 512. Prove reserve and late-prefix allocations fit with two slots. Establish independent in-flight inputs and graph ownership; `process_ubatch` synchronization before input reuse cannot simply be deleted.

Preserve per-layer recurrent/KV ordering, MTP catch-up state, and producer/consumer events. Use a non-serializing timeline to demonstrate different microbatches overlapping across stages. Extra slots without overlap are only extra memory.

Accept only a complete-request improvement with unchanged generation and supported context. Do not raise slot count, alter placement, or reduce context merely to make this experiment fit.

## Deferred or excluded work

- No compact/sparse attention implementation changing reduction order; no new attention tiling/partition tuning under a claim of harmless numerical noise.
- No chunked recurrence, FP16 expert replacement, or other new arithmetic reformulation under current policy.
- No ubatch 640 or deeper MTP as defaults; measured tradeoffs do not satisfy the generation requirement.
- No compressed-domain top-k shortcut until exact token-level membership, tail handling, and ties are proven. Selecting 2048/4 blocks is not automatically equivalent to selecting 2051 tokens.
- No blind toolkit upgrade, global force-MMQ/cuBLAS setting, or repeat of the now-upstream MMVQ unpack optimization.
- No shared-NCCL/cross-driver P2P project, tensor/expert parallel rewrite, CPU expert offload, or multi-request batching as a claimed single-request speedup.
- Placement, clock/CPU-policy tuning, expert work queues, and larger persistent kernels remain evidence-gated follow-ups, not next actions. Their benefit under current constraints is not established.

## Acceptance and safe execution

For each candidate:

1. Keep an immutable validated baseline and separate candidate runtime. Record source/patch identity, executable/library hashes, compiler/backend versions, effective environment, model paths, actual integer placement, and startup fallbacks.
2. Change one factor at a time. Require a structural argument for preserved values/selection/order, backed by exact tests where required. A few identical responses or small NMSE do not prove a numerically different algorithm cannot degrade quality. If equivalence cannot be justified, stop rather than silently relax the criterion.
3. Reuse existing operator tests. Cover sm_70 and sm_120, capture/replay with changed inputs, shape transitions, repeated requests, and relevant rollback/reset behavior.
4. Compare identical tokenized fixtures with fixed sampling/cache policy. Use interleaved A/B/B/A runs, at least five short samples, and appropriate long repeats. Cover 5k/30k/100k/150k prefill and sustained generation after short/50k/150k prefixes, leaving continuation room inside the context limit. Include mixed-length requests for allocator/graph lifetime checks.
5. Report raw prefill, TTFT, accepted-token generation rate, inter-token latency, MTP accepted/proposed counts, per-device memory peaks, and correctness. Disable diagnostic per-operation/phase profilers for throughput. They cannot prove overlap because they synchronize execution.
6. Do not accept repeatable generation regression or reduced context. Noise is inconclusive, not permission to spend a slowdown. Attribute changed output before interpreting an MTP throughput difference.
7. Use an owned-process harness and verified free port. A separate port does not create VRAM: arrange an exclusive GPU window when needed, without killing a user-owned server. Preserve deployed binaries until validation and explicit promotion.

No generic quality benchmark proves absence of regressions on all inputs. A future proposal to allow numerically different algorithms requires a separate policy decision and representative paired full-logprob/task-quality evaluation; it is not part of this exactness-preserving plan.

## Fixtures and historical evidence

- Harnesses: `/home/despc/llama.cpp/bench/harness.sh`, `longctx.sh`, `tg50.sh`, `census.sh`, and `greedy-diff.sh`. Check current behavior and process ownership before reuse.
- Long-prompt fixtures: `/home/despc/llama.cpp/bench/prompt30k.json`, `prompt100k.json`, and `prompt150k.json`.
- Old diary and temporary evidence paths: `git show 91a4d956e:docs/backend/CUDA-flash-next-prefill-plan.md`. Temporary artifact existence is not guaranteed; archive reused raw evidence in durable storage.
- Earlier dense-27B tensor-parallel work is separate: [CUDA mixed-runtime AllReduce](CUDA-mixed-runtime-allreduce.md). Do not apply its bottleneck model to this layer-split workload.

Next concrete work: P0 baseline validation, then P1 selection-contract tests before implementing faster TOP_K. No GPU experiments or runtime changes were performed for this cleanup.
