# CUDA mixed-runtime tensor parallelism

## Current native-type mode (2026-09-08)

Mixed AllReduce is off by default. Different CUDA runtimes use the meta-backend reduction unless `GGML_CUDA_ALLREDUCE=mixed` is explicitly set before startup. `GGML_CUDA_ALLREDUCE=none` disables the specialized collective without disabling tensor parallelism.

The opt-in mixed path transmits the tensor's original F32, F16, or BF16 representation. In particular, F32 is not converted to BF16 or INT8. Every rank accumulates in FP32 in the same global-rank order. There are no hierarchical pair sums, compressed streams, or requantized totals. Old compression flags have no effect on this path.

Two slots are protected by completion events across all runtime groups. ABI version 5 includes the wire layout, so both CUDA backend libraries must be rebuilt together. No build, runtime test, throughput result, or quality result has been obtained for this revision. Native transmission does not imply bit identity with a different floating-point reduction tree.

See [CUDA numerical optimization rollback](CUDA-numerics-rollback.md) for scope and deployment limits.

## Historical compressed implementation (removed)

The remaining sections record the removed implementation. Their flags, tuning values, quality claims, and measurements are not guidance for the native-type mode above.

This fork supports tensor-parallel AllReduce across CUDA devices that cannot be opened by one CUDA runtime. The original target system has two Blackwell GPUs on the normal NVIDIA driver and one V100 on a separately loaded driver:

| Global rank | Device | Runtime group | Tensor split |
| --- | --- | --- | --- |
| 0 | GeForce RTX 5080 16 GB | CUDA | 1 |
| 1 | GeForce RTX 5070 Ti 16 GB | CUDA | 1 |
| 2 | Tesla V100 SXM2 32 GB | V100_CUDA | 2 |

The tested model is `unsloth/Qwen3.8-27B-GGUF:Q8_K_L`. The server uses `--split-mode tensor`, `--tensor-split 1,1,2`, flash attention, a batch size of 4096, and an ubatch size of 2048.

## Why NCCL cannot cover all three ranks

NCCL requires all ranks in a process to use compatible CUDA driver and runtime state. The V100 is exposed through a separate redirected driver ABI, so one NCCL communicator cannot include both the Blackwell runtime and the V100 runtime.

The mixed implementation resolves a small group API from each backend DSO. Each runtime registers the same ordinary host allocation with `cudaHostRegisterPortable | cudaHostRegisterMapped`. Cross-runtime ordering uses cache-line-separated tokens in that shared mapped allocation.

`GGML_CUDA_ALLREDUCE=mixed` enables this path.

`GGML_CUDA_MIXED_AR_INT8=1` enables the optimized large-buffer wire used by
the Q8 launch script. Set it to `0` to restore the BF16 cross-runtime wire.

## Reduction paths

Small tensors use a flat mapped-host kernel. Every active rank publishes its contribution, waits for all other ranks, reads every peer contribution, and writes the sum to its local tensor.

Large tensors use a hierarchical path for the 2+1 topology:

1. Ranks 0 and 1 run the regular two-device local AllReduce.
2. Rank 0 publishes the Blackwell aggregate to mapped host memory.
3. Rank 2 publishes the V100 contribution in parallel.
4. Every rank waits for the peer runtime leader.
5. Every rank reads the peer aggregate and adds it to its local tensor.

The hierarchical path is selected at 1 MiB. The per-rank mapped staging capacity is 32 MiB. It supports the 20 MiB BF16 wire buffers produced by a 2048-token ubatch with an embedding width of 5120.

The local Blackwell pipeline currently stages large data through pinned host memory with D2H and H2D copies. It is not a direct P2P transfer. This is an important optimization target even though both devices share one CUDA runtime.

The optimized large-buffer path quantizes each local runtime aggregate to
symmetric INT8 with one FP32 scale per 4096 elements. The kernel rounds its
local aggregate through the same representation before adding the peer, so
all tensor-parallel ranks receive the same result. INT8 values are moved
between mapped host memory and shared memory as aligned 16-byte vectors. This
reduces physical cross-runtime tensor traffic by approximately one half
relative to BF16 without issuing thousands of scalar PCIe loads.

The small flat path remains unchanged. Decode therefore retains the BF16
transport and its original latency characteristics.

## Profiling controls

`GGML_CUDA_MIXED_AR_PROFILE=1` enables detailed GPU timing for large hierarchical calls. CUDA events measure the local reduction and full rank interval. The hierarchical kernel records `%globaltimer` timestamps for:

- mapped-host publication;
- exposed peer wait;
- mapped-host peer read and local add.

The process prints totals by rank and wire size during context destruction. This mode synchronizes every large reduction while collecting the records and changes scheduling. Use it to split the critical path, not as the final throughput result.

`GGML_CUDA_MIXED_AR_PROFILE=2` also logs every profiled call. High llama.cpp log verbosity may be needed for the per-call `INFO` lines.

`GGML_CUDA_MIXED_AR_CPU_PROFILE=1` measures existing `prepare` and `enqueue` calls with `steady_clock`. It does not introduce CUDA synchronization and has negligible measured overhead. It identifies CPU blocking caused by slot reuse.

`GGML_CUDA_MIXED_AR_FUSED_STREAM=1` switches the fused kernel to chunked
progress publication, `GGML_CUDA_MIXED_AR_STREAM_CHUNK` sets the qblocks per
progress step (default 4), and `GGML_CUDA_MIXED_AR_STREAM_DUPLEX=0` turns off
the opportunistic inbound drain that overlaps the two link directions. All
require `GGML_CUDA_MIXED_AR_FUSED_INT8=1`; see the streamed publication and
duplex overlap experiments below.

`GGML_CUDA_MIXED_AR_DEVICE_SLOTS=0` restores the original CPU event wait for
the two-slot ring. The default is `1`: kernels publish a per-rank departure
token after consuming a slot, and a future writer waits for that token on the
GPU before reusing the slot. This removed host-side blocking but did not by
itself improve end-to-end prefill speed because the wait represented real GPU
dependencies.

Example:

```bash
GGML_CUDA_ALLREDUCE=mixed \
GGML_CUDA_MIXED_AR_PROFILE=1 \
./llama-server \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_K_L \
    --device CUDA0,CUDA1,V100_CUDA0 \
    --split-mode tensor \
    --tensor-split 1,1,2 \
    -b 4096 -ub 2048 -fa on
```

Leave both profiling variables unset for normal inference.

## Qwen3.8-27B Q8 prefill profile

The reference prompt contains exactly 5000 tokenizer tokens. It is evaluated as three physical chunks: 2048, 2048, and 900 tokens.

Normal runs after instrumentation:

| Run | Prompt time | Prompt speed |
| --- | ---: | ---: |
| Profiling disabled | 7070.258 ms | 707.19 tokens/s |
| CPU markers enabled | 7079.689 ms | 706.25 tokens/s |
| Nsight Systems | 7082.291 ms | 705.99 tokens/s |
| Detailed GPU markers | 7392.019 ms | 676.41 tokens/s |

The GPU marker run recorded 384 large hierarchical calls:

- 256 calls with a 20,971,520-byte wire buffer;
- 128 calls with a 9,216,000-byte wire buffer;
- 6245 MiB of logical wire payload in total.

The measured critical AllReduce path was 4824.502 ms, or 65.27 percent of the profiled prompt:

| Phase | Total time | Share of AllReduce |
| --- | ---: | ---: |
| Local Blackwell reduction | 1979.513 ms | 41.03 percent |
| Publish aggregate | 1029.725 ms | 21.34 percent |
| Exposed peer wait | 812.529 ms | 16.84 percent |
| Read peer and add | 1001.588 ms | 20.76 percent |

The average critical time was 15.241 ms for a 20 MiB call and 7.210 ms for a 9,216,000-byte call.

Rank totals were:

| Rank | Full profiled interval |
| --- | ---: |
| 0, RTX 5080 | 4824.082 ms |
| 1, RTX 5070 Ti | 4704.987 ms |
| 2, V100 | 4215.631 ms |

Rank 0 was usually the critical rank. The V100 publication overlaps the local Blackwell reduction, so optimizing only the local reduction can expose more V100 wait instead of returning its full isolated gain.

The V100 mapped-host throughput derived from the profile was approximately 2.81 GiB/s for publication and 3.00 GiB/s for peer reads. Rank 0 reached approximately 5.95 GiB/s for publication and 6.11 GiB/s for peer reads.

## INT8 wire result

The final vectorized INT8 wire was compared with BF16 using the same Q8 model,
5000-token prompt, `1,1,2` split, 4096 batch, and 2048 ubatch. After the first
warm-up request, three consecutive runs were stable:

| Cross-runtime wire | Prompt time | Prompt speed |
| --- | ---: | ---: |
| BF16 | 6969.4 ms average | 717.42 tokens/s |
| INT8, scalar prototype | 6738.3 ms average | 741.88 tokens/s |
| INT8, vectorized | 4999.3 ms average | 1000.15 tokens/s |

The vectorized path improved warmed prefill throughput by 39.4 percent. Its
first request took 5107.878 ms, or 978.88 tokens/s.

Detailed profiling changes scheduling, but it exposes the phase improvement:

| Critical phase | BF16 | Vectorized INT8 |
| --- | ---: | ---: |
| Local Blackwell reduction | 1979.513 ms | 2168.069 ms |
| Publish aggregate | 1029.725 ms | 417.735 ms |
| Exposed peer wait | 812.529 ms | 299.843 ms |
| Read peer and add | 1001.588 ms | 539.752 ms |
| Full critical AllReduce | 4824.502 ms | 3426.508 ms |

The logical tensor volume is still 6245 MiB in the profiler because it reports
the BF16 tensor size. The INT8 payload itself is approximately half that size,
plus one FP32 scale for every 4096 values.

A deterministic 4969-token prompt produced the exact same 64 generated tokens
with BF16 and INT8. The selected first-token log probability changed from
-0.0154033 to -0.0146416. This is a smoke test, not a substitute for a model
quality or perplexity evaluation; use `GGML_CUDA_MIXED_AR_INT8=0` when exact
BF16-wire behavior is required.

An optional NCCL local reduction between the two Blackwell devices was also
tested. It reduced the isolated local phase but shifted time into peer waits
and regressed the full 5000-token prompt to 7253.207 ms in the profiled run.
The experiment was therefore removed from the production path.

## Nsight Systems findings

Nsight Systems can trace the normal Blackwell runtime, but it cannot see the redirected V100 runtime. The in-kernel records are therefore required for V100 timing.

Inside the exact prompt window, the two visible Blackwell devices were busy for about 6.40 seconds and idle for about 0.68 seconds. Major visible costs were:

| Work | RTX 5080 | RTX 5070 Ti |
| --- | ---: | ---: |
| Hierarchical mapped-host kernel | 3667.5 ms | 3053.0 ms |
| Local AllReduce H2D | 1093.2 ms | 1520.7 ms |
| Local AllReduce D2H | 945.4 ms | 1014.3 ms |
| Main Q8 matrix kernel | 313.6 ms | 360.5 ms |
| Gated delta net | 116.7 ms | 111.5 ms |
| Flash attention | 17.2 ms | 19.2 ms |

The visible CUDA runtime spent 4985.1 ms in 1552 `cudaEventSynchronize` calls inside the prompt window. Source-level CPU counters also showed about 1.02 seconds of additional waiting when the host reached the V100 runtime group. The wait time overlaps GPU work and must not be added to the GPU phase totals.

The original two-slot ring waited on the CPU before a slot was reused. This
prevented deep host-side submission and left approximately 0.68 seconds of
visible GPU bubbles. Moving reuse ordering to device tokens removed the CPU
blocking, but did not materially improve BF16 throughput because the same
dependency still had to complete on the GPU. Reaching 1500 tokens/s requires
a total prompt time near 3.33 seconds.

## Current development status (2026-09-02)

This section is a checkpoint for the work after the vectorized cross-runtime
INT8 path. It deliberately distinguishes measured code from build-only
experiments.

### Stable production baseline

The current production launch configuration remains:

- model: `unsloth/Qwen3.8-27B-GGUF:Q8_K_L`;
- devices: `CUDA0,CUDA1,V100_CUDA0`;
- split mode: tensor;
- tensor split: `1,1,2`;
- batch and ubatch: 4096 and 2048;
- context: 262144 tokens;
- cross-runtime wire: vectorized INT8, enabled with
  `GGML_CUDA_MIXED_AR_INT8=1`.

The latest exact 5000-token run of this baseline completed in 4993.778 ms, or
1001.25 tokens/s. The detailed profile, which adds synchronization and is not
a throughput benchmark, completed in 5379.015 ms, or 929.54 tokens/s. It
reported the following totals across 384 large reductions and 6245 MiB of
logical BF16 tensor payload:

| Phase | Critical-path total |
| --- | ---: |
| Local Blackwell reduction | 2167.716 ms |
| Publish aggregate | 418.020 ms |
| Exposed peer wait | 299.321 ms |
| Read peer and add | 539.690 ms |
| Other measured time | 1.038 ms |
| Full critical AllReduce | 3425.785 ms |

The rank totals were 3359.099 ms for rank 0, 3364.879 ms for rank 1, and
2206.952 ms for rank 2. CPU enqueue time was 26.079 ms for the Blackwell
runtime and 2.625 ms for the V100 runtime. The local Blackwell reduction is
therefore the largest isolated phase in this baseline.

The deployed DSOs under `/home/despc/llama.cpp/fork_v100` now contain the
stable cross-runtime INT8 path, the separately tested local INT8
implementation, and the fused experiment. The production launch script
enables only `GGML_CUDA_MIXED_AR_INT8`; both new experimental modes remain
disabled.

### Topology and current transport limit

`nvidia-smi topo -m` reports a PHB connection between the RTX 5080 and RTX
5070 Ti. `nvidia-smi topo -p2p r` reports `CNS` in both directions, and a CUDA
probe confirmed `cudaDeviceCanAccessPeer == 0` for both device pairs. Direct
CUDA P2P access between the two Blackwell cards is therefore unavailable on
this host. The measured link is PCIe Gen3 x8 under load.

An exact-window Nsight Systems trace of the stable cross-runtime INT8 path
completed the prompt in 5119.926 ms, or 976.58 tokens/s. In the 5.120-second
prompt interval, device 0 was active for 4.464 seconds and device 1 for 4.452
seconds. Both visible GPUs were simultaneously idle for approximately 0.562
seconds. The main visible costs were:

| Work | RTX 5080 | RTX 5070 Ti |
| --- | ---: | ---: |
| Mixed hierarchical INT8 kernel | 1711 ms | 1102 ms |
| Q8 matrix kernel | 315 ms | 361 ms |
| Gated delta net | 117 ms | 112 ms |
| Type-14 conversion work | 73 ms | 84 ms |
| H2D copies | 1097 ms, 6686.2 MiB | 1512 ms, 6682.2 MiB |
| D2H copies | 948 ms, 6480.4 MiB | 1026 ms, 6382.9 MiB |

The report and exported SQLite database are stored outside the repository at:

- `/home/despc/llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000-int8.nsys-rep`;
- `/home/despc/llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000-int8.sqlite`.

No thermal or power throttling was observed. During the run, the RTX 5080 and
RTX 5070 Ti stayed in P1 with core clocks near 2.9 and 3.06 GHz respectively.
The current bottleneck is transport and synchronization, not clocking.

### Experimental local INT8 reduction

`GGML_CUDA_AR_LOCAL_INT8_THRESHOLD=<bytes>` enables a new two-device local
path for large F32 tensors. A value of zero, which is the default, disables
it. The implementation:

- quantizes each local contribution in blocks of 4096 values using one FP32
  scale per block;
- exchanges aligned 16-byte INT8 vectors through mapped pinned host memory;
- uses a dedicated two-slot 32 MiB-per-slot host ring on each local device;
- rounds each local result through the same representation before adding the
  peer contribution.

The first implementation could deadlock after ring wraparound because one
rank re-recorded an event before the peer queued its wait on the previous
generation. The corrected implementation queues both N-2 dependencies before
either completion event is re-recorded.

With both local and cross-runtime INT8 enabled, warmed exact 5000-token runs
were 4968.420, 4952.042, and 4951.248 ms, corresponding to 1006.36, 1009.68,
and 1009.85 tokens/s. Compared with the 4993.778 ms stable baseline, the best
end-to-end gain was approximately 0.9 percent.

The detailed experimental profile completed in 5203.953 ms, or 960.81
tokens/s:

| Phase | Stable cross INT8 | Local plus cross INT8 |
| --- | ---: | ---: |
| Local Blackwell reduction | 2167.716 ms | 1152.708 ms |
| Publish aggregate | 418.020 ms | 503.842 ms |
| Exposed peer wait | 299.321 ms | 1002.615 ms |
| Read peer and add | 539.690 ms | 539.939 ms |
| Full critical AllReduce | 3425.785 ms | 3200.181 ms |

The local phase improved by approximately 1.015 seconds, but exposed peer wait
grew by approximately 0.703 seconds. The optimization reveals the V100
arrival and publication dependency, so most of its isolated gain does not
reach end-to-end prompt time. CPU enqueue time fell to 7.145 ms on the
Blackwell runtime and was 3.462 ms on the V100 runtime.

This path is intentionally default-off. Its result quality has not yet been
evaluated independently, and its extra 64 MiB mapped-host allocation per local
GPU is currently made when the local pipeline is initialized even when the
feature threshold is zero.

### Experimental fused local and cross-runtime INT8

`GGML_CUDA_MIXED_AR_FUSED_INT8=1`, together with
`GGML_CUDA_MIXED_AR_INT8=1`, selects a new fused 2+1 kernel for large F32
reductions. It is disabled by default. The kernel is intended to remove the
full-tensor boundary between local and cross-runtime reduction:

1. Both Blackwell ranks quantize and publish their original contributions to
   the local mapped-host ring.
2. Each Blackwell rank builds the same local aggregate and rounds it through
   INT8; only the leader publishes it to the cross-runtime buffer.
3. The V100 publishes its contribution directly to the cross-runtime buffer.
4. All ranks wait for the peer runtime and add the peer aggregate.

The code builds successfully for both CUDA backends with:

```bash
cmake --build build-v100 --target ggml-cuda ggml-v100-cuda -j8
```

The fused DSO was deployed to the separate runtime directory and passed a
short smoke test without a deadlock. A deterministic 962-token prompt with
eight generated tokens produced exactly the same text as cross-only INT8:
`<think>\nThe user is asking me to`. This is only a text smoke test and does not
prove numerical equivalence or model quality.

On a reproducible prompt whose round-trip tokenization is exactly 5000 tokens,
with speculative decoding disabled, the warmed normal runs were:

| Wire mode | Prompt runs | Prompt speed |
| --- | ---: | ---: |
| Cross-only INT8 | 4657.413, 4645.811 ms | 1073.56, 1076.24 tokens/s |
| Fused local plus cross INT8 | 4615.574, 4599.058, 4602.205 ms | 1083.29, 1087.18, 1086.44 tokens/s |

The same prompt was run with detailed GPU profiling enabled. Profiling
synchronizes each large reduction, so its absolute time is not comparable to
the normal benchmark, but the phase split is useful:

| Phase | Cross-only INT8 | Fused INT8 |
| --- | ---: | ---: |
| Local Blackwell reduction | 2169.840 ms | 0.143 ms |
| Publish aggregate | 417.906 ms | 1201.922 ms |
| Exposed peer wait | 310.892 ms | 1471.693 ms |
| Read peer and add | 539.457 ms | 539.399 ms |
| Full critical AllReduce | 3439.160 ms | 3214.279 ms |

The fused kernel removes the separate local reduction phase and reduces the
profiled critical path by 224.881 ms. It also moves local aggregate work into
the Blackwell publication interval, so the exposed wait becomes the dominant
phase. The profiled end-to-end prompt times were 4945.125 ms for cross-only and
4798.406 ms for fused. The normal end-to-end gain on this exact prompt was only
about one percent because the AllReduce critical path overlaps other graph
work.

A quick stripe-count sweep changed `GGML_CUDA_MIXED_AR_BLOCKS` for all
hierarchical kernels. The normal fused runs were approximately:

| Hierarchical blocks | 5000-token prompt |
| ---: | ---: |
| 16 | 4580-4626 ms |
| 32 | 4578-4618 ms |
| 64 | 4587-4618 ms |

The 32-block setting is kept. It reduced the profiled fused critical path to
3085.967 ms and read/add to 405.210 ms; 64 blocks did not improve the normal
end-to-end result and increased synchronization pressure. The setting also
sizes the mapped-host arrival, departure, and trace arrays, so it cannot be
changed independently for only one runtime group.

The fused implementation must still be treated as experimental. It is not
enabled by the production script, and it has not passed a broader output
quality or perplexity evaluation, a multi-seed rank-equivalence test, or a
decode regression test. Keep `GGML_CUDA_MIXED_AR_FUSED_INT8=0` for production
until those checks pass.

### Local arrival ring sizing fix

The two-device local pipeline sizes its arrival ring with
`GGML_CUDA_AR_KERNEL_BLOCKS`, which is 8. The fused kernel reuses that same
ring but launches `GGML_CUDA_MIXED_AR_BLOCKS` blocks, which is 32, so blocks
8 to 31 wrote their arrival token onto another (slot, rank) line. All ranks
publish the same token value in a call, so the aliasing did not show up as a
hang; it could let a block observe a peer arrival before that peer had
finished writing its stripe, which is a silent correctness hazard for every
fused measurement taken before this fix. The ring is now sized with
`GGML_CUDA_AR_ARRIVAL_BLOCKS`, the larger of the two launch widths.

The fix does not change the stable cross-runtime INT8 production path, which
drives the ring with 8 blocks. Its warmed exact 5000-token prompt after the
fix was 4613.174, 4614.447, and 4614.600 ms, or about 1083.5 tokens/s.

### Streamed publication experiment

`GGML_CUDA_MIXED_AR_FUSED_STREAM=1`, together with the fused mode, replaces
the fused kernel's stripe-level barrier with a chunked progress protocol.
Every block publishes a 64-bit word holding the current token in the high half
and the number of published qblocks in the low half, at offset 8 of the same
64-byte signal line the coarse token uses. A single aligned 8-byte store
crosses PCIe as one transaction, so a reader sees either the previous call's
word or a complete new one. `GGML_CUDA_MIXED_AR_STREAM_CHUNK` sets how many
qblocks one progress step covers; the default is 20.

The three phases stay separate. Interleaving publication and consumption per
qblock, which was the first implementation, breaks the pipelining of the
publication loop's PCIe writes and measured clearly slower.

Warmed exact 5000-token prompts, same host state, speculative decoding off:

| Mode | Prompt runs | Prompt speed |
| --- | ---: | ---: |
| Stable cross-only INT8 | 4613.2-4614.6 ms | 1083.5 tokens/s |
| Fused, coarse barrier | 4575.4-4581.1 ms | 1091.4-1092.8 tokens/s |
| Streamed, chunk 1 | 4733.7-4738.7 ms | 1055.1-1056.3 tokens/s |
| Streamed, chunk 8 | 4616.9-4618.7 ms | 1082.6-1083.0 tokens/s |
| Streamed, chunk 16 | 4581.2-4584.6 ms | 1090.6-1091.4 tokens/s |
| Streamed, chunk 20 | 4567.9-4571.7 ms | 1093.7-1094.6 tokens/s |
| Streamed, chunk 40 | 4571.9-4577.3 ms | 1092.4-1093.7 tokens/s |
| Streamed, chunk 80 | 4577.8-4578.0 ms | 1092.2 tokens/s |

Chunk 80 covers a whole stripe for the 20 MiB wire, so it is the control that
isolates the protocol's own overhead: it lands on the coarse kernel's result.
The best streamed setting is worth about 8 ms, or 0.2 percent, over the coarse
fused kernel.

The profiled run explains why the win is that small. Per prompt, with the
streamed accounting where publish covers all productive work and wait covers
both peer waits:

| Rank | Publish | Wait |
| --- | ---: | ---: |
| 0, RTX 5080 | 1265.9 ms | 1491.6 ms |
| 1, RTX 5070 Ti | 1147.1 ms | 1501.3 ms |
| 2, V100 | 1912.2 ms | 100.5 ms |

The profiled critical AllReduce path fell from 3085.967 ms for the coarse
fused kernel to 2759.097 ms, an 11 percent reduction, but end-to-end prompt
time barely moved. The V100 spends 1912 ms per prompt publishing and only
100 ms waiting: it is never blocked by the Blackwell side. The 1491 ms the
Blackwell ranks wait is the V100's mapped-host write time itself, measured
earlier at approximately 2.81 GiB/s for publication and 3.00 GiB/s for peer
reads. Finer publication granularity reschedules that transfer, it does not
shorten it, so the exposed wait is transport-bound rather than
synchronization-bound.

A deterministic 5000-token prompt with 16 generated tokens produced exactly
the same text on the stable cross-only path, the coarse fused kernel, and the
streamed kernel at chunk 20. This is a smoke test, not a quality evaluation.

Both experimental modes remain default-off. The production script still
enables only `GGML_CUDA_MIXED_AR_INT8`.

### Duplex overlap on the V100 link

`lspci` reports the V100 as `LnkCap: Speed 8GT/s, Width x16` but
`LnkSta: Speed 8GT/s, Width x4`. The card negotiates full Gen3 speed on four
lanes, which is a board limit: the host has no spare lanes, so the card cannot
be moved to a wider slot. Gen3 x4 gives 3.94 GB/s in theory, and the measured
2.81 GiB/s publication rate is 3.02 GB/s, about 77 percent of that. Only
10 to 15 percent of raw bandwidth is left, so switching the publication from
in-kernel stores to the copy engine cannot pay for itself.

PCIe is full duplex, and the kernel was using it as if it were not: every rank
published its whole contribution before reading anything, so the inbound and
outbound streams were strictly serialized and one direction always sat idle.
`GGML_CUDA_MIXED_AR_STREAM_DUPLEX=1`, the default under the streamed mode,
adds an opportunistic drain: after publishing a chunk, a block folds in
whatever peer chunks are already visible without blocking, and only the
remainder is drained with a blocking wait at the end. The drain never runs
ahead of the chunks whose own contribution is final, because publication still
writes `recvbuf` outright.

Warmed exact 5000-token prompts, speculative decoding off:

| Mode | Prompt runs | Prompt speed |
| --- | ---: | ---: |
| Stable cross-only INT8 | 4613.2-4614.6 ms | 1083.5 tokens/s |
| Fused, coarse barrier | 4575.4-4581.1 ms | 1091.4-1092.8 tokens/s |
| Streamed, chunk 20, no duplex | 4567.9-4571.7 ms | 1093.7-1094.6 tokens/s |
| Streamed, chunk 4, no duplex | 4681.1-4684.3 ms | 1067.4-1068.1 tokens/s |
| Streamed, chunk 20, duplex | 4418.3-4420.9 ms | 1131.0-1131.7 tokens/s |
| Streamed, chunk 8, duplex | 4389.2-4393.1 ms | 1138.2-1139.2 tokens/s |
| Streamed, chunk 5, duplex | 4262.0-4264.3 ms | 1172.5-1173.2 tokens/s |
| **Streamed, chunk 4, duplex** | **4242.4-4245.2 ms** | **1177.8-1178.6 tokens/s** |
| Streamed, chunk 3, duplex | 4260.9-4263.1 ms | 1172.9-1173.5 tokens/s |
| Streamed, chunk 2, duplex | 4284.8-4285.2 ms | 1166.8-1166.9 tokens/s |

Chunk 4 with the drain disabled is the control that isolates the effect: the
same chunk size costs 4681 ms without duplex and 4243 ms with it, so the whole
441 ms belongs to the overlap rather than to the chunk size. Against the
stable production path the streamed duplex kernel is 8.7 percent faster, and
against the coarse fused kernel 7.3 percent.

Small chunks only pay once the drain exists. Without it the optimum was 20
qblocks per step, because every step costs system fences and buys nothing;
with it the optimum moves to 4, because a smaller step puts inbound traffic on
the wire sooner.

The V100 rank's own kernel time fell from about 2006 ms to about 1680 ms per
prompt, and its exposed wait from 100 ms to under 2 ms. The Blackwell-side
profile is no longer comparable across modes: `GGML_CUDA_MIXED_AR_PROFILE=1`
synchronizes every large reduction, which removes exactly the overlap this
mode exploits, so its totals grow while normal throughput improves. Use the
normal benchmark for this mode and the profile only for the phase split.

Token generation is unaffected, as expected from decode using the small flat
path: 128 generated tokens ran at 32.79 and 32.82 tokens/s under the streamed
duplex kernel against 32.93 and 32.98 tokens/s on the production path.

A deterministic 5000-token prompt with 16 generated tokens produced exactly the
same text under the streamed duplex kernel as on the stable cross-only path,
the coarse fused kernel, and the streamed kernel without duplex. This is a
smoke test, not a quality evaluation, and the fused family stays default-off.

### Where the prefill time actually goes (2026-09-02)

An exact-window Nsight trace of the streamed duplex kernel at 64 blocks, taken
over one 5000-token prompt, changes the picture that guided the earlier work.
Within the 3964 ms prompt window, per Blackwell device:

| Work | RTX 5080 | Share |
| --- | ---: | ---: |
| Mixed AllReduce kernel | 3170.6 ms | 80.0 percent |
| Q8 matrix kernel | 390.6 ms | 9.9 percent |
| Gated delta net | 116.4 ms | 2.9 percent |
| Everything else | ~143 ms | 3.6 percent |
| True idle, no kernel resident | 143.1 ms | 3.6 percent |

The device is almost never idle, but four fifths of the time it is resident in
the AllReduce kernel. That bounds the classic optimization of overlapping
communication with computation: all compute on a Blackwell device is about
650 ms, so perfect pipelining could hide at most that much, and it would
require splitting the ubatch across two in-flight micro-batches in the graph
and the meta-backend scheduler.

The in-kernel phases show why the AllReduce kernel is so long, and it is not
transport. Per prompt:

| Rank | Own work | Waiting |
| --- | ---: | ---: |
| 0, RTX 5080 | 991 ms | 2222 ms |
| 1, RTX 5070 Ti | 821 ms | 2280 ms |
| 2, V100 | 1576 ms | 1.4 ms |

The Blackwell ranks wait 2.2 seconds, while the V100's entire kernel lasts
1.58 seconds and it never waits itself. The V100 therefore enters each
collective late: with `--tensor-split 1,1,2` it owns half the model and is the
slowest device, so every collective is a barrier behind its computation.

A diagnostic split sweep confirms it. At a context short enough to allow the
allocation, the exact same 5000-token prompt gives:

| Tensor split | Prompt time | Prompt speed |
| --- | ---: | ---: |
| 1,1,2 | 4126.2 ms | 1211.8 tokens/s |
| 1.3,1.3,1.4 | 3530.3 ms | 1416.3 tokens/s |
| 1.5,1.5,1.0 | 3175.4 ms | 1574.6 tokens/s |
| 1.7,1.7,0.6 | 2972.0 ms | 1682.3 tokens/s |
| 1.9,1.9,0.2 | 3030.6 ms | 1649.8 tokens/s |

This is a measurement, not a proposal: the deployment needs a 262144-token
context, which only fits when the V100 holds half the model. The rebalanced
splits do not allocate at that context, and with the MTP draft they do not
allocate even at 131072. The number is here because it quantifies the cost of
the constraint: roughly 1150 ms of the prompt is the V100 arriving late.

### Ubatch size

Larger ubatches produce fewer and larger collectives and better GEMM shapes on
Volta. The cross-runtime staging capacity was raised from 32 to 64 MiB per rank
so that a 4096-token BF16 wire fits. Measured with the streamed duplex kernel,
`1,1,2`, 262144 context, speculative decoding off:

| Ubatch | Prompt time | Prompt speed |
| --- | ---: | ---: |
| 1024 | 4382.3 ms | 1141.0 tokens/s |
| 2048 | 4138.8 ms | 1208.1 tokens/s |
| 4096 | 4024.4 ms | 1242.4 tokens/s |
| 6144 | 3830.8 ms | 1305.2 tokens/s |

With the MTP draft model loaded, however, neither 4096 nor 6144 allocates at
that context: device 0 fails a 2592 MiB compute buffer at 4096 and a 3888 MiB
one at 6144. The production configuration therefore stays at 2048 unless the
draft is dropped or the context is shortened.

### Standing configuration

Production shape, 262144 context, `1,1,2`, MTP draft on, three prompt and
generation cycles:

| Path | Prefill | Generation |
| --- | ---: | ---: |
| Stable cross-only INT8 | 4988-5086 ms, ~1000 tokens/s | 54.3-55.2 tokens/s |
| Streamed duplex fused INT8 | 4494-4601 ms, 1087-1113 tokens/s | 54.1-54.7 tokens/s |

### Generation profile (2026-09-02)

The flat small-tensor kernel now writes the same globaltimer records as the
striped ones, and profile collection runs for it too, so decode is finally
visible on both runtimes. `GGML_CUDA_MIXED_AR_HIER_THRESHOLD` sets the wire
size at which the striped path takes over; the default remains 1 MiB.

A 128-token generation with speculative decoding off runs at 34.9 tokens/s,
or 28.45 ms per token. An exact-window Nsight trace over the generation phase
gives, per Blackwell device:

| Work | Per token | Share |
| --- | ---: | ---: |
| Mixed AllReduce kernel | 17.4 ms | 60.9 percent |
| Q8 mat-vec over its own weights | 8.2 ms | 28.7 percent |
| Everything else | 2.3 ms | 8.2 percent |
| True idle | 1.8 ms | 6.2 percent |

The in-kernel records show what those 17.4 ms are. Over 16768 collectives:

| Rank | Total | Publish | Wait | Read and add |
| --- | ---: | ---: | ---: | ---: |
| 0, RTX 5080 | 2712.9 ms | 48.5 ms | 2518.8 ms | 60.3 ms |
| 1, RTX 5070 Ti | 2572.3 ms | 45.7 ms | 2385.4 ms | 57.8 ms |
| 2, V100 | 769.9 ms | 173.2 ms | 168.4 ms | 255.3 ms |

93 percent of the Blackwell AllReduce time is waiting. Per 10240-byte
collective that is 149 us of wait against 6 us of own work. Host-side enqueue
is not the problem either: 8.0 us per call for the Blackwell group and 4.4 us
for the V100 group.

The cause is the same as in prefill. Generation is memory-bound, and with
`1,1,2` the V100 streams half the model, 13.5 GiB, on every forward pass. Both
runtimes reach a similar effective bandwidth -- about 610 GB/s for a Blackwell
reading 6.75 GiB in 11 ms, about 640 GB/s for the V100 reading 13.5 GiB in
roughly 21 ms -- so the V100 simply has twice the work and every collective is
a barrier behind it. A forward pass therefore cannot go below roughly 17 to
21 ms while the split stands.

Measured attempts that did not pay:

- Deeper speculation. MTP at `--spec-draft-n-max` 2 gives 54.9 tokens/s at
  60.0 percent acceptance; 3 gives 53.1 at 47.1 percent; 4 gives 49.1 at
  38.0 percent. Two is the optimum.
- Routing small collectives through the striped hierarchical path with
  `GGML_CUDA_MIXED_AR_HIER_THRESHOLD=4096`. That path runs a full local
  AllReduce pipeline at 64 blocks and costs far more than it saves on a 10 KiB
  payload: 28.9 tokens/s against 35.1 on the flat path.

What remains, in order of expected value:

1. The V100's mat-vec kernels. At about 640 GB/s against roughly 900 GB/s of
   peak HBM2 bandwidth, every 10 percent recovered is about 2 ms per forward
   pass, or 7 percent of generation. This is the largest lever left and it
   trades nothing away.
2. A lightweight two-stage small path: reduce the Blackwell pair over their own
   link and let the leader publish one aggregate, so the V100 reads 10 KiB
   instead of 20 KiB per collective. Its own kernel currently costs 46 us per
   call against 6 us on a Blackwell. The existing striped path is the wrong
   tool for this; it would need a flat two-stage kernel.
3. Serving several requests at once. The barrier cost is per forward pass, not
   per token, so concurrent sequences amortize it almost linearly while VRAM
   allows.

### Volta mat-vec: the configuration knobs are exhausted

Generation is bounded by the V100 streaming its half of the weights, so the
first question is whether its mat-vec kernel leaves bandwidth on the table.
Measured alone with llama-bench, `--device V100_CUDA0 -sm none`, the card runs
the whole 26.11 GiB model at 24.72 +/- 0.06 tokens/s, which is 40.45 ms per
token and about 645 GB/s against roughly 900 GB/s of peak HBM2, or 72 percent.
For comparison, a Blackwell in the three-GPU run reads its 6.75 GiB share in
11 ms, about 610 GB/s of 960, or 64 percent, so the V100 is not the outlier.

The fork already carries a Volta-specific mmvq table taken from
dalzyu/llama.cpp-volta. Re-tuning it on the isolated harness changed nothing:

| Configuration | tg64 |
| --- | ---: |
| nwarps 2 for Q8_0, the tuned default | 24.72 tokens/s |
| nwarps 4 | 24.80 tokens/s |
| 2 rows per block | 24.50 tokens/s |

The differences are at or below the measurement's own spread, so the table
stands. `should_halve_iters` is gated to the GB10 table and never fires on
Volta; since doubling nwarps by hand did nothing, enabling it would not help
either. Beyond these knobs, this lever needs a new Volta mat-vec kernel rather
than tuning, with an uncertain payoff against the 72 percent already achieved.

### Two-stage small path

`GGML_CUDA_MIXED_AR_FLAT_2STAGE=1` selects a two-stage flat kernel for the 2+1
topology. The flat path has every rank read every peer's contribution, so the
V100 pulls two payloads per collective across its x4 link. In the two-stage
kernel the local pair sums itself first and only the leader republishes that
sum, into the upper half of its own staging region and announced by a second
word in the same signal line, so the single-rank runtime reads one payload
instead of two. No new buffers are needed.

Generation of 128 tokens, speculative decoding off:

| Path | Generation |
| --- | ---: |
| Flat, BF16 wire, the default | 35.07-35.10 tokens/s |
| Two-stage, BF16 wire | 35.92-36.05 tokens/s |
| Flat, F32 wire | 33.46-33.53 tokens/s |
| Two-stage, F32 wire | 34.73-34.81 tokens/s |

The in-kernel records confirm the mechanism. The V100's kernel total falls from
769.9 to 669.4 ms over 16768 collectives and its read-and-add from 255.3 to
188.6 ms; the Blackwell ranks fall from 2712.9 to 2656.9 ms.

In the production shape, 262144 context, `1,1,2`, MTP draft on, generation goes
from 54.4-55.2 to 57.2-57.9 tokens/s, about 5 percent, with prefill unchanged
at 982-1002 tokens/s.

There is a numerical caveat. The pair sum is rounded through the wire type
before it is published, and every rank uses that rounded value so the ranks
stay consistent. On the BF16 wire that is one extra BF16 rounding of an
intermediate sum, of the same kind the wire already applies to each
contribution but not bit-identical to the current path. On an F32 wire the
rounding disappears -- the cast is a no-op -- and the two-stage kernel is still
worth 1.3 tokens/s over the flat one, but the wider wire costs more than the
saving: 34.8 against 35.1 for today's default. The mode is therefore off by
default, and choosing it is a precision decision, not a performance one.

A deterministic 5000-token prompt with 16 generated tokens produced the same
text on the two-stage path as on the default one.

### Generation: what else was tried (2026-09-02)

With the two-stage small path in place, generation in the production shape runs
at 57.5-58.3 tokens/s. The remaining levers were measured and none of them
moved it:

| Attempt | Result |
| --- | --- |
| CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) | 57.4-58.2 tokens/s, no change. Launch overhead is not the constraint; the devices are 94 percent busy. |
| `--no-mmproj` to free VRAM for a larger ubatch | Still fails a 2592 MiB compute buffer at ubatch 4096. The projector was not what blocked it. |
| `--spec-draft-n-max` 3 and 4 | 53.1 and 49.1 tokens/s against 54.9 at 2. Acceptance falls from 60.0 to 47.1 and 38.0 percent. |
| `--spec-draft-p-min` 0.1 / 0.3 / 0.6 | 57.4-57.9 / 53.2-53.9 / 43.5-44.0 tokens/s. Acceptance rises to 61.9, 65.7, 81.1 percent, but fewer drafted tokens more than cancel it. Zero stays optimal. |
| Volta mmvq table re-tuning | Within measurement spread; see above. |

Speculation already halves the collectives per output token: with MTP the run
issues 61 collectives per output token against 131 without it, because one
forward pass verifies three tokens and about 1.6 are accepted. What it cannot
change is the pass itself, and the pass is what the V100 gates: waiting is
still 91 percent of the AllReduce time with MTP on.

The V100's own ceiling was measured directly with llama-bench on the card
alone, `-sm none`:

| Model | Size | Generation | Effective bandwidth |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_K_L | 26.11 GiB | 24.72 tokens/s | 645 GB/s |
| Qwen3.8-27B Q4_K_XL | 16.34 GiB | 34.75 tokens/s | 568 GB/s |

Q8_0 reaches a higher effective bandwidth than Q4_K, so the quantization's
unpacking is not the limit -- memory is. Against roughly 900 GB/s of peak HBM2
the kernel sits at 72 percent, and a well-behaved streaming kernel can reach
about 85 percent. That remaining 15 to 20 percent, worth 3 to 4 ms per forward
pass, is the only untapped generation lever left, and it requires a better
Volta mat-vec kernel rather than any configuration change.

### Spending Blackwell VRAM headroom on the split

The `1,1,2` split was treated as fixed because the 262144-token context only
fits when the V100 carries the large share. That is true of the context, but it
left the two 16 GB cards at 86 percent: 14048 and 14004 MiB of 16303, or about
2.2 GiB unused each. Since the V100 gates every collective, that headroom is
worth more as model share than as free memory.

Moving share is not free on the Blackwell side, because the KV cache is split by
the same ratio and at this context it is far larger than the weights. A 2.5
point shift adds roughly 670 MiB of weights and 535 MiB of KV per card, so
`1.1,1.1,1.8` alone fails the 1296 MiB compute buffer that ubatch 2048 needs.

The compute buffer is the thing to trade away: it scales with ubatch, roughly
1296 MiB at 2048 and 650 MiB at 1024. Measured at 262144 context with the MTP
draft loaded:

| Split | Ubatch | Generation | Prefill | Device 0 |
| --- | ---: | ---: | ---: | ---: |
| 1,1,2 | 2048 | 57.5 tokens/s | 1096 tokens/s | 86 percent |
| 1.1,1.1,1.8 | 1024 | 61.0 tokens/s | 1084 tokens/s | 92 percent |
| **1.2,1.2,1.6** | **1024** | **63.7 tokens/s** | **1130 tokens/s** | **95 percent** |
| 1.2,1.2,1.6 | 1536 | fails a 972 MiB buffer | | |
| 1.25,1.25,1.5 | 1024 | loads, dies on the first request | | |

Both phases improve, which is the tell that the V100 gated both: a smaller
ubatch on its own costs prefill about 5 percent, yet prefill still ends up
ahead. Three prompt-and-generation cycles at the chosen setting gave 1126-1155
tokens/s prefill and 62.4-64.0 tokens/s generation.

The edge is sharp and every neighbouring setting moves it, so the split has to
be re-checked after any change to context length, the draft model, or the
projector.

### Quality against speed, measured (2026-09-02)

Every approximation in the current configuration was measured against the same
corpus and the same reference prompt, so the cost of each one is on the record.

Perplexity was run with `llama-perplexity` over 306 KB of English prose from
this repository's own documentation, 120 chunks of 512 tokens, tensor split
`1.2,1.2,1.6`. The absolute value is not comparable to published wikitext
numbers, but every configuration sees the identical chunks in the identical
order, so the differences are meaningful well below the quoted standard error.

| Cross-runtime wire | Perplexity | Delta | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| BF16, everything experimental off | 4.0647 | reference | 734.1 tok/s | 61.82 tok/s |
| Vectorised INT8 | 4.0876 | +0.0229, +0.56% | 1037.0 tok/s | 61.83 tok/s |
| INT8, fused and streamed | 4.0948 | +0.0301, +0.74% | 1153.8 tok/s | 61.66 tok/s |
| The above plus the two-stage small path | 4.0948 | +0.0301, +0.74% | 1154.8 tok/s | 63.80 tok/s |

The INT8 wire buys 41 percent of prefill for 0.56 percent of perplexity. The
fused streamed kernel adds another 11 percent of prefill for 0.18 percent more.
Neither touches generation, because decode's collectives are far below the
1 MiB threshold where the striped INT8 path takes over -- they run on the flat
BF16 path throughout.

The two-stage small path therefore does not appear in the table above at all:
the perplexity run exercises only large collectives. It was measured separately
by shrinking the ubatch to 64 tokens, which puts every collective on the flat
path, over 60 chunks:

| Flat path | Perplexity |
| --- | ---: |
| Stock | 3.7176 +/- 0.06507 |
| Two-stage | 3.7172 +/- 0.06505 |

The two-stage kernel comes out 0.0004 lower, which is to say the extra rounding
of the pair sum is lost in floating-point reordering noise. It is worth 3.5
percent of generation, and 2.1 tokens/s in the table above, at no measurable
quality cost.

Read together: the whole optimisation stack cost 0.74 percent of perplexity,
all of it from the INT8 wire family on the prefill path, and returned 57 percent
of prefill and 3 percent of generation. The remaining generation gain came from
the tensor split and speculation, which do not approximate anything.

### Narrowing the INT8 scale block

That 0.74 percent turned out to be mostly avoidable. The INT8 wire carried one
FP32 scale per 4096 values, which is a very long run to put under a single
scale. The transfer tile has to stay at 4096, because a 256-thread block moves
it as one 16-byte-per-thread vector, but the scale block does not: the streamed
kernel now computes and applies `GGML_CUDA_MIXED_AR_SCALE_VALUES` per group
inside the tile. Each extra scale costs 4 bytes.

| Values per scale | Perplexity | Delta vs BF16 wire | Prefill | Generation |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 4.0948 | +0.74 percent | 1154.8 tok/s | 63.80 tok/s |
| 1024 | 4.0720 | +0.18 percent | 1165.3 tok/s | 63.35 tok/s |
| 512 | 4.0688 | +0.10 percent | 1163.6 tok/s | 63.73 tok/s |
| **256** | **4.0661** | **+0.03 percent** | **1157.6 tok/s** | **63.65 tok/s** |
| BF16 wire reference | 4.0647 | reference | 734.1 tok/s | 61.82 tok/s |

Speed is flat across the sweep -- the extra scales are 1.6 percent more wire
bytes at 256 values, which is nothing against what the INT8 wire saves in the
first place. 256 is now the default, so the cross-runtime approximation costs
0.03 percent of perplexity instead of 0.74 while keeping the 57 percent prefill
gain.

Only the streamed kernel implements the finer scale. The coarse fused kernel
and the plain hierarchical INT8 kernel still use one scale per tile, so turning
`GGML_CUDA_MIXED_AR_FUSED_STREAM` off also reverts to the coarser quantization.

### Tensor split experiments

The `1,1,2` tensor split remains required with the current 262144-token
context. Attempts to shift more model weight from the V100 to the two
Blackwell GPUs failed while reserving the draft context graph:

- `1.2,1.2,1.6`: device 0 failed a 1296.06 MiB allocation;
- `1.1,1.1,1.8`: device 0 failed the same 1296.06 MiB allocation.

These failures are configuration-level VRAM limits, not AllReduce failures.

## Optimization order

Two levers have been tried and measured, and both results narrow what is left.
Finer publication granularity on its own is worth 0.2 percent, because the
exposed wait is transfer time rather than a synchronization artifact.
Overlapping the two link directions is worth 7.3 percent over the coarse fused
kernel, because the V100's Gen3 x4 link was being driven as if it were half
duplex. Raw bandwidth is now within 10 to 15 percent of the link's ceiling and
the card cannot move to a wider slot, so the remaining levers have to move
bytes or calls.

The profile above reorders this list. The exposed wait is no longer transport:
it is the V100 arriving late at each collective because it computes half the
model. Transport work now has a much smaller ceiling than device throughput.

1. Speed up the V100's own computation, which is the critical term at roughly
   2.8 seconds per prompt. Its share cannot move, so this means the Volta
   matrix kernels themselves: check which MMQ path sm_70 selects for Q8_K_L and
   whether a better one exists. This is the only remaining lever that trades
   nothing away.
2. Use a larger ubatch wherever VRAM allows; 6144 is worth 8 percent over 2048.
   This currently conflicts with the MTP draft at a 262144 context.
3. Extend the duplex drain to the local Blackwell exchange. Implemented and
   measured neutral, default off; revisit only if the balance changes.
4. Overlap communication with computation by pipelining two micro-batches.
   Bounded by about 650 ms, the total compute on one Blackwell device, and it
   requires graph and scheduler changes.
5. Reduce the bytes crossing to and from the V100 with a narrower wire. Ruled
   out for now: it would change numerics, and the link is no longer the binding
   constraint.
6. Run a broader quality or perplexity evaluation for the fused, streamed, and
   local INT8 stages before treating them as lossless, and add a multi-seed
   rank-equivalence test and a decode regression test.
6. Re-run the same exact 5000-token prompt after every change. Also test token
   generation because the small flat path has different latency requirements.

The reference Nsight report is stored outside the source tree under `llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000.nsys-rep` on the measured host.

### Flat publication for the slow group (2026-09-03)

With a second V100 the topology becomes 2+2, and the hierarchical scheme costs
the Tesla side dearly. Reducing a pair locally means each rank publishes its
contribution and reads its partner's before anything crosses, so a Tesla moves
4N over its Gen3 x4 link where a single Tesla moved 2N. The profile showed it
exactly: on the 5000-token prompt the Tesla leader's productive time went from
1691.6 ms with one card to 3071.4 ms with two, and the Blackwell ranks' exposed
wait grew from 2029.1 to 2635.8 ms. The compute saving from halving each
Tesla's share was spent on transport twice over.

`GGML_CUDA_MIXED_AR_FLAT_GROUP=1` makes the group that does not hold global
rank 0 publish flat: every rank writes its own contribution into its own cross
slot and skips the local exchange entirely. Each rank then folds in two
cross-runtime payloads instead of one -- the aggregating group reads both
Teslas, and each Tesla reads the Blackwell aggregate plus its partner. The fast
side absorbs the extra read on its far wider links.

Striped path, 1024 calls, 6255 MiB of logical wire:

| Rank | Hierarchical | Flat group |
| --- | ---: | ---: |
| 0, RTX 5080, total | 3837.3 ms | 2758.4 ms |
| 0, exposed wait | 2635.8 ms | 1390.8 ms |
| 1, RTX 5070 Ti, total | 3708.6 ms | 2633.2 ms |
| 2, V100 SXM2, own work | 3071.4 ms | 2444.6 ms |
| 3, V100 PCIe, own work | 2115.5 ms | 2265.5 ms |

The critical AllReduce path falls 28 percent and prefill goes from 936-960 to
974-1000 tokens/s. Generation does not move: at 10 KiB its collectives are far
below the 1 MiB threshold and run on the flat small path, which still reduces
the pair. Converting that path the same way is the obvious next step and is
worth roughly one publish step per collective on the Tesla leader.

Two mistakes were made getting here and both hung the large collectives until
fixed: the progress word must be indexed by the publisher's own rank in flat
mode rather than by its group leader, and the mode must be selected from
`config->n_backends` rather than `group->backends.size()`, which is still empty
at that point in the init function.

### Flat publication on the small path, and the unequal Tesla pair

Generation runs entirely on the flat small path, so converting only the striped
kernel left it untouched. Giving the small path the same treatment -- each rank
of the flat group publishes its own contribution and every rank folds in two
cross payloads instead of one -- is worth 5 percent: generation goes from 70.2
to 72.4-74.2 tokens/s with prefill unchanged at 962-990.

The decode profile also showed that the two Teslas are not interchangeable.
Per 128 tokens on four GPUs, the V100 PCIe card did 286 ms of its own work and
waited only 241 ms, while the SXM2 card did 232 ms and waited 592 ms: the PCIe
card, on a 250 W limit against the SXM2's 300 W, is the one everybody waits
for. Shifting share away from it pays:

| Tensor split | Generation | Prefill |
| --- | ---: | ---: |
| 1.1,1.1,0.9,0.9 | 68.5 tok/s | 998 tok/s |
| 1.1,1.1,1.0,0.8 | 69.5 tok/s | 972 tok/s |
| **1.2,1.1,0.95,0.75** | **70.2 tok/s** | **989 tok/s** |
| 1.25,1.1,0.95,0.7 | 70.1 tok/s | 984 tok/s |
| 1.2,1.1,1.2,0.6 | 66.9 tok/s | 931 tok/s |

The 5080 also takes more than the 5070 Ti, simply because it had VRAM left at
the previous setting; 1.3 there fails a 756 MiB allocation.

Together with the striped-path change, four GPUs now run generation at
72.4-74.2 tokens/s against 66.7-66.9 on three, with prefill at 962-990 against
1136-1150. Three GPUs remain the better choice for prompt-heavy work.

### Handing the finished total back to the slow group

Flat publication left the Teslas moving 3N on their Gen3 x4 links: N out for
their own contribution, then N in for the Blackwell aggregate and another N in
for their partner's. The prefill profile put them squarely at the top --
2940 and 2716 ms of their own work against 2793 for the fastest Blackwell, and
they barely waited.

`GGML_CUDA_MIXED_AR_REPUBLISH_TOTAL=1` moves the last step to the fast side:
the aggregating group folds both flat contributions in, finishes the whole sum
and republishes it, so each Tesla reads one payload instead of two. Its link
then carries N out and N in, and the two directions overlap.

Doing this in whole tensors would be a loss -- it inserts a second round trip
into the dependency chain. It only pays because both halves are streamed per
chunk: the aggregating side consumes chunk k from both sources, rounds the
total through the wire and publishes it immediately, so the slow side starts
reading the total while it is still publishing its own contribution. Every rank
of the aggregating group performs the same rounding so the ranks stay
identical; only the leader writes.

Prefill goes from 962-990 to 1092-1123 tokens/s with generation unchanged at
72.1-73.4. The mode is restricted to the striped path: decode's collectives are
10 KiB and latency-bound, where the extra hop would cost more than the halved
inbound traffic saves.

Four GPUs now stand at 1092-1123 prefill and 72.1-73.4 generation against
1135-1150 and 66.2-66.9 on three, so the fourth card is finally worth its place
in both phases rather than trading one against the other.

### MTP for Qwen3.8-Flash-Next (2026-09-03)

The model ships its own multi-token-prediction head, trained jointly with it and
distributed as a separate GGUF. Unsloth's guide points at a dedicated llama.cpp
branch for it; that turned out to be unnecessary. The upstream merge of
2026-09-02 already brought the support into this fork -- `load_mtp`,
`borrow_shared_tensor` and the draft-only export path are all present, and
cherry-picking the branch's first commits came back empty or conflicted only
where our tree was already ahead.

What the model needs is placement and depth, not code:

- The draft costs about 1.9 GB and, without `--spec-draft-device`, lands on
  CUDA1 where it fails to allocate. It goes on the second Tesla, which has the
  headroom.
- Depth matters far more than the guide suggests. Every drafted token costs a
  full set of barriers across four ranks, so acceptance falls off fast:
  `n_max` 1 gives 78.9 percent acceptance and 57.3-65.3 tokens/s, 2 gives 52.4
  percent and 53.0-60.6, 3 gives 44.7 percent and 49.5, and the guide's 5 gives
  27.2 percent and 47.4 -- barely above no speculation at all.
- The draft's share of the CUDA pool during a large prefill is what caps the
  context: 49152 works, 65536 loads and then dies in the pool at top-k inside
  the MoE router.

At 49152 context, against the same configuration without the draft:

| | With MTP | Without |
| --- | ---: | ---: |
| Prefill | 566-600 tok/s | 607-642 tok/s |
| Generation | 58.1-64.9 tok/s | 44.8-48.1 tok/s |

Generation gains 35 percent for 6 percent of prefill. The script without the
draft remains the one to use when context matters: it reaches 98304 tokens.

### Long context for Qwen3.8-Flash-Next, and why -ncmoe is the wrong lever

The deployment needs at least 160072 tokens. The obvious route is `-ncmoe`,
pushing expert layers to the CPU to free VRAM, and it is the wrong one: every
offloaded layer then runs its MoE on the CPU, which costs far more than the
memory it frees.

What actually grows with context is the compute buffer, not the KV cache. The
sparse attention indexer sorts 2048 candidates against the whole cache, so per
device it is about 1.5 GiB at 49152 tokens, 2.4 at 160072 and 3.9 at 262144.
The KV cache is almost incidental by comparison: only 12 of the 48 layers are
full attention, the rest are recurrent, so at 160072 it is 1995 MiB for the
main cache plus 748 MiB for the indexer against 8873 MiB of compute buffers.
The 26.8 GiB per-layer-embedding table never enters VRAM at all.

So the lever is the ubatch, with every expert kept on the GPU. Measured at
160072 tokens on UD-Q4_K_XL with the MTP draft:

| n_cpu_moe | Ubatch | Prefill | Generation |
| ---: | ---: | ---: | ---: |
| 10 | 1024 | 354 tok/s | 42.7 tok/s |
| 2 | 512 | 386 tok/s | 52.0 tok/s |
| 0 | 512 | 469 tok/s | 62.1 tok/s |
| **0** | **640** | **497-505 tok/s** | **64.7-64.8 tok/s** |
| 0 | 704 | loads, dies on a 5000-token prompt | |

Keeping the experts resident and paying with a smaller ubatch is worth 42
percent of prefill and 52 percent of generation over the offload route.

For the full 262144 context there is a second quant. UD-IQ4_XS puts 60.4 GiB
in VRAM against 76.9 for UD-Q4_K_XL, and 16.5 GiB is almost exactly what the
extra 100k tokens cost. It runs the architecture's maximum context at 523
tokens/s prefill and 64.1 generation -- 4 percent of prefill against the
Q4_K_XL script at 160072, and nothing measurable in generation. The same MTP
draft head serves both: it carries no embeddings and borrows them from whatever
target it is loaded against.

### A long prompt is not a big prompt: the pool allocation

The Qwen3.8-Flash-Next MTP configuration was tuned against a 5000-token prompt
and passed everything, then failed in production once the context filled:

    prompt processing, n_tokens = 12144, progress = 0.66
    CUDA error: out of memory ... in function alloc

The sparse attention indexer's top-k takes its temporaries from the CUDA pool,
and they are sized by the current cache length rather than by the ubatch. A
configuration can therefore start, reserve all its declared buffers, run a
short prompt at full speed, and still run out somewhere past ten thousand
tokens. Nothing in the startup allocation predicts it.

At ubatch 640 the run dies around 12-30k tokens; at 512 it completes a
100000-token prompt at 309 tokens/s, and a 30000-token one at 411. The cost of
the smaller ubatch on the short benchmark is 497 -> 478 tokens/s of prefill.

The practical rule for this architecture: any change to the memory balance --
split, context, draft, quant -- has to be re-checked with a prompt long enough
to fill a real fraction of the context. `bench/prompt30k.json` and
`bench/prompt100k.json` exist for that.

The [2026-09-06 Flash-Next prefill plan](CUDA-flash-next-prefill-plan.md) records the opt-in bounded sort workspace experiment, its successful 150k prompt run, the deployment stop point, and further optimization options for the four-GPU layer-split configuration. The earlier 27B tensor-parallel AllReduce findings are a different workload.

## Bit-identity with the reference, 2026-09-08

The question this path kept raising was whether it costs quality. It no longer has
an answer that needs measuring.

The kernel used to sum ranks sequentially. The meta backend's AllReduce folds any
ranks past the largest power of two into the first block, then combines at halving
XOR offsets -- for four ranks `(a0+a2) + (a1+a3)` -- with every step a ggml ADD in
the tensor's own type. Two different trees over the same FP32 values round
differently, so the two paths agreed closely and never exactly. The sequential sum
was also the marginally less accurate of the two: `n-1` roundings against the
tree's `log2(n)`.

The kernel now reproduces that tree, including the per-step rounding back to the
tensor's type -- a no-op for F32, and what the reference does for F16 and BF16.
XOR pairing is symmetric, so every rank still reduces the same tree and all of
them finish with the same bits.

Measured on the 27B tensor-parallel deployment across four GPUs, greedy,
`GGML_CUDA_ALLREDUCE=none` against `=mixed`:

| | Prefill, t/s | Generation, t/s | Output |
| --- | ---: | ---: | --- |
| `none` (reference) | 176.9 | 42.75 | `dc4fb163e3e27c45` |
| `mixed` | 232.6 | 66.11 | `dc4fb163e3e27c45` |

2000 generated tokens, 8407 characters, identical. A 200-token run after a 6k
prefill likewise matched (`756a2b7872419b29`). Mixed is 55% faster on generation
and 31% on prefill while producing the same bits.

The structural argument is that a difference cannot arise: the tree, the operand
order and the per-step rounding all match, and under reduce-scatter each element
is reduced once by one rank.

**The hash does not prove that.** An identical continuation means the greedy pick
matched at every position, and argmax absorbs logit differences that never reach
the top of the distribution. It is strong evidence for the argument and not a
substitute for it. Bit-identity of the collective's own output has not been
measured, and the review that found the signal-word collision below is a reminder
that a structural argument can be right about the arithmetic and wrong about
everything around it. A direct test of the collective -- comparing the reduced
tensor on every rank against the reference tree across sizes, modes, tails,
inactive contributions and slot reuse -- is owed and not yet written.

### What the runs do not cover

Four ranks and F32 only. Bit-identity for F16 and BF16 follows from the same code
-- the per-step rounding is to the tensor's type -- but no run exercised it. The
non-power-of-two fold follows the reference by construction and there is no
three-card configuration here to run it on.

### What remains removed

The INT8 packing, the F32-to-BF16 wire, and the fused, streamed, flat-group and
two-stage variants are gone from source, not disabled, along with the environment
variables that selected them. `GGML_CUDA_ALLREDUCE` is the only remaining choice:
`mixed` for the kernel above, `none` for the meta backend's own reduction. With
CUDA and V100_CUDA in separate registries an unset variable behaves as `none`.

## Restoring prefill without the compression, 2026-09-08

The numerical rollback removed eight things at once. They were not one kind of
thing: some traded exactness for speed, and some only changed how bytes move.
This is the second group put back, and what happened when it was.

### The profile decided what to work on

A 10k prefill of the 27B, per-operation profiler on:

| op | GPU ms, summed over four devices | share |
| --- | ---: | ---: |
| MUL_MAT | 6180.8 | 67.3% |
| GATED_DELTA_NET | 1143.9 | 12.5% |
| FLASH_ATTN_EXT | 622.8 | 6.8% |
| RMS_NORM | 395.0 | 4.3% |
| everything else | ~840 | 9% |

Per device: CUDA0 1625, CUDA1 1829, V100_CUDA0 3013, V100_CUDA1 2712 ms.

Tensor parallelism runs the cards concurrently, so model compute costs at least
the busiest device -- **3.0 s** -- against **35.6 s** of wall.

That bounds compute from below and does not decompose the remaining 32 s: the
maximum of per-device operation time is not the sum of the critical path's
stages, and what is left holds publication, waiting, the reduction itself, the
gather, launch overhead and host-side slot waits together. It was enough to
retire kernel-level work -- there is no large win hiding in the matmuls -- and it
is not enough to say the collective is 92%. Splitting those stages is what the
next prototype should start from.

With no P2P between these cards every byte crosses PCIe twice through mapped
host memory, so the collective's cost is bytes, and the question became which
bytes can be removed without touching the arithmetic.

### What was restored

**Streaming publication with a duplex drain.** The kernel wrote its whole
contribution, signalled, and only then read anyone else's, so nothing moved
inbound until the slowest publisher finished and one direction of a full-duplex
link sat idle. Each block now publishes in steps, advertises progress, and folds
in whatever peers have already advertised. Prefill 241.5 -> 281 tokens/s.

The chunk is vectors per thread per step. Swept: 1 -> 208.3, 2 -> 258.8,
4 -> 275.5, 8 -> 281.1, 16 -> 272.8. Signalling too often spends more on system
fences than the earlier drain recovers. Default 8.

**Reduce-scatter with all-gather.** Every rank used to reduce the whole tensor
and therefore pull every peer's contribution: N out, 3N in. Each now reduces one
shard and the finished shards are exchanged: N + N/4 out, 3N/4 + 3N/4 in. That is
2.75N against 4N, and 1.5N of reads against 3N in the direction that dominates.
Prefill 281.6 -> **371.1 tokens/s**, against 187.5 for the meta backend.

Both are transport. Steps partition elements and shards partition elements;
neither ever splits the operands of a sum. Every output element is still reduced
exactly once, by one rank, with the same butterfly over the same values, so all
of it stays bit-identical -- confirmed over 2000 generated tokens, same hash for
the meta backend, the single-shot kernel, streaming, and reduce-scatter.

### What was tried and rejected, with numbers

**Republishing to a leader, twice.** One rank reduces and broadcasts the total so
the others read one payload instead of three. On the single-shot kernel: 241.6 ->
65.8 tokens/s. Rebuilt on top of streaming, where the removed code said it
belonged: 281.9 -> 72.1. Both correct, both catastrophic. Concentrating the
reduction makes three ranks wait for one, and that is slower than four ranks
reducing redundantly in parallel. The removed code's own comment predicted the
first failure -- "only worth it because both sides stream it per chunk" -- and was
read before the attempt without being applied. Reduce-scatter is the same traffic
saving with the work distributed instead of concentrated, which is why it works.

**Staggering the gather order** so ranks do not all read peer 0's region first:
273.8 against 281.9. No win, reverted.

### The threshold, and why it is not arbitrary

`GGML_CUDA_MIXED_AR_RS_MIN_BYTES`, default 256 KiB; below it the single-shot
kernel runs. The collective's tensor is `n_embd * n_tokens * 4`, and this model
has n_embd 5120 over 65 layers:

| | tokens | bytes |
| --- | ---: | ---: |
| decode | 1 | 20 KB |
| MTP verification | 2-4 | 40-80 KB |
| prefill at ubatch 1024 | 1024 | 21 MB |

The two regimes are 260x apart and the threshold sits in the empty space between
them, so its exact value changes nothing. It was chosen by argument -- two extra
grid-wide barriers need a tensor large enough to repay them -- and then measured:
with the threshold at zero, so reduce-scatter also runs at decode, generation
falls from 66.14 to 50.84 tokens/s at unchanged prefill. The argument was right
and is now a measurement.

### The bug that looked like a result

The first reduce-scatter was faster and wrong. Its barrier waited only on each
peer's block of the same index, while the phases stripe by global thread, so a
block could read bytes that another block had not published yet. The server ran,
the text was coherent, the speed was up -- and the output hash differed from the
reference. Nothing else would have caught it.

The barrier is now grid-wide across every rank and block; eight blocks are
resident together, so it cannot deadlock. This is the case the bit-identity
discipline exists for: under "no difference was detected" it would have shipped.

### P2P: available on one pair, and slower than the host

The collective moves every byte through mapped host memory. Whether a direct
device-to-device route would be better is a property of the chipset, so it was
measured rather than assumed, with a probe running inside each library's own
namespace -- device ids repeat across the two stacks, so a probe in the wrong one
silently reports the wrong pair.

| route | GB/s |
| --- | ---: |
| Blackwell to Blackwell, peer | unavailable (`canAccessPeer` no, `nvidia-smi` reports CNS) |
| Blackwell to host / host to Blackwell | 7.15 / 5.96 |
| **V100 to V100, peer** | **1.38** |
| V100 to host / host to V100 | 3.29 / 3.32 |

The Tesla pair can do peer access and it runs at less than half the speed of the
path already in use. Both pairs sit behind a host bridge -- `nvidia-smi topo -m`
reports PHB, not a switch and not NVLink -- and ACS is on (`ReqRedir+`
`CmpltRedir+`), so peer traffic is redirected up to the root complex and back
down. That route is evidently worse than letting the root complex DMA to and from
DRAM.

So P2P is not a way out of the host path here. It would have to be re-checked if
the cards moved to slots under a common switch, or if ACS were changed -- neither
is a software decision, and neither should be made on the strength of an
unmeasured expectation.

The probe stays behind `GGML_CUDA_P2P_PROBE`, off by default and one-shot: it
enables peer access, and `group_init` runs more than once, which crashed the
server the first time it ran.

### Where this leaves the collective

The critical rank is the SXM2 Tesla, and it is not waiting: 291 ms of `wait_pub`
and 394 ms of `wait_red` out of 20.8 s. The other 20.1 s is its own work --
publish 8.1, reduce 4.1, gather 8.0 -- so there is no idle time on the critical
path to overlap into, which is why a streamed publish-to-reduce overlap was
started and then abandoned: it targets a wait that does not exist.

Its traffic per collective is `2N + 3wN` for a share w: 2.75N at an even split,
2.45N at 35/35/15/15, and 2.00N if it owned nothing. Publication is the floor --
every element is reduced by one owner who needs every rank's contribution, so all
of N must be published -- and driving w to zero was measured worse overall (345
against 391 tokens/s) because the Blackwells become the bottleneck first.

What remains would have to change the bytes or the topology: compression, which
this deployment excludes on accuracy grounds and which is where the historical
1000+ tokens/s came from; P2P, measured above and slower; or fewer collectives,
which is a property of the model graph. Within the current constraints this is
close to the floor.

### Where a prefill's time goes, as it stands

A 10k prefill of the 27B at 391.6 tokens/s, 25534 ms of wall, shares
35/35/15/15. The table is written from the V100 SXM2, which is the critical
rank: it waits for almost nothing and everyone else waits for it.

| stage on the critical rank | ms | % of wall |
| --- | ---: | ---: |
| publish -- writing its own contribution to host | 8056 | 31.5% |
| gather -- reading the shards it does not own | 7952 | 31.1% |
| reduce -- summing the shard it owns | 4090 | 16.0% |
| waiting for peers | 685 | 2.7% |
| **collective, total** | **20762** | **81.3%** |
| model compute, launches, host gaps | 4772 | 18.7% |
| **wall** | **25534** | **100%** |

Model compute is about 3013 ms of that last row, from the operation profile;
that reading was taken on an earlier configuration, so it is indicative rather
than exact. Within it: MUL_MAT 67.3%, GATED_DELTA_NET 12.5%, FLASH_ATTN_EXT
6.8%, RMS_NORM 4.3%.

Three things follow directly, and they are why the remaining ideas are the ones
they are.

**Four fifths of a prefill is the exchange**, all of it against the Teslas' PCIe
Gen3 x4 at 3.3 GB/s against the Blackwells' 7.15.

**Waiting is 2.7%.** The critical rank has no idle time to hide work in, which is
what retired the streamed publish-to-reduce overlap after it had been started:
overlap converts waiting into work, and there is no waiting here.

**Publish and gather together are 63%, and both are pure transfer.**

This section first said publication could not be reduced at all, because every
element is reduced by one owner who needs every rank's contribution. That is true
of the *other* ranks' contributions and was wrong about a rank's own: a peer
reducing shard p reads this rank's slot at region p, never at region `rank`, and
the owner takes its own values from `sendbuf`. Nobody ever read the region a rank
published for its own shard, and it was then overwritten with the total. It is no
longer published -- see below.

Kernel work is 12% of the wall, so there is nothing left to win there. What
remains is bytes on the link, and the three routes to fewer of them are
compression (excluded on accuracy grounds), P2P (measured above at less than half
the host path) and more lanes or a newer link generation. On the last: the
Blackwells' devices report `LnkCap 32GT/s x16` while their ports report
`8GT/s x8`, and both x4 ports report `8GT/s`, so the board or its firmware is
what limits this, not the cards. Swapping the cards between slots is predicted to
change nothing -- the cost model that reproduces all three measured share
configurations gives 14.89 ms either way, because the floor is `N / slowest link`
and two cards sit on x4 whichever pair they are -- but Teslas on the x8 ports
together with Gen4 on the x4 ports would remove that floor and is worth 1.84x on
the collective by the same model. Whether those ports can run Gen4 is a firmware
question, not a software one.

### A rank no longer publishes its own shard, 2026-09-08

Phase 1 used to write the whole contribution to host memory. The region covering
a rank's own shard had no reader: peers reducing shard p read that rank's slot at
region p, the owner reads its own values from `sendbuf`, and phase 2 overwrites
the region with the total anyway. It was `w*N` crossing the link for nobody.

Skipping it costs nothing in arithmetic -- no value moves and no sum changes -- and
the tail follows the same rule, published by everyone except the rank that owns
it.

Prefill went 391.3 to 411.6 tokens/s at unchanged shares, and the curve was
re-swept because the balance moves when a phase changes:

| shares | t/s | | shares | t/s |
| --- | ---: | --- | --- | ---: |
| 20/20/30/30 | 392.1 | | 30/30/20/20 | 408.6 |
| 25/25/25/25 | 400.3 | | 35/35/15/15 | 411.6 |
| 40/40/10/10 | 387.2 | | **35/35/17/13** | **413.7** |

The optimum stayed at the same place and the whole curve rose. Splitting the two
Teslas unevenly is worth a further 0.6%, small but reproducible to a tenth of a
token per second across interleaved runs -- the deployment takes 35/35/17/13.

Every setting returns the same output hash as the meta backend, which is what
establishes that the skipped region really had no reader: had anything consumed
it, the sums would have been short a contribution and the hash would have moved.

Total against where the 27B started this session: **241 -> 414 tokens/s** of
prefill, all of it bit-identical to the reference reduction.

### Still open

### The link is full duplex, and the kernel uses one direction at a time

Measured, because the argument for skipping it was wrong. "Waiting is 2.7%, so
there is nothing to overlap" answers whether there is idle time to hide work in.
It does not answer whether sending and receiving at the same time beats doing
them in sequence, and that is what a pipelined reduce-scatter would rest on.

`GGML_CUDA_DUPLEX_PROBE`, every card in a registry driven at once, at the
collective's own sizes:

| | send | receive | both at once | sequential would be |
| --- | ---: | ---: | ---: | ---: |
| V100 pair | 6.59 GB/s | 5.64 | **11.22** | 6.08 |
| Blackwell pair | 14.30 | 8.41 | **25.79** | 10.59 |

The Tesla link delivers **1.85x** when both directions run together, close to the
sum of the two one-way figures, and the Blackwell link 2.44x. Sizes of 21 and 64
MiB agree to within noise.

The kernel runs its phases strictly in order, and each is one-directional:
publish sends, reduce receives, gather receives. Half the link is idle
throughout. On the critical rank publish is 8.1 s and reduce 4.1 s; overlapped,
that pair costs about what the longer of them costs rather than their sum, which
is roughly 4 s of a 20.8 s collective.

What the probe does not cover: the two libraries probe separately, so contention
between the Blackwell pair and the Tesla pair over the shared root complex and
host DRAM is not in these numbers. A pipeline would have all four cards sending
and receiving at once, so the real figure will be lower than 1.85x.

**Superseded.** It is lower, and by more than "contention" -- the probe also let
the faster pair finish and go quiet while the slower one measured the tail of its
run on an emptied link. With both pairs guaranteed to be working throughout, the
Tesla figure is 1.63x for the copy engines and 1.34x for kernel-issued access at
64 blocks, not 1.85x. And the projection in the paragraph above -- that
overlapping publish with reduce saves about 4 s of a 20.8 s collective -- is
wrong twice over: the phases were re-measured after the publication and share
changes, and overlapping them was built and lost. See *Duplex, built and measured
to a conclusion* at the end of this document.

### Still open

And the shares will want re-measuring again after any further change to the
transport, for the same reason they moved here.

### The pipeline, built and not paying -- and why

Built behind `GGML_CUDA_MIXED_AR_PIPE_CHUNKS`, off by default. Passes the checks
we have: every chunk count returns the same hash as the phase kernel and the meta
backend, which is a matching continuation, not a tensor comparison.
Slower: 402.7 tokens/s at 8 chunks, 397.4 at 16, 396.3 at 32, against 413.7 for
the phase kernel it was meant to beat.

Two things were learned on the way, one of them a correction to this document.

**Vectorised access is not optional.** The first version read peers element by
element inside the chunk loop instead of a whole vector per peer, and ran at 142
tokens/s -- a third of the phase kernel. Four times the transactions over a Gen3
x4 link costs about what it sounds like.

**The fence placement is load-bearing.** `advertise` ends with a system fence, so
announcing a chunk immediately after publishing it drains the store queue before
any load is issued, and the two directions never coexist. Moving the fence after
the reads was worth nothing measurable, which is the clue to the real problem.

**The link will carry both directions, under conditions that are not ours.** The
first probe measured the copy engines and could have been dismissed as the wrong
mechanism, since the collective uses kernel-issued loads and stores rather than
DMA. Measured rather than assumed: kernel-issued access gave 10.00 GB/s with both
directions against 6.07 in sequence, 1.65x, nearly the copy engines' 1.85x.

That number is a property of the transport probe, not a forecast for the
collective, and two conditions separate them. The probe drove one pair of cards
while the other pair was idle -- and the two runtimes probe from separate
libraries, so neither ever saw the contention on the shared root complex and host
DRAM that this document has recorded as a limit since the beginning. And it ran
the directions as two kernels on two streams, which a fused collective cannot do.
The collective also has dependencies, a reduction, readiness signals, and unequal
volumes in the two directions, none of which the probe carries.

**Why the pipeline does not reach it is a hypothesis, not a diagnosis.** The
appealing story is that a block publishes and then reads, in that order, so the
posted stores have mostly landed by the time the reads issue. But blocks of the
same grid are not in lockstep, so serial order within one block does not by
itself mean the GPU has no overlap. There are at least two other candidates, and
the fence experiment separates none of them: the kernel hands each block a
contiguous stripe of the whole tensor and each owner a contiguous shard, so with
a Tesla share of 13-17% only a few blocks do its reduction while the rest carry a
different load and still wait; and the per-chunk readiness traffic costs
something on its own.

**Next step is a probe, not a rewrite.** Test the proposed cure apart from the
disease: one kernel whose blocks have different roles, some storing and some
loading independent buffers, swept over several splits and several grid sizes,
against the two-kernel form -- and repeated with all four cards loaded together at
the collective's real volumes and its real direction ratio, which is nearer 1.7
reads per write than 1:1. If the advantage survives both, move roles into the
reduce-scatter; if it does not, the variant closes on a measurement.

**It survived the probe and lost in the collective**, and it did so for a reason
the probe could not show: reaching the overlap needs the collective cut into
parts, and each part costs two cross-card synchronisation rounds on ranks that
are 1.7x apart. The fused role grid was never needed -- two streams reach the
same overlap -- so the role-split kernel was not built. See the last section.

No ratio is proposed here. Three-to-five would come from phase times measured
before the publication and share changes, and throughput is not linear in the
block count anyway -- a few blocks may already saturate a direction, or may not
issue enough requests to fill it. A correction to the comments in the code, too:
the reducing blocks do not only read, they also publish the finished sum, so
"publishers carry the outgoing traffic and the rest the incoming" is an
approximation, not the split.

**The risk that construction carries** is a hang from block scheduling: if waiting
blocks occupy the machine before the producers are resident, nothing progresses.
"There are only a few blocks" is not an argument. It needs a co-residency
guarantee that can be checked on both architectures, or a construction that does
not depend on one -- and two separate kernels spinning on each other are no safer
by default.

The current pipeline is left in place, off, as scaffolding rather than as a
candidate. "Correct" above should be read as "passed the checks we have":
matching output is not a tensor comparison, and tails, empty shares, and signal
reuse are exactly where it would not be.

### The role-split probe: the cure, measured apart from the disease

Two questions, asked before building anything: can one grid whose blocks have
different jobs reach the duplex that two kernels on two streams reach, and does
any of it survive all four cards being on the link at once at the volumes each
card actually moves?

Getting them asked honestly took three corrections to the probe, and each one
moved the answer.

The rendezvous could not work where it first sat. `group_init` is entered once
per registry from a single thread, so a meeting point inside it waits for a
participant that has not been called yet: the Tesla side had not reached the
probe while the Blackwell side was already waiting for it, and each went on
measuring a link the other was idle on. It runs from `comm_init_mixed` now, after
every runtime is up, on a thread each.

Starting together is not the same as working together. Equal work finishes at
unequal times, so the faster pair went quiet while the slower one measured the
tail of its run on a link that had emptied. The pairs take turns now: one is
timed while the other keeps both directions busy until told to stop, then they
swap. This was not a detail. Symmetric traffic, two kernels, 64 blocks, the
Teslas: 1.65x with the pair alone, 1.44x when the pairs merely start together,
**1.34x** when the other pair is guaranteed to be working throughout. Contention
costs about a fifth of the headroom, not the eighth this document said.

And the direction ratio is per card, not per machine. A rank publishes N and
reads (1+2w)N, so 35/35/17/13 is 1.70, 1.70, 1.34, 1.26 -- four workloads. Each
device is driven at its own now; the single 1.7 was the Blackwells' question
asked of everybody.

Teslas, 21 MiB, each card at its own ratio, both pairs loaded throughout:

| blocks | two kernels | best role split | sequential | best/seq |
|--------|-------------|-----------------|------------|----------|
| 8      | 6.48        | 6.36 (2:6)      | 5.90       | 1.08x    |
| 16     | 6.88        | 7.03 (4:12)     | 5.90       | 1.19x    |
| 32     | 7.11        | 7.83 (8:24)     | 5.91       | 1.32x    |
| 64     | 8.10        | **8.72 (16:48)**| 5.91       | **1.48x**|

Blackwells peak earlier, at 32 blocks and 8:24, with 19.14 against 12.95
sequential, also 1.48x. 64 is where the sweep stops, not a maximum anyone has
established, and the two architectures do not want the same number.

**A claim this document made and the sustained load withdraws.** It said the role
split earned the contention back, citing 10.65 against the 10.00 the two-kernel
form reached with the pair alone. Under a link the other pair never leaves, the
same configuration gives 8.99. It does not earn it back. What survives is
narrower and still useful: at equal grid size the role split beats two kernels --
8.99 against 8.13 at 64 blocks, 8.34 against 7.46 at 32 -- and the gap is real at
every size above eight.

**Grid size and split both matter, and the grid size matters more.** Going from 8
to 64 blocks is worth 19% to the two-kernel form (6.81 to 8.13) and 36% to the
role split (6.63 to 8.99). So most of what the role split gains over the
collective's present eight blocks is not overlap at all -- it is having enough
requests in flight. That has to be separated on the collective itself before any
of it is attributed to duplex: the phased reduce-scatter with a larger grid and
no role pipeline is the control, and until it is run, the share of this that
belongs to overlap is unknown.

Read-heavy splits win at every size and both ratios -- 2:6, 4:12, 8:24, 16:48 --
which fits the mechanism, stores being posted and cheap to issue while loads need
many outstanding requests to cover latency. It is a good starting choice, not an
established rule: the collective's readers also reduce, with a register and
shared-memory load this probe's readers do not carry.

Two things this still does not measure. The roles never synchronise here, while
the collective's would on every chunk, so these are a ceiling for the
construction rather than a forecast. And the copy engines under the same
sustained load reach 10.74 against 6.07, 1.77x -- a reference point for a
different transport, worth keeping as the next option if a role-split collective
turns out to lose its margin in the protocol, not a bound on what kernel-issued
access can do.

### The control: a larger grid, no role pipeline

The transport probe said most of what a role split gains over the collective's
eight blocks is having more requests in flight, not overlapping the directions --
8 to 64 blocks was worth 19% to two plain kernels and 36% to the role split. So
before crediting any of it to duplex, the same question had to be put to the
collective: does the phased reduce-scatter go faster with a larger grid?

The grid is negotiated now rather than fixed (`GGML_CUDA_MIXED_AR_BLOCKS`,
default 8, the value every earlier figure was measured at). The constant it is
capped by is the signal stride, which both runtimes must agree on whatever grid
they run. And the old comment that eight blocks are resident so the grid-wide
wait cannot deadlock is replaced by asking: `group_init` measures
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` for every kernel that can be
launched, at each type it can be instantiated at, on each device, and refuses the
group if the negotiated grid does not fit. A grid nobody can hold is a hang, and
"it is only a few blocks" is not a check.

| AllReduce | prefill tokens/s | output |
|---|---:|---|
| none (meta backend) | 177.4 | a96a6cf7... |
| phased RS, 8 blocks | **390.8** | a96a6cf7... |
| phased RS, 16 blocks | 387.2 | |
| phased RS, 32 blocks | 343.0 | |
| phased RS, 64 blocks | 270.3 | a96a6cf7... |

The answer is no, and it is not close. The gain does not merely fail to transfer,
it reverses: 64 blocks costs 31% against 8. The output is identical at 8, at 64
and against the meta backend, so this is the algorithm's cost, not a breakage,
and the occupancy gate did not fire on any of the four devices, so it is not
residency either.

The reason is in the barrier. The phased kernels wait grid-wide across every
rank, so each block polls `n_ranks * blocks` signal words and every block does
it: the words read per barrier per rank go as the square of the grid -- 256 at
eight blocks, 16384 at sixty-four -- and every one of them is a read over the
link the collective is otherwise trying to use.

Two things follow. The 19% and 36% the probe measured cannot be claimed for the
collective as it stands; whatever a role split is worth here, it is not that. And
a role-split reduce-scatter is only worth building if it drops the grid-wide
barrier, because the grid size it needs is exactly the grid size that barrier
cannot afford. A consumer must wait on the specific producers of the chunk it is
about to read, not on everyone.

### Duplex, built and measured to a conclusion

The question was whether the link's ability to carry both directions at once can
be turned into prefill. It can be built, it is bit-exact, and it loses. The
reason is not the one this document kept guessing at, and every guess that turned
out wrong was retired by a measurement rather than an argument.

**The barrier was the cause of the large-grid collapse.** The control the review
asked for -- the same phases, the same volumes, the same arithmetic, with the
waiting moved out of the working grid into a one-block kernel between launches --
makes the grid size stop mattering:

| grid | one kernel, grid-wide barrier | phases as launches, gate between |
|---|---:|---:|
| 8 | 390.3 | 392.2 |
| 16 | 387.2 | 391.9 |
| 32 | 343.0 | 391.0 |
| 64 | 270.3 | 390.0 |

So the suspicion was right, and it is now a measurement: 64 blocks cost 31% with
the grid-wide barrier and nothing without it. But the second half of that table
is the more useful half -- **a larger grid buys the collective nothing either.**

**Because every phase already runs the link flat out in one direction.** With the
tensor shape logged and the phase times measured, publication moves 3.41 GB/s a
card against the probe's 3.29 store ceiling, and the gather 2.89 against 2.83.
There is no unused bandwidth in a phase to give more blocks. The only thing left
unused is the other direction.

**So the directions were overlapped.** The tensor is cut into parts and part
q+1's publication runs on a second stream beside part q's reduction and gather;
readiness is a count rather than a flag, so three signal words carry any number
of parts, and the publishing stream announces without waiting, since publication
depends on no peer. Verified elementwise against the flat kernel: 965,345,280
elements per rank, zero differences, on all four cards, at every setting.

It is slower, and taking it apart says exactly why:

| | one stream, no overlap | two streams, overlap | what overlap did |
|---|---:|---:|---:|
| no parts | 392.5 | -- | -- |
| 2 parts | 366.0 | 354.4 | -3.2% |
| 4 parts | 343.8 | 355.7 | +3.5% |

Overlap works. At four parts it is worth +3.5%. But overlap requires parts, and
parts cost 6.8% at two and 12.4% at four -- measured with the identical part
structure on a single stream, where nothing can overlap. The thing that makes the
gain possible costs several times the gain.

**What the parts actually cost.** Not launch overhead: at four parts the extra
gates are about 2300 more launches over a prefill, some 23 ms against 3.6 s. It
is the straggle. Each part adds two cross-card synchronisation rounds, and these
ranks are far apart -- the Blackwells finish their share about 1.7x faster than
the Teslas, and they already spend 8-12% of a collective waiting. Every extra
round makes the fast ranks stop and wait again, at roughly 320 microseconds of
rank skew per round. Two rounds per part, four parts, 384 collectives in a
prefill.

**Two hypotheses this killed on the way.** That kernel-issued access might not
get the duplex the copy engines do -- measured, it does, 1.65x against 1.85x.
And that the collective's pattern of reading pages another card is writing might
be what destroys the overlap -- measured with a probe variant where each device
loads exactly what its neighbour is storing, and it costs nothing at all: 6.55
against 6.57 GB/s at eight blocks, 8.01 against 8.07 at sixty-four.

**The conclusion.** The duplex headroom is real -- 1.11x at the collective's grid
of eight, 1.37x at sixty-four -- and it is smaller than the price of the
partitioning needed to reach it on ranks this uneven. Nothing in the collective is
left to make faster by moving bytes differently: each phase saturates its
direction, the reduction is bit-exact and minimal, and the only remaining
resource costs more to unlock than it yields. The next gain, if there is one, is
not in the transport. It is in making the ranks less uneven, which is a placement
and share question, or in not sending the bytes at all.

**And then removed.** The code that produced all of the above is gone from the
tree: the pipelined reduce-scatter, the phase-split kernels and their gate, the
two-stream overlap and its control, the wall-clock stamps, and the duplex probe
with its cross-runtime rendezvous. None of it was ever on by default, and none of
it will be turned on, so keeping four implementations of one collective would buy
nothing but the chance of maintaining the wrong one. The numbers are the
deliverable; the scaffolding is not. Recovering any of it is `git log` away --
the last commit that holds it all is the one this section was written in.

What stayed, because neither belongs to this question: elementwise verification
against the flat kernel (`GGML_CUDA_MIXED_AR_VERIFY=1`), which is how any future
change to the collective gets checked, and the occupancy gate in `group_init`,
which is what makes the reduce-scatter's grid-wide barrier safe rather than
assumed. `GGML_CUDA_MIXED_AR_RS_BLOCKS=n` stayed with it, since the gate needs
something to check. The deployed configuration is unchanged and unchanged in
speed: the single-kernel reduce-scatter at eight blocks, 35/35/17/13, 391
tokens/s, verified elementwise after the removal.

## Where this stands, 2026-09-08

Prefill on the 27B went from 187.5 tokens/s on the meta backend to 390 on the
deployed collective, every step of it bit-identical to the reference reduction,
and the last of that came from arithmetic-free changes to how bytes move. The
transport is now finished in a specific sense: there is nothing left in it that
a measurement says is worth taking.

**Closed, with numbers.**

| question | answer |
|---|---|
| Does the link carry both directions at once? | Yes. 1.34x kernel-issued at 64 blocks, 1.63x for the copy engines, with all four cards working throughout. |
| Does a role-split grid reach that overlap? | Yes, and it beats two kernels at equal grid size. |
| Does a larger grid help the collective? | No. It is neutral without the grid-wide barrier and costs 31% with it. |
| Was the grid-wide barrier the cause of that? | Yes. The phase-split control makes grid size stop mattering. |
| Is there unused bandwidth inside a phase? | No. Publication runs at 3.41 GB/s a card against a 3.29 ceiling, the gather 2.89 against 2.83. |
| Does overlapping the directions pay? | No. Overlap is worth +3.5% at four parts; the partitioning it requires costs 12.4%. |
| Is the pipelined reduce-scatter worth keeping? | No. Correct, 402.7 against 413.7 at the time; left off as scaffolding. |
| Do the collective's results match the reference? | Yes, elementwise: 965,345,280 elements a rank, zero differences, four cards, every variant. |

**Still open, and worth something.**

The ranks are uneven, and that is now the whole story. The Blackwells finish
their share about 1.7x faster than the Teslas and spend 8-12% of every collective
waiting; every synchronisation round costs roughly 320 microseconds of that skew.
Shares (35/35/17/13) were tuned against the old phase structure and want
re-measuring against the current one. Beyond that the remaining levers are not in
the transport: which cards hold which layers, and whether some of these bytes
need to cross at all.

Two older items are still unfinished and unrelated to any of this: a `MUL_MAT_ID`
output comparison to substantiate the Volta dispatch equivalence, and the
Flash-Next work, which is a layer-split workload with its own plan and its own
bottleneck model.

**Answered the next day**, and it was the larger of the two levers: not sending
the bytes rather than sending them faster. Per-layer participation took prefill
from 369.5 to 627.1 tokens/s and generation from 63.3 to 78.5 -- see *Per-layer
participation* below, which supersedes the sentence above about where the levers
are. The shares did want re-measuring and still do: all four numbers are still used,
but they no longer describe one split across four cards -- 35:35 applies between
the Blackwells on the layers they share and 17:13 between the Teslas on theirs.

**Not worth revisiting without new hardware.** Direct device-to-device transfer
(1.38 GB/s against 3.29 through the host on the Tesla pair, unavailable on the
Blackwells), swapping cards between slots (the cost model came out exactly
neutral), republishing to a leader (catastrophic, twice), and overlapping the
directions (this document's last three sections).

Duplex in particular is **closed, not paused**. The link has the headroom, a
role-split grid reaches it, and the collective cannot use it: reaching the
overlap requires cutting the collective into parts, and on ranks 1.7x apart each
part costs more in synchronisation than the overlap returns. That arithmetic does
not improve with a cleverer kernel -- three were written -- it improves only if
the ranks stop being uneven. Do not start here again without that having changed.

## Per-layer participation, 2026-09-09

The collective was finished as transport: every phase saturated its direction and
the only unused resource cost more to reach than it returned. What was left was
not moving the same bytes faster but moving fewer of them, and that meant asking
who has to take part in each layer at all.

Four cards of two speeds do not want the same shape everywhere. The Blackwell
pair reduces a layer between themselves far faster than four cards can; the
Teslas are worth more as owners of whole layers -- holding weights and KV the
Blackwells have no room for -- than as participants in every reduction. The
split state applied one `--tensor-split` to every tensor, so neither was
expressible.

It came in three parts, and only the third and second together paid.

**Placement.** `llama_meta_device_get_split_state` already knows the layer;
making the per-device share a function of it is a few lines. `LLAMA_META_TP`
names the devices that share an unowned layer, `LLAMA_META_OWN` the layers a
device or a set owns outright. A device with no share gets a zero-length slice,
and the meta backend already clears that node's compute flag and zeroes its
contribution, so nothing about what an operation means had to change.

**On its own it was worth nothing, and the measurement says why.** With 40 layers
on the Blackwell pair and 25 owned by the Teslas, the collective still ran 384
times -- exactly as many as before -- and cost the same 3.0 s on the critical
rank. Every layer was still reduced over four ranks, including the 25 where one
card held everything and there was nothing to reduce. Prefill 367.9 tokens/s
against 369.6, generation 51.9 against 63.3: pipeline serialisation added, no
traffic removed.

**The collective's active mask was the load-bearing half.** A rank whose slice is
empty must not publish, must own no shard, and must not be waited for; it still
needs the result, so it still gathers. The shard boundaries fall to the ranks
that remain and the scalar tail follows the highest active rank rather than the
highest rank. Both runtimes derive the mask from the same per-rank flags, so both
reach the same one. That is what made placement pay: the same 40/25 split went
from 367.9 to 581.3 tokens/s.

**Skipping the transfer, not the summand.** A first version compacted the active
ranks into the front of the array and reduced over the shorter list, on the
argument that an exact zero folds out of a sum. The argument is wrong as stated:
compacting reorders the butterfly. With ranks 0, 1 and 3 active, `a0 + (a1 + a3)`
becomes `(a0 + a3) + a1`, and at 16777216, -16777216, 1 those give 1 and 0. The
sets this repository actually runs -- {0,1}, {2,3}, and single owners -- happen to
produce the same tree either way, but the interface allows the ones that do not,
and the sign of zero is a second question the argument never addressed. So the
positions and the tree are left exactly as they were and only the fetch is
skipped: the inactive rank's zeros are written locally instead of read across the
link. Bit-identity then holds by construction rather than by an argument about
zeros, and the traffic saving -- which was the point -- is unchanged.

**Then placement was tuned, and the constraint turned out to be KV.** Moving the
boundary gave 545.7, 562.2 and 580.3 tokens/s at 32, 36 and 40 Blackwell layers,
and ran out of memory at 44. The Blackwells hold 16.3 GiB and the model costs
about 0.45 GiB a layer -- but a layer with KV costs another 1 GiB at this context,
and only every fourth layer has KV. Giving the Teslas the KV-bearing layers
rather than a contiguous block let the Blackwells hold 49 layers instead of 40:
595.5. Keeping four of the KV layers on the Blackwells, which have room for about
that many, gave 628.8.

**Decode wanted the mask in the flat kernel.** Below the reduce-scatter threshold
the collective runs the flat kernel, which had no mask at all, so decode kept
paying for four ranks on layers that had two. Adding it took generation from 60.5
to 66.5 tokens/s -- past the 63.3 it had before any of this.

**And one idea that measured to nothing.** A device that computes nothing in the
next subgraph has its copy of this result rewritten by the next collective before
anything reads it, so gathering it here looked like pure traffic. The meta
backend marks the node (`GGML_TENSOR_FLAG_NEEDED`) and the collective can skip
the gather for ranks not marked. In the deployed configuration it is worth
nothing at all: 623.3 against 625.2 tokens/s of prefill and 78.2 against 78.1 of
generation, same output hash. It shipped in the same build as the flat-kernel
mask above, and the decode gain was first credited to it; measuring it on its own
gives that credit back.

It is also off by default for a second reason. "Computes nothing in the next
subgraph" is not "nothing reads this": a residual or a view can reach the tensor
from further along, and the next collective rewrites its own output rather than
necessarily this one. No such read was found in the 27B graph, but not finding
one is not the same as there being none, so the flag needs a consumer analysis
before it is worth switching on. `GGML_CUDA_MIXED_AR_SKIP_GATHER=1` turns it on
for whoever wants to measure it again.

Two things changed along the way and they are easy to conflate, so they are
listed apart. The mechanism steps are changes to the collective, each measured at
a fixed placement; the placement steps are changes to which card holds what, each
measured on the mechanism of the row above it.

| step | kind | prefill | generation |
|---|---|---:|---:|
| four-way tensor split, as deployed | -- | 369.5 | 63.3 |
| 40 layers to the Blackwells, 25 to the Teslas | placement | 367.9 | 51.9 |
| active mask in the reduce-scatter | mechanism | 581.3 | 52.0 |
| KV-bearing layers to the Teslas, one each | placement | 628.8 | 60.5 |
| active mask in the flat kernel | mechanism | 625.9 | 66.5 |
| the Tesla pair shares those layers | placement | 625.0 | **78.4** |

The last row is a placement change, not a mechanism one: giving the pair a layer
rather than giving one Tesla the whole of it is what took generation from 66.5 to
78.4. An earlier version of this table put that step under "gather only where
needed", which shipped in the same period and, measured on its own at the
deployed split, is worth nothing: 625.2 against 623.3 tokens/s of prefill and
78.1 against 78.2 of generation, same output hash.

**Two holes in the verification, both found by this work.**

The reference collective reused the tested call's token. During decode the tested
path *is* the flat kernel, so the reference found the arrival words already set,
its barrier passed without waiting, and it read whatever the slot held. With
every rank publishing the same bytes that was invisible; with a mask, a rank that
publishes nothing leaves the previous call's data there, and 115k elements came
out different. The reference has its own token now, and the decode path is
verified for the first time.

And verification widened `needed_mask` to every rank, which checked everything
except the one thing that flag decides -- then left the reference's result in the
tensor, filling every copy and hiding exactly the reads a skipped gather might
break. It now runs the real mask, compares only where the tested path was
supposed to produce a whole result, and puts the tested path's result back before
carrying on.

### Sharing rather than owning, and where it landed

Giving a Tesla a layer outright means one slow card computes it while three
wait. Giving the *pair* the layer halves that, at the cost of a two-rank
collective between them -- and the two are on the same driver stack, so it is
still host memory, but it is two ranks rather than four and both are the same
speed. Prefill barely notices; generation does:

| layers the Teslas take | prefill | generation |
|---|---:|---:|
| one Tesla each, 12 layers | 628.8 | 66.5 |
| shared by the pair, 16 layers | 591.9 | 73.1 |
| shared by the pair, 14 | 606.8 | 72.6 |
| **shared by the pair, 12** | **625.0** | **78.4** |
| the same at `-ub 512`, 10 | 634.0 | 74.9 |

The last row buys 1.4% of prefill for 4.7% of generation, and a server generates
continuously and prefills once a request, so the deployed split is the fourth.
`-ub 512` also frees enough compute buffer for two more layers on the Blackwells,
which is why it can hold ten KV layers where 1024 holds twelve.

Against the four-way split this replaces: **prefill 369.5 -> 627.1, generation
63.3 -> 78.5**. Everything verified elementwise against the reference reduction,
and clearing both variables restores the previous behaviour exactly, output hash
included.

The reduction shares are still four numbers and all four are still used:
35/35/17/13 now means 35:35 between the Blackwells on the layers they share and
17:13 between the Teslas on the layers they share, rather than one split across
all four. They were tuned against a distribution of work that no longer exists
and want re-measuring against this one -- 17:13 against 1:1 first, since the
Teslas now share whole layers between themselves rather than taking a slice of
everything.

**Where the collective's time goes now.** It costs 1602 ms on the critical rank
against 2786 before, and the shape has changed completely:

| rank | total | publish | wait_pub | reduce | wait_red | gather |
|---|---:|---:|---:|---:|---:|---:|
| 5080 | 1454 | 16.2% | 28.6% | 19.6% | 11.4% | 24.2% |
| 5070 Ti | 1421 | 16.5% | 27.2% | 20.1% | 11.5% | 24.7% |
| V100 SXM2 | 1602 | 5.6% | 7.4% | 9.4% | 3.6% | **74.0%** |
| V100 PCIe | 1593 | 7.4% | 7.5% | 7.1% | 12.8% | **65.3%** |

Measured with `GGML_CUDA_MIXED_AR_SKIP_GATHER=1`, which was the behaviour at the
time and is not the default now -- the flag was made opt-in afterwards, for the
reasons above. So this is the profile of a configuration with the gather skip on;
since the skip measures at nothing in throughput the shape is unlikely to differ
much, but that is an expectation and not a measurement, and the profile under the
default has not been taken.

What it does show is that the Teslas spend two thirds of their collective
gathering. They publish and reduce almost nothing -- they own twelve layers out of
sixty-five -- so what is left is fetching results, and the next thing worth
attacking is which of those fetches are real. A single percentage does not say:
the exchange inside the Blackwell pair, the exchange inside the Tesla pair, and
the activation crossing between pairs are three different things and want
separating, with bytes and time recorded for each alongside the masks and the
size.

**What this does not reach.** A real `-sm layer` run does 942 tokens/s of prefill
against this 627, and 47.6 of generation against 78.5. Expressing pure layer
splitting through this mask gives 550.7 -- worse than either. The obvious
explanation is that a collective with one participant still publishes to host
memory and the next owner reads it back where layer splitting copies device to
device once; that is a hypothesis for a decomposition, not an established cause.
The two arrangements also give the matrix operations different shapes, place the
compute differently and synchronise differently, and none of that has been
separated. What is measured is the trade: the hybrid is the better one for a
server and the worse one for a batch prefill.

**And a claim not to make.** That the collective is bit-exact under this
placement says the reduction is what it was; it says nothing about the model's
quality. Splitting a matrix two ways instead of four changes which partial sums
exist and how each rounds, and fewer participants is not by itself more accurate.
Nothing here measures output quality, and the deployed split changes the model's
text -- an argument for measuring it, not for assuming either direction.

### What the Teslas are actually doing, 2026-09-09

A single "gather is 65-74% of the collective" does not say which gather. The
phase timers are bucketed by the shape of the collective now -- every rank active,
this runtime's own pair only, the other pair only -- and the bytes beside them are
computed on the host from the shard boundaries, so time and volume divide the
same way. Prefill of 1430 tokens plus 200 generated:

| rank | shape | collectives | time | gather share | bytes in from the other pair |
|---|---|---:|---:|---:|---:|
| V100 SXM2 | own pair | 72 | 390 ms | 27% | -- |
| V100 SXM2 | **other pair** | **312** | **1215 ms** | **88.8%** | **3.32 GiB** |
| V100 PCIe | own pair | 72 | 428 ms | 28% | -- |
| V100 PCIe | other pair | 312 | 1168 ms | 78.9% | 3.32 GiB |
| 5080 | own pair | 312 | 813 ms | 29% | -- |
| 5080 | other pair | 72 | 643 ms | 17.5% | 0.77 GiB |

The Teslas move 3.32 GiB in from the Blackwell pair against 0.33 GiB of traffic
inside their own, ten to one, and three quarters of their collective time goes to
the 312 collectives they take no part in at all -- no publication, no reduction,
only fetching a result for a layer that is not theirs.

That is what `GGML_TENSOR_FLAG_NEEDED` was supposed to remove, and it explains
why the flag measured at nothing. Its test is "does this device compute anything
in the next subgraph", and mirrored operations -- norms and the like -- are
computed everywhere, so the answer is almost always yes. The question it needs to
ask is whether anything on this device reads *this tensor*, following the
consumers of the result and its aliases, which is what the review asked for and
what the coarse test stands in for.

The Blackwells show the other side of the same thing: 0.77 GiB in from the Tesla
pair over 72 collectives, and 57.9% of that time waiting for the Teslas to
publish rather than transferring. Their own pair's exchange is 1.66 GiB out and
1.66 GiB back in over 312 collectives, evenly split between publishing, reducing
and gathering -- that one looks balanced and is not where the next gain is.

### Why the cross-pair traffic cannot be dropped at the collective

The obvious reading of the table above is that the Teslas fetch 3.32 GiB they do
not need, and that a precise enough test would drop it. The coarse test --
"computes anything in the next subgraph" -- was replaced with the real question:
does any node this device will compute, anywhere later in the graph, read this
tensor or a view of it, answered in one backward pass over the subgraphs rather
than by scanning forward at every collective.

The answer is that they do need it. Every result is read on every device, so the
precise test marks exactly what the coarse one did and buys the same nothing:
624.2 against 624.9 tokens/s of prefill, 76.6 against 76.1 of generation, same
output hash, and the same 965,345,280 elements compared on every rank.

The reason is upstream of the collective. `GGML_TENSOR_FLAG_COMPUTE` is cleared
only when a source has a zero-length slice, and a mirrored tensor has no slice to
be zero -- so the mirrored operations of a layer, its norms and the like, run on
all four cards whether or not the card owns any of that layer's matrices. Those
operations read the previous layer's output, and that read is what the fetch
serves. The traffic is not unnecessary; the work that consumes it is.

So the 1215 ms is not reachable from inside the collective, and the analysis that
was supposed to reach it is now in place and says so. What would reach it is not
running a layer's mirrored operations on a device that owns nothing in that
layer -- which is a question about how the meta backend assigns mirrored work, not
about how the reduction moves bytes. That is the next thing to look at, and it is
a larger change than anything in this document so far.

The consumer analysis is kept even though it changes no number here. It answers
the question the flag is supposed to answer, where the old test answered a
different one that happened to agree; on a placement where a device really does
stop reading, the two would part company.
