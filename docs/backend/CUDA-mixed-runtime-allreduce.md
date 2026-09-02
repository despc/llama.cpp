# CUDA mixed-runtime tensor parallelism

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

1. Reduce the bytes crossing to and from the V100. At 3.02 GB/s measured on a
   3.94 GB/s link, the transfer itself is the floor; packed INT6 or INT4 with
   the existing per-4096 scales would cut it close to linearly. This needs a
   real quality evaluation before it can be considered.
2. Reduce the number of cross-runtime reductions per layer. There are 384 large
   reductions per 5000-token prompt. Reaching 1500 tokens/s requires a total
   prompt time near 3.33 seconds, so transport compression alone is not
   sufficient.
3. Extend the duplex drain to the local Blackwell exchange, which still
   publishes a whole chunk before reading the peer's.
4. Run a broader quality or perplexity evaluation for the fused, streamed, and
   local INT8 stages before treating them as lossless.
5. Add a multi-seed rank-equivalence test and a decode regression test for the
   fused and streamed kernels.
6. Re-run the same exact 5000-token prompt after every change. Also test token
   generation because the small flat path has different latency requirements.

The reference Nsight report is stored outside the source tree under `llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000.nsys-rep` on the measured host.
