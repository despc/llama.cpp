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

The deployed DSOs under `/home/despc/llama.cpp/fork_v100` contain the stable
cross-runtime INT8 path and the separately tested local INT8 implementation.
They do not contain the newest fused INT8 experiment described below. The
production launch script enables only `GGML_CUDA_MIXED_AR_INT8`; both new
experimental modes remain disabled.

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

At this checkpoint the fused kernel has not been copied into the deployed
runtime directory, launched, benchmarked, or checked for numerical
correctness. It must be treated as build-only experimental code. In
particular, do not enable `GGML_CUDA_MIXED_AR_FUSED_INT8` in the production
script until deadlock, rank-equivalence, output-quality, warm-prefill, and
decode tests have passed.

### Tensor split experiments

The `1,1,2` tensor split remains required with the current 262144-token
context. Attempts to shift more model weight from the V100 to the two
Blackwell GPUs failed while reserving the draft context graph:

- `1.2,1.2,1.6`: device 0 failed a 1296.06 MiB allocation;
- `1.1,1.1,1.8`: device 0 failed the same 1296.06 MiB allocation.

These failures are configuration-level VRAM limits, not AllReduce failures.

## Optimization order

1. Validate the fused INT8 kernel for deadlocks and rank-equivalent results on
   a short prompt before collecting performance data.
2. Compare fused INT8 against stable cross-only INT8 with the exact 5000-token
   prompt, including warm runs and detailed phase metrics.
3. Run a broader quality or perplexity evaluation for every INT8 stage before
   treating it as lossless.
4. Reduce or overlap the V100 publication dependency exposed by the faster
   local path. Direct CUDA P2P between the Blackwell devices is unavailable on
   this host, so further local transport work must account for the PHB
   mapped-host path.
5. Reduce the number of cross-runtime reductions per layer. Reaching 1500
   tokens/s requires a total prompt time near 3.33 seconds, so transport
   compression alone is not sufficient.
6. Re-run the same exact 5000-token prompt after every change. Also test token
   generation because the small flat path has different latency requirements.

The reference Nsight report is stored outside the source tree under `llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000.nsys-rep` on the measured host.
