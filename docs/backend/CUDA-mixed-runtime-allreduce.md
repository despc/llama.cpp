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

## Profiling controls

`GGML_CUDA_MIXED_AR_PROFILE=1` enables detailed GPU timing for large hierarchical calls. CUDA events measure the local reduction and full rank interval. The hierarchical kernel records `%globaltimer` timestamps for:

- mapped-host publication;
- exposed peer wait;
- mapped-host peer read and local add.

The process prints totals by rank and wire size during context destruction. This mode synchronizes every large reduction while collecting the records and changes scheduling. Use it to split the critical path, not as the final throughput result.

`GGML_CUDA_MIXED_AR_PROFILE=2` also logs every profiled call. High llama.cpp log verbosity may be needed for the per-call `INFO` lines.

`GGML_CUDA_MIXED_AR_CPU_PROFILE=1` measures existing `prepare` and `enqueue` calls with `steady_clock`. It does not introduce CUDA synchronization and has negligible measured overhead. It identifies CPU blocking caused by slot reuse.

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

The current two-slot ring waits on the CPU before a slot is reused. This prevents deep host-side submission and leaves approximately 0.68 seconds of visible GPU bubbles. Removing only these bubbles cannot reach the 1500 tokens/s target: 5000 tokens at that rate requires a total prompt time near 3.33 seconds.

## Optimization order

1. Replace CPU slot-reuse synchronization with a device-side consumed-token handshake. Keep the two-slot memory footprint and make overwrite safety explicit in the kernels.
2. Retest the local Blackwell path with true P2P or NCCL using phase markers. Compare end-to-end critical time, not only the local microbenchmark.
3. Reduce cross-runtime wire traffic or the number of reductions per layer. This is required for a large gain after the slot bubbles are removed.
4. Re-run the same exact 5000-token prompt after every change. Also test token generation because the small flat path has different latency requirements.

The reference Nsight report is stored outside the source tree under `llama.cpp/profiles/qwen27b-q8-3gpu-prefill-5000.nsys-rep` on the measured host.
