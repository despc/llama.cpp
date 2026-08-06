# V100_CUDA backend

Runs a Tesla V100 next to the consumer GPUs in one `llama-server` process, with a
separate NVIDIA driver stack for each.

The V100 needs the proprietary driver; the RTX 5080 / 5070 Ti need the open one.
Two kernel modules already coexist on this host (`nvidia.ko` and a renamed
`nvidia_v100.ko`). The hard part is user space: both drivers ship a `libcuda`
with the same SONAME, so `ld.so` keeps only one copy, and whichever loads first
serves every GPU.

## How the isolation works

1. **Renamed SONAMEs.** `tools/rename_soname.py` copies the V100 driver stack to
   `libcv*` names (`libcuda.so.1` -> `libcvda.so.1`, and so on). It only rewrites
   bytes of equal length inside `.dynstr` plus the version hashes, so section
   sizes, symbol tables and `.gnu.hash` all stay valid. `patchelf` cannot be used
   here: it rebuilds the whole ELF and corrupts `libcublasLt.so.12` (749 MB).
2. **Private symbol scope.** `libggml-v100-cuda.so` links against the `libcv*`
   copies and is `dlopen`ed with `RTLD_LOCAL`, so its `cu*` symbols never reach
   the system backend.
3. **Device redirect.** `loader/v100_redirect.c` is `LD_PRELOAD`ed. It sees every
   `open()` in the process, so it redirects only calls whose return address lies
   in a `libcv*` module, sending them to `/dev/nvidia-v100-*`. It also renames the
   V100 UVM socket, which would otherwise collide with the system one and make
   `cuInit` return 304.

An earlier attempt used `dlmopen(LM_ID_NEWLM)` instead. It works until a thread
exits: the second namespace gets its own `libc`, so `_dl_deallocate_tls` frees the
thread TLS block with a different allocator and the heap breaks. Do not go back
to it.

## Build

The renamed driver stack is generated once and installed to `/opt/nvidia-v100/lib`:

```sh
cd ggml/src/ggml-v100-cuda
python3 tools/rename_soname.py ~/nvidia-v100-build/NVIDIA-Linux-x86_64-580.159.03/libcuda.so.580.159.03 /tmp/libcvda.so.1
python3 tools/rename_soname.py /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12   /tmp/libcvdart.so.12
python3 tools/rename_soname.py /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.12 /tmp/libcvblasLt.so.12
python3 tools/rename_soname.py /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.12   /tmp/libcvblas.so.12
sudo cp /tmp/libcv*.so.* /opt/nvidia-v100/lib/

gcc -shared -fPIC -O2 -o /tmp/v100_redirect.so loader/v100_redirect.c -ldl
sudo cp /tmp/v100_redirect.so /opt/nvidia-v100/lib/
```

Then build llama.cpp with both CUDA backends:

```sh
cmake -B build-v100 -DGGML_CUDA=ON -DGGML_V100_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=ON
cmake --build build-v100 -j
```

`GGML_V100_CUDA` compiles the stock CUDA backend a second time for `sm_70` only.
The build takes a while - it is the whole CUDA backend, twice.

## Run

`LD_PRELOAD` is required. Nothing else in the environment matters, because the
backend carries an `RPATH` (not `RUNPATH`, which `LD_LIBRARY_PATH` would beat).

```sh
LD_PRELOAD=/opt/nvidia-v100/lib/v100_redirect.so ./bin/llama-server -m model.gguf -ngl 99 ...
```

Devices show up as `V100_CUDA0` next to `CUDA0` / `CUDA1`. `--tensor-split` counts
the V100 first, since `ggml_backend_load_all_from_path` loads `v100-cuda` before
`cuda`.

`scripts/llama-server-v100.sh` wraps this.

## Known limits

- `FLASH_ATTN_EXT` with `hsk=320,hsv=256` aborts on the V100: the MMA kernel asks
  for more dynamic shared memory than `sm_70` allows (96 KB vs 164 KB on Ampere),
  and the supported-op check does not catch it. Head sizes 64/80/128 are fine, so
  ordinary models are not affected. This is an upstream `fattn-mma-f16.cuh` gap,
  not an isolation problem.
- The consumer GPUs and the V100 cannot do peer-to-peer copies - they are on
  different drivers. Cross-device tensor traffic goes through host memory.
