# V100_CUDA backend

Runs a Tesla V100 next to the consumer GPUs in one `llama-server` process, with a
separate NVIDIA driver stack for each.

The V100 needs the proprietary driver; the RTX 5080 / 5070 Ti need the open one.
Getting two kernel modules to coexist is a prerequisite, done once - see
"Prerequisites: the two kernel drivers" below. The hard part is user space: both
drivers ship a `libcuda` with the same SONAME, so `ld.so` keeps only one copy,
and whichever loads first serves every GPU.

Read the sections in order - prerequisites, build, run - to reproduce the whole
setup from a machine that only has the open driver installed.

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

## Prerequisites: the two kernel drivers

This backend assumes the host already runs two NVIDIA kernel modules. That setup
is done once, before any of the build steps below.

**Why it is possible at all.** The open 580 `nvidia.ko` has no GSP firmware for
Volta, so it rejects the V100 in `nv_pci_probe` and the kernel releases the PCI
device for another driver to claim. `NVreg_ExcludedGpus` is *not* the mechanism
and must not be used here - it marks a GPU excluded by UUID but keeps it bound.

### 1. Extract the proprietary driver, do not install it

```sh
mkdir -p ~/nvidia-v100-build && cd ~/nvidia-v100-build
wget https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/NVIDIA-Linux-x86_64-580.159.03.run
chmod +x NVIDIA-Linux-x86_64-580.159.03.run
./NVIDIA-Linux-x86_64-580.159.03.run --extract-only --target-dir NVIDIA-Linux-x86_64-580.159.03
```

Installing it would replace the open driver the consumer GPUs need. Only the
kernel module sources and a few user space libraries are taken from the archive.

### 2. Build renamed kernel modules

Both modules are rebuilt from `kernel/` (the proprietary tree, not `kernel-open/`)
into a working copy, with every name that would collide against the open driver
changed:

| Category | From | To |
|---|---|---|
| module / obj-m | `nvidia.o` | `nvidia_v100.o` |
| `MODULE_BASE_NAME` | `nvidia` | `nvidia_v100` |
| `NV_MAJOR_DEVICE_NUMBER` | 195 | 196 |
| pci_driver name, chrdevs | `nvidia`, `nvidiactl` | `nvidia_v100`, `nvidia_v100_ctl` |
| procfs trees | `driver/nvidia*` | `driver/nvidia_v100`, `driver/nvidia-v100-*` |
| exported symbols (17) | `nvidia_*` | `nvidia_v100_*` |
| exported symbols (76) | `nvUvmInterface*` | `nvUvmInterfaceV100*` |

Renamed headers (`nv-linux.h`, `nv-chardev-numbers.h`, `nv_uvm_interface.h`) are
kept as local copies inside the renamed source dir, and its include path must come
*before* `common/inc` so the local copies win.

```sh
cd ~/nvidia-v100-build/NVIDIA-Linux-x86_64-580.159.03/kernel_v100_work
make module SYSSRC=/lib/modules/$(uname -r)/build \
     NV_KERNEL_MODULES="nvidia_v100 nvidia-uvm-v100" -j$(nproc)
modinfo nvidia-uvm-v100.ko | grep -E '^name:|^depends:'   # nvidia_uvm_v100, depends nvidia_v100
```

The UVM module is not optional. CUDA opens `/dev/nvidia-uvm` during init, and
without a private UVM module the V100 fails with `initialization error`.

### 3. Load the modules and create the device nodes

`nvidia_v100.ko` first, then `nvidia-uvm-v100.ko`. Majors are not fixed, so the
nodes are made by reading `/proc/devices`:

```sh
MAJOR=$(grep -m1 '^.* nvidia_v100$' /proc/devices | awk '{print $1}')
mknod -m 666 /dev/nvidia-v100-0   c "$MAJOR" 0
mknod -m 666 /dev/nvidia-v100-ctl c "$MAJOR" 255
UVM_MAJOR=$(grep -m1 'nvidia-v100-uvm' /proc/devices | awk '{print $1}')
mknod -m 666 /dev/nvidia-v100-uvm       c "$UVM_MAJOR" 0
mknod -m 666 /dev/nvidia-v100-uvm-tools c "$UVM_MAJOR" 1
```

All four nodes are required. A systemd unit does the two `insmod`s and runs this
script at boot, otherwise nothing comes back after a reboot.

Check the split with `lspci -d 10de: -k`: the V100 must show
`Kernel driver in use: nvidia_v100`, the consumer GPUs `nvidia`.

### 4. Copy the V100 user space libraries

```sh
sudo mkdir -p /opt/nvidia-v100/lib
cd ~/nvidia-v100-build/NVIDIA-Linux-x86_64-580.159.03
sudo cp libcuda.so.580.159.03 libnvidia-ml.so.580.159.03 /opt/nvidia-v100/lib/
sudo cp nvidia-smi /opt/nvidia-v100/
```

`libcuda.so.580.159.03` is the input for the SONAME renaming in the next section.
`libnvidia-ml` and `nvidia-smi` are only needed to inspect the V100, and they need
their own `LD_PRELOAD` shim because libnvidia-ml hardcodes `/dev/nvidiactl`.

Note there are two separate shims, and they are not interchangeable:
`nvidia_v100_redirect.so` redirects unconditionally and suits a dedicated process
such as `nvidia-smi` or an RPC server; `loader/v100_redirect.c` in this directory
filters by caller and is the one this backend needs.

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
