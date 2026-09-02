// Build and run on the mixed-driver host with:
//   g++ -O2 -std=c++17 mixed-ar-host-register-probe.cpp -ldl -o /tmp/mixed-ar-probe
//   LD_PRELOAD=/opt/nvidia-v100/lib/v100_redirect.so /tmp/mixed-ar-probe
#include <dlfcn.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace {

using cuda_error_t = int;
using set_device_t = cuda_error_t (*)(int);
using host_register_t = cuda_error_t (*)(void *, size_t, unsigned int);
using host_get_device_pointer_t = cuda_error_t (*)(void **, void *, unsigned int);
using host_unregister_t = cuda_error_t (*)(void *);
using device_synchronize_t = cuda_error_t (*)();
using memset_t = cuda_error_t (*)(void *, int, size_t);
using get_error_string_t = const char * (*)(cuda_error_t);

constexpr unsigned int cuda_host_register_portable = 0x01;
constexpr unsigned int cuda_host_register_mapped   = 0x02;

struct runtime {
    const char * name;
    void * handle;
    set_device_t set_device;
    host_register_t host_register;
    host_get_device_pointer_t host_get_device_pointer;
    host_unregister_t host_unregister;
    device_synchronize_t device_synchronize;
    memset_t memset;
    get_error_string_t get_error_string;
    void * device_pointer = nullptr;
    bool registered = false;
};

template <typename T>
T symbol(void * handle, const char * name) {
    dlerror();
    void * value = dlsym(handle, name);
    if (const char * error = dlerror()) {
        std::fprintf(stderr, "dlsym(%s): %s\n", name, error);
        std::exit(2);
    }
    return reinterpret_cast<T>(value);
}

runtime load_runtime(const char * name, const char * library) {
    void * handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen(%s): %s\n", library, dlerror());
        std::exit(2);
    }
    return {
        name,
        handle,
        symbol<set_device_t>(handle, "cudaSetDevice"),
        symbol<host_register_t>(handle, "cudaHostRegister"),
        symbol<host_get_device_pointer_t>(handle, "cudaHostGetDevicePointer"),
        symbol<host_unregister_t>(handle, "cudaHostUnregister"),
        symbol<device_synchronize_t>(handle, "cudaDeviceSynchronize"),
        symbol<memset_t>(handle, "cudaMemset"),
        symbol<get_error_string_t>(handle, "cudaGetErrorString"),
    };
}

bool check(runtime & rt, cuda_error_t rc, const char * operation) {
    if (rc == 0) {
        return true;
    }
    std::fprintf(stderr, "%s: %s failed: %s (%d)\n", rt.name, operation, rt.get_error_string(rc), rc);
    return false;
}

} // namespace

int main() {
    constexpr size_t bytes = 2 * 1024 * 1024;
    void * memory = nullptr;
    const int alloc_rc = posix_memalign(&memory, 64 * 1024, bytes);
    if (alloc_rc != 0) {
        std::fprintf(stderr, "posix_memalign: %s\n", std::strerror(alloc_rc));
        return 2;
    }
    std::memset(memory, 0, bytes);

    runtime cuda = load_runtime("CUDA", "libcudart.so.12");
    void * v100_driver = dlopen("/opt/nvidia-v100/lib/libcvda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!v100_driver) {
        std::fprintf(stderr, "dlopen(libcvda): %s\n", dlerror());
        return 2;
    }
    runtime v100 = load_runtime("V100_CUDA", "/opt/nvidia-v100/lib/libcvdart.so.12");
    runtime * runtimes[] = {&cuda, &v100};

    bool ok = true;
    for (runtime * rt : runtimes) {
        ok &= check(*rt, rt->set_device(0), "cudaSetDevice(0)");
        if (!ok) {
            break;
        }
        const cuda_error_t register_rc = rt->host_register(
            memory, bytes, cuda_host_register_portable | cuda_host_register_mapped);
        rt->registered = register_rc == 0;
        ok &= check(*rt, register_rc, "cudaHostRegister(shared, Portable|Mapped)");
        if (!ok) {
            break;
        }
        ok &= check(*rt, rt->host_get_device_pointer(&rt->device_pointer, memory, 0),
                    "cudaHostGetDevicePointer(shared)");
        if (!ok) {
            break;
        }
        std::printf("%s: host=%p device=%p\n", rt->name, memory, rt->device_pointer);
    }

    if (ok) {
        ok &= check(cuda, cuda.set_device(0), "cudaSetDevice(0)");
        ok &= check(cuda, cuda.memset(cuda.device_pointer, 0x5a, 4096), "cudaMemset(mapped)");
        ok &= check(cuda, cuda.device_synchronize(), "cudaDeviceSynchronize()");
        if (ok && static_cast<unsigned char *>(memory)[0] != 0x5a) {
            std::fprintf(stderr, "CUDA write is not visible in shared host memory\n");
            ok = false;
        }
        ok &= check(v100, v100.set_device(0), "cudaSetDevice(0)");
        ok &= check(v100, v100.memset(v100.device_pointer, 0xa5, 4096), "cudaMemset(mapped)");
        ok &= check(v100, v100.device_synchronize(), "cudaDeviceSynchronize()");
        if (ok && static_cast<unsigned char *>(memory)[0] != 0xa5) {
            std::fprintf(stderr, "V100_CUDA write is not visible in shared host memory\n");
            ok = false;
        }
    }

    for (auto it = std::rbegin(runtimes); it != std::rend(runtimes); ++it) {
        runtime & rt = **it;
        if (rt.registered) {
            check(rt, rt.set_device(0), "cudaSetDevice(0)");
            check(rt, rt.host_unregister(memory), "cudaHostUnregister(shared)");
        }
        dlclose(rt.handle);
    }
    dlclose(v100_driver);
    std::free(memory);

    if (!ok) {
        std::fprintf(stderr, "MIXED_AR_HOST_MAPPING=unsupported\n");
        return 1;
    }
    std::printf("MIXED_AR_HOST_MAPPING=supported\n");
    return 0;
}
