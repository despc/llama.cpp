/* Device node redirect for the isolated V100 driver stack.
 *
 * The host runs two NVIDIA kernel modules: the open "nvidia" driver for the
 * consumer GPUs and a renamed proprietary "nvidia_v100" driver for the Tesla
 * V100. Both drivers ship a libcuda that asks for the same paths, so the V100
 * copy must be pointed at its own device nodes.
 *
 * This library is LD_PRELOADed, so it sees every open() in the process,
 * including the ones from the system libcuda. A redirect must apply only to
 * calls made by the V100 stack. The caller is identified by its return address:
 * if the code that called open() lives in V100_LIB_DIR, the path is rewritten.
 *
 * The caller check runs dladdr, which is not cheap, so it happens only after a
 * cheap prefix test shows the path is an NVIDIA device or procfs node.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

/* The V100 stack is the set of libraries renamed by tools/rename_soname.py:
 * libcvda.so.1, libcvdart.so.12, libcvblas.so.12, libcvblasLt.so.12.
 * Matching on the "libcv" prefix keeps the check independent of install path.
 */
static int from_v100_stack(void * ret_addr) {
    Dl_info info;
    if (!dladdr(ret_addr, &info) || !info.dli_fname) {
        return 0;
    }
    const char * base = strrchr(info.dli_fname, '/');
    base = base ? base + 1 : info.dli_fname;
    return strncmp(base, "libcv", 5) == 0;
}

/* Cheap test: only paths that can name an NVIDIA node are worth inspecting. */
static int maybe_nvidia_path(const char * p) {
    return p && (strncmp(p, "/dev/nvidia", 11) == 0 ||
                 strncmp(p, "/proc/driver/nvidia", 19) == 0);
}

static const char * remap(const char * p) {
    static __thread char buf[256];

    if (strcmp(p, "/dev/nvidiactl") == 0)         return "/dev/nvidia-v100-ctl";
    if (strcmp(p, "/dev/nvidia-uvm") == 0)        return "/dev/nvidia-v100-uvm";
    if (strcmp(p, "/dev/nvidia-uvm-tools") == 0)  return "/dev/nvidia-v100-uvm-tools";

    if (strncmp(p, "/dev/nvidia", 11) == 0 && p[11] >= '0' && p[11] <= '9') {
        snprintf(buf, sizeof(buf), "/dev/nvidia-v100-%s", p + 11);
        return buf;
    }
    if (strncmp(p, "/proc/driver/nvidia-uvm", 23) == 0) {
        snprintf(buf, sizeof(buf), "/proc/driver/nvidia-v100-uvm%s", p + 23);
        return buf;
    }
    if (strncmp(p, "/proc/driver/nvidia-caps", 24) == 0) {
        snprintf(buf, sizeof(buf), "/proc/driver/nvidia-v100-caps%s", p + 24);
        return buf;
    }
    if (strncmp(p, "/proc/driver/nvidia/", 20) == 0) {
        snprintf(buf, sizeof(buf), "/proc/driver/nvidia_v100/%s", p + 20);
        return buf;
    }
    return p;
}

static const char * redirect(const char * p, void * ret_addr) {
    if (!maybe_nvidia_path(p) || !from_v100_stack(ret_addr)) {
        return p;
    }
    return remap(p);
}

int open(const char * p, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open");
    p = redirect(p, __builtin_return_address(0));
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t); va_end(ap);
        return real(p, flags, mode);
    }
    return real(p, flags);
}

int open64(const char * p, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    p = redirect(p, __builtin_return_address(0));
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t); va_end(ap);
        return real(p, flags, mode);
    }
    return real(p, flags);
}

int openat(int dirfd, const char * p, int flags, ...) {
    static int (*real)() = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "openat");
    p = redirect(p, __builtin_return_address(0));
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t); va_end(ap);
        return real(dirfd, p, flags, mode);
    }
    return real(dirfd, p, flags);
}

int openat64(int dirfd, const char * p, int flags, ...) {
    static int (*real)() = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "openat64");
    p = redirect(p, __builtin_return_address(0));
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags);
        mode_t mode = va_arg(ap, mode_t); va_end(ap);
        return real(dirfd, p, flags, mode);
    }
    return real(dirfd, p, flags);
}

/* Both libcuda copies name their UVM socket "@cuda-uvmfd-<ns>-<pid>". Same
 * process means the same name, so the second bind() fails with EADDRINUSE and
 * cuInit reports CUDA_ERROR_OPERATING_SYSTEM. Give the V100 stack its own name.
 */
int bind(int fd, const struct sockaddr * addr, socklen_t len) {
    static int (*real)(int, const struct sockaddr *, socklen_t) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "bind");

    if (addr && addr->sa_family == AF_UNIX && from_v100_stack(__builtin_return_address(0))) {
        const struct sockaddr_un * un = (const struct sockaddr_un *) addr;
        size_t path_len = len - offsetof(struct sockaddr_un, sun_path);

        /* abstract socket: sun_path[0] is NUL, the name follows */
        if (path_len > 1 && un->sun_path[0] == '\0' &&
            strncmp(un->sun_path + 1, "cuda-uvmfd-", 11) == 0) {
            struct sockaddr_un ren;
            size_t tail = path_len - 1 - 11;

            if (sizeof(ren.sun_path) >= 1 + 16 + tail) {
                memset(&ren, 0, sizeof(ren));
                ren.sun_family = AF_UNIX;
                memcpy(ren.sun_path + 1, "cuda-v100-uvmfd-", 16);
                memcpy(ren.sun_path + 1 + 16, un->sun_path + 1 + 11, tail);
                return real(fd, (const struct sockaddr *) &ren,
                            offsetof(struct sockaddr_un, sun_path) + 1 + 16 + tail);
            }
        }
    }
    return real(fd, addr, len);
}
