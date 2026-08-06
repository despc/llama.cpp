cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

# V100 CUDA backend — isolated proprietary driver stack for Tesla V100.
# Uses dlmopen + nvidia_v100_redirect.so to redirect /dev/nvidia* → /dev/nvidia-v100-*.

find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found (required for V100_CUDA backend)")
endif()

message(STATUS "V100_CUDA: CUDA Toolkit ${CUDAToolkit_VERSION} found")

# V100 is sm_70 only — override any global CMAKE_CUDA_ARCHITECTURES.
# The V100 backend must NOT compile for sm_120/sm_120a (Blackwell-only instructions fail on Volta).
set(CMAKE_CUDA_ARCHITECTURES 70-real)
message(STATUS "V100_CUDA: Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

# ---- Collect sources from the original CUDA backend ----
# We compile the original .cu files from ggml-cuda/ with V100 defines.
# The wrapper file ggml-v100-cuda.cu #includes ggml-cuda.cu with our overrides.

set(GGML_V100_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda")

# The wrapper (includes the entire ggml-cuda.cu).
set(GGML_SOURCES_V100_CUDA
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-v100-cuda.cu
)

# All .cu files from the original backend EXCEPT ggml-cuda.cu (included by wrapper).
file(GLOB GGML_SOURCES_CUDA_ORIG "${GGML_V100_CUDA_DIR}/*.cu")
list(REMOVE_ITEM GGML_SOURCES_CUDA_ORIG "${GGML_V100_CUDA_DIR}/ggml-cuda.cu")
list(APPEND GGML_SOURCES_V100_CUDA ${GGML_SOURCES_CUDA_ORIG})

# Template instances.
file(GLOB SRCS "${GGML_V100_CUDA_DIR}/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_V100_CUDA ${SRCS})
file(GLOB SRCS "${GGML_V100_CUDA_DIR}/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_V100_CUDA ${SRCS})
file(GLOB SRCS "${GGML_V100_CUDA_DIR}/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_V100_CUDA ${SRCS})
file(GLOB SRCS "${GGML_V100_CUDA_DIR}/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_V100_CUDA ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB SRCS "${GGML_V100_CUDA_DIR}/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_V100_CUDA ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_V100_CUDA
        ${GGML_V100_CUDA_DIR}/template-instances/fattn-vec-instance-f16-f16.cu
        ${GGML_V100_CUDA_DIR}/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ${GGML_V100_CUDA_DIR}/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ${GGML_V100_CUDA_DIR}/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

# Headers (for IDE / install purposes).
file(GLOB GGML_HEADERS_V100_CUDA "${GGML_V100_CUDA_DIR}/*.cuh")
list(APPEND GGML_HEADERS_V100_CUDA "../../include/ggml-cuda.h")

# ---- Build as dynamic backend ----
ggml_add_backend_library(ggml-v100-cuda
    ${GGML_HEADERS_V100_CUDA}
    ${GGML_SOURCES_V100_CUDA}
)

# Defines matching the original backend.
add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

if (GGML_CUDA_GRAPHS)
    add_compile_definitions(GGML_CUDA_USE_GRAPHS)
endif()

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_VMM)
    add_compile_definitions(GGML_CUDA_NO_VMM)
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

# ---- Link with the renamed V100 driver stack ----
# The system CUDA backend and this one live in the same process. ld.so keys
# libraries by SONAME, so a second copy of libcuda.so.1 would be dropped in
# favour of the already loaded system one. tools/rename_soname.py produces
# libcv* copies with distinct SONAMEs, which lets both stacks coexist.
#
# The backend .so is dlopen'ed with RTLD_LOCAL, so the cu* symbols resolved
# here stay private to this dependency tree and never reach the system backend.
set(V100_LIB_DIR "/opt/nvidia-v100/lib")
target_link_libraries(ggml-v100-cuda PRIVATE
    "${V100_LIB_DIR}/libcvdart.so.12"
    "${V100_LIB_DIR}/libcvblas.so.12"
    "${V100_LIB_DIR}/libcvblasLt.so.12"
    "${V100_LIB_DIR}/libcvda.so.1"
    dl)

# RPATH, not RUNPATH: RUNPATH is searched after LD_LIBRARY_PATH, so a stray
# /usr/lib entry in the environment would pull in the system driver instead.
target_link_options(ggml-v100-cuda PRIVATE "LINKER:--disable-new-dtags")

set_target_properties(ggml-v100-cuda PROPERTIES
    BUILD_RPATH "${V100_LIB_DIR}"
    INSTALL_RPATH "${V100_LIB_DIR}")

# Include paths: need ggml-cuda/ for the .cu files' relative includes.
target_include_directories(ggml-v100-cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GGML_V100_CUDA_DIR}
    ..)

# CUDA compile flags (same as original backend).
set(CUDA_CXX_FLAGS "")
set(CUDA_FLAGS -use_fast_math -extended-lambda)

if (GGML_CUDA_DEBUG)
    list(APPEND CUDA_FLAGS -lineinfo)
    add_compile_definitions(GGML_CUDA_DEBUG)
endif()

if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
endif()

if (GGML_FATAL_WARNINGS)
    list(APPEND CUDA_FLAGS -Werror all-warnings)
endif()

if (GGML_ALL_WARNINGS AND NOT MSVC)
    set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
    if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
        list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
    endif()

    execute_process(
        COMMAND ${NVCC_CMD} -Xcompiler --version
        OUTPUT_VARIABLE CUDA_CCFULLVER
        ERROR_QUIET
    )

    if (NOT CUDA_CCFULLVER MATCHES clang)
        set(CUDA_CCID "GNU")
        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
            OUTPUT_VARIABLE CUDA_CCVER
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        if (CUDA_CCFULLVER MATCHES Apple)
            set(CUDA_CCID "AppleClang")
        else()
            set(CUDA_CCID "Clang")
        endif()
        string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
    endif()

    message(STATUS "V100_CUDA: CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

    ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
    list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})
endif()

if (NOT MSVC)
    list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
endif()

list(JOIN CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)

if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
    list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
endif()

target_compile_options(ggml-v100-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")