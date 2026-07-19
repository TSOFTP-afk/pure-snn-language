# =============================================================================
# Stage 2: Joint Cortex Development - Columnar Topology + Unsupervised STDP
# =============================================================================
# Goal: replace stage0's uniform-random sparse connectivity with a columnar
#       topology (10 columns x 1000 neurons, dense intra-column / sparse
#       inter-column), feed DailyDialog character streams, and run 1M-step
#       unsupervised STDP training.
#
# Build principle:
#   1. Reuse stage0 CUDA kernels (neuron/stdp/synapse/io) and host wrappers
#      (network/io/monitor). Do NOT link network_init.cu (replaced by
#      columnar_topology.cu), trainer.cpp (replaced by unsupervised_trainer.cu),
#      or main.cpp (replaced by stage2/main.cpp).
#   2. SNNNetwork::initialize() still references init_synapses; we compile
#      network.cpp with -DSNN_STAGE2_BUILD so that reference is excluded.
#      Stage2 calls allocate_only() + init_columnar_synapses() instead.
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(snn_stage2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# UTF-8 source encoding (MSVC in Chinese locale otherwise mis-parses UTF-8).
# /utf-8 and /W4 are MSVC-only; nvcc rejects them, so route via generator expr.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

set(CMAKE_CUDA_STANDARD 17)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA 11.0+ required")
endif()

find_package(CUDAToolkit REQUIRED)

# Stage2 own source files (training)
set(STAGE2_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/columnar_topology.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/text_codec_ext.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/text_stream.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unsupervised_trainer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/competition.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# Stage2 analyzer source files (analysis only, no training main)
set(STAGE2_ANALYZE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/columnar_topology.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/text_codec_ext.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/text_stream.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unsupervised_trainer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/competition.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/analyzer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/analyze_main.cpp
)

# Reused stage0 CUDA kernels (4 files; excluding network_init.cu)
set(STAGE0_REUSE_CU
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/neuron_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/stdp_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/synapse_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/io_kernel.cu
)

# Reused stage0 host wrappers (3 files; excluding trainer.cpp and main.cpp)
set(STAGE0_REUSE_CPP
    ${CMAKE_CURRENT_SOURCE_DIR}/../host/network.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../host/io.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../host/monitor.cpp
)

# =============================================================================
# Target 1: snn_stage2 (training executable)
# =============================================================================
add_executable(snn_stage2
    ${STAGE2_SRCS}
    ${STAGE0_REUSE_CU}
    ${STAGE0_REUSE_CPP}
)

target_include_directories(snn_stage2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(snn_stage2 PRIVATE SNN_STAGE2_BUILD)

target_compile_options(snn_stage2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

set_target_properties(snn_stage2 PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# =============================================================================
# Target 2: snn_stage2_analyze (analysis executable)
# =============================================================================
add_executable(snn_stage2_analyze
    ${STAGE2_ANALYZE_SRCS}
    ${STAGE0_REUSE_CU}
    ${STAGE0_REUSE_CPP}
)

target_include_directories(snn_stage2_analyze PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(snn_stage2_analyze PRIVATE SNN_STAGE2_BUILD)

target_compile_options(snn_stage2_analyze PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(snn_stage2_analyze PROPERTIES
    CUDA_ARCHITECTURES "86"
)
