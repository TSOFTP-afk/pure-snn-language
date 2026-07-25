# =============================================================================
# Stage 2e: 多层级生物机制增强方案 v4 - Phase 1
# =============================================================================
# 设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md
#
# P1 范围:
#   - 显存分配器 (1.33GB GPU 缓冲)
#   - 网络初始化 (55K 神经元 / 10.7M 突触 / CSR 拓扑)
#   - 快时间尺度 kernel: AdEx + NMDA + STDP + STP + 延迟队列
#   - 群体编码输入注入 (50 柱 × 50 神经元)
#   - 调度器 (v4 流水线, 中/慢时间尺度占位)
#   - 10K 步烟雾测试 + P1 判据检查
#
# 依赖: 仅 CUDA Toolkit, 不复用 stage0/1/2 代码 (完全独立)
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(snn_stage2e LANGUAGES CXX CUDA)

# 默认 Release 配置：避免 MSVC Debug 模式 /RTC1 与 nvcc -O3 传过去的 /O2 冲突
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    # UTF-8 源编码 (MSVC 中文 locale 否则会误解析 UTF-8)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    # 关闭 MSVC 对未使用参数的警告 (P1 占位 kernel 必然有未用参数)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4100>)
    # 关闭 size_t 到 int 转换警告 (简化代码用 int)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4267>)
    # 关闭无符号/有符号比较警告
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4018>)
else()
    # Quote list-valued generator expressions so CMake does not split them into
    # malformed literal arguments such as "$<1:-Wall" on Linux generators.
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>"
    )
endif()

set(CMAKE_CUDA_STANDARD 17)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA 11.0+ required")
endif()

find_package(CUDAToolkit REQUIRED)

# 源文件 (P1 + P2, 不含 stage0 复用, 完全独立)
set(STAGE2E_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/network_init.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/neuron_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/synapse_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/input_encoding.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/modulatory_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/thalamic_gate.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/run_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler_checkpoint.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# =============================================================================
# Target: snn_stage2e_p1
# =============================================================================
add_executable(snn_stage2e_p1 ${STAGE2E_SRCS})

target_include_directories(snn_stage2e_p1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# P1 不需要任何 stage0/1/2 编译定义
# 不链接任何外部库 (除 CUDA runtime)

target_compile_options(snn_stage2e_p1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# CUDA 架构: 同时支持 Ampere (RTX 3060, sm_86) 和 Blackwell (DGX Spark GB10, sm_120)
# CMake 3.18+ 的 CUDA_ARCHITECTURES "a;b" 语法会为两个架构生成代码
# 本地运行用 sm_86, DGX Spark 运行用 sm_120
set_target_properties(snn_stage2e_p1 PROPERTIES
    CUDA_ARCHITECTURES "86;120"
    CUDA_SEPARABLE_COMPILATION ON
)

include(CTest)
if(BUILD_TESTING)
    add_executable(stage2e_run_config_test
        ${CMAKE_CURRENT_SOURCE_DIR}/run_config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_config_test.cpp)
    target_include_directories(stage2e_run_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME stage2e_run_config_test COMMAND stage2e_run_config_test)

    # 离线工具: inspect_ckpt (host-only, 不依赖 CUDA, 可直接用 C++ 编译器)
    add_executable(inspect_ckpt
        ${CMAKE_CURRENT_SOURCE_DIR}/inspect_ckpt.cpp)
    target_include_directories(inspect_ckpt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # 离线工具: decoder (host-only, 不依赖 CUDA runtime, 但包含 config.h/types.h)
    # 注: decoder.cu 中已不再依赖 BioSynapse (改用 v3 section 读取), 故可用 C++ 编译
    add_executable(snn_stage2e_decoder
        ${CMAKE_CURRENT_SOURCE_DIR}/decoder.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/decoder_main.cpp)
    target_include_directories(snn_stage2e_decoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # decoder.cu 用 .cu 扩展名但实际是 host 代码, 关闭 CUDA 编译以加速
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/decoder.cu
        PROPERTIES LANGUAGE CXX)
endif()
