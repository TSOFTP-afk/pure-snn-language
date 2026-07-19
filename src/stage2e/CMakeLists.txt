# =============================================================================
# Stage 2e: 多层级生物机制增强方案 v4 - Phase 0
# =============================================================================
# 设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md
#
# P0 范围:
#   - 显存分配器 (1.33GB GPU 缓冲)
#   - 调度器骨架 (v4 流水线, 占位 kernel)
#   - 10K 步烟雾测试
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

# UTF-8 源编码 (MSVC 中文 locale 否则会误解析 UTF-8)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
    # 关闭 MSVC 对未使用参数的警告 (P0 占位 kernel 必然有未用参数)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4100>)
    # 关闭 size_t 到 int 转换警告 (P0 简化代码用 int)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4267>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>)
endif()

set(CMAKE_CUDA_STANDARD 17)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA 11.0+ required")
endif()

find_package(CUDAToolkit REQUIRED)

# P0 源文件 (不含 stage0 复用, 完全独立)
set(STAGE2E_P0_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_allocator.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# =============================================================================
# Target: snn_stage2e_p0
# =============================================================================
add_executable(snn_stage2e_p0 ${STAGE2E_P0_SRCS})

target_include_directories(snn_stage2e_p0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# P0 不需要任何 stage0/1/2 编译定义
# 不链接任何外部库 (除 CUDA runtime)

target_compile_options(snn_stage2e_p0 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# RTX 3060 = compute capability 8.6 (Ampere)
set_target_properties(snn_stage2e_p0 PROPERTIES
    CUDA_ARCHITECTURES "86"
)
