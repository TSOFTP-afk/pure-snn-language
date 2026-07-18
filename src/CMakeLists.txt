cmake_minimum_required(VERSION 3.18)
project(SNNDialogue CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA Toolkit（用于让 host C++ 也能 include cuda_runtime.h）
find_package(CUDAToolkit REQUIRED)

# CUDA 标准
set(CMAKE_CUDA_STANDARD 17)

# 架构：RTX 3060 = compute_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# 优化等级
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --ptxas-options=-v")

# 源文件编码：源码使用 UTF-8（无 BOM）+ 含中文注释
# MSVC 在中文 locale 默认按 GBK 解析，必须显式声明 UTF-8
# 否则 enum class 等会被中文注释的字节序列干扰导致词法分析失败
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# 警告（不启用 -Werror，避免 CUDA header 警告阻断编译）

# 包含路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# Host C++ 也需要 cuda_runtime.h（config.h 引用了它）
if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# 源文件
set(CUDA_SOURCES
    cuda/neuron_kernel.cu
    cuda/stdp_kernel.cu
    cuda/network_init.cu
    cuda/synapse_kernel.cu
    cuda/io_kernel.cu
)

set(HOST_SOURCES
    host/main.cpp
    host/network.cpp
    host/trainer.cpp
    host/io.cpp
    host/monitor.cpp
)

# 可执行目标
add_executable(snn_dialogue ${CUDA_SOURCES} ${HOST_SOURCES})

# 链接库
target_link_libraries(snn_dialogue)

# 编译选项（MSVC 用 /W4，GCC/Clang 用 -Wall -Wextra）
target_compile_options(snn_dialogue PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# 安装（可选）
install(TARGETS snn_dialogue DESTINATION bin)
