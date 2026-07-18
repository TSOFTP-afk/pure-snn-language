# =============================================================================
# 阶段 1：语言感觉皮层 - BPTT + 替代梯度
# =============================================================================
# 目标：自实现 BPTT（不依赖 PyTorch），让 SNN 学会字符级 autoencoder
# 判据：编码-解码往返保真度 > 70%
#
# 设计原则：
#   1. 复用 src/include/ 的 config.h、types.h
#   2. 新增 bptt/ 子目录放置前向/反向 kernel
#   3. 最小验证优先：先 10 神经元 + 50 时间步跑通梯度检查
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(snn_stage1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# UTF-8 编码（解决 MSVC 在中文 locale 下解析 UTF-8 源文件失败的问题）
# 注意：/utf-8 和 /W4 是 MSVC 选项，nvcc 不识别（以 / 开头会被当作输入文件路径，
#       导致 "A single input file is required" 错误）
# 必须用生成器表达式限制只对 CXX 生效，CUDA 单独用 -Xcompiler 传递
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

# MSVC 警告级别（同样只对 CXX 生效）
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# CUDA flags
set(CMAKE_CUDA_STANDARD 17)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "需要 CUDA 11.0+")
endif()

# 让 host 代码也能 include cuda_runtime.h
find_package(CUDAToolkit REQUIRED)

add_executable(snn_stage1
    bptt_demo.cu
    bptt_kernels.cu
    text_codec.cu
    main.cpp
)

target_include_directories(snn_stage1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(snn_stage1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# 指定 CUDA 架构（RTX 3060 = sm_86）
set_target_properties(snn_stage1 PROPERTIES
    CUDA_ARCHITECTURES "86"
)
