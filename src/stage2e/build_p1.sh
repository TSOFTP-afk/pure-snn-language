#!/bin/bash
# =============================================================================
# Stage 2e 构建脚本 (Linux / DGX Spark / Blackwell)
# =============================================================================
# 用法:
#   ./build_p1.sh              # 构建 snn_stage2e_p1
#   ./build_p1.sh clean        # 清理后重新构建
#
# 依赖: CUDA Toolkit 13.x, CMake 3.18+, Ninja, GCC 11+
# =============================================================================
set -e

cd "$(dirname "$0")"

# DGX Spark ships the toolkit under /usr/local/cuda, but interactive SSH
# profiles do not always add nvcc to PATH.
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# 清理选项
if [ "$1" = "clean" ]; then
    echo "=== Cleaning build directory ==="
    rm -rf build
fi

# 检查依赖
echo "=== Checking dependencies ==="
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found"; exit 1; }

echo "nvcc:  $(nvcc --version | grep release)"
echo "cmake: $(cmake --version | head -1)"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
    echo "ninja: $(ninja --version)"
else
    GENERATOR="Unix Makefiles"
    echo "ninja: not found; using Unix Makefiles"
fi

mkdir -p build

# CMake 配置 (首次)
if [ ! -f build/CMakeCache.txt ]; then
    echo ""
    echo "=== cmake configure ==="
    cmake -S . -B build -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
else
    echo "CMakeCache.txt exists, skipping configure"
fi

# 编译
echo ""
echo "=== build snn_stage2e_p1 + tests ==="
cmake --build build --target snn_stage2e_p1 stage2e_run_config_test --parallel "$(nproc)"
ctest --test-dir build --output-on-failure

# 验证
echo ""
echo "=== Check binary ==="
if [ -f build/snn_stage2e_p1 ]; then
    ls -lh build/snn_stage2e_p1
    echo "Build OK"
else
    echo "FAILED: snn_stage2e_p1 not found"
    exit 1
fi
