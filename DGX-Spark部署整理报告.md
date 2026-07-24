# THE TRUE AI — DGX Spark 部署整理报告

> **报告日期**：2026-07-25
> **目标平台**：NVIDIA DGX Spark（Grace Blackwell GB10）
> **当前平台**：Windows 11 + RTX 3060 Laptop (6GB) + MSVC
> **目的**：将 Stage 2e SNN 训练从本地笔记本迁移到 DGX Spark，利用 128GB 统一内存跑完整 3M 步发育训练

---

## 一、平台对比

| 维度 | 当前平台（笔记本） | 目标平台（DGX Spark） | 影响 |
|------|------------------|---------------------|------|
| GPU | RTX 3060 Laptop (Ampere sm_86) | GB10 Grace Blackwell (sm_120) | 需更新 CUDA_ARCHITECTURES |
| 显存 | 6 GB GDDR6（独立） | 128 GB 统一内存（CPU-GPU 共享） | **可扩大网络规模 20+ 倍** |
| 算力 | ~10 TFLOPS FP32 | 1 PFLOP FP4 / 250 TFLOPS FP8 | 训练速度大幅提升 |
| OS | Windows 11 | DGX OS（Ubuntu 22.04 基础） | 编译器/路径/脚本需适配 |
| 编译器 | MSVC 2022 (x64 cl.exe) | GCC 11+ / NVCC | 移除 MSVC 特定编译选项 |
| CUDA | 13.3 | 预装 CUDA 13.x | 兼容 |
| 构建 | PowerShell + Ninja | Bash + Ninja / Make | 替换构建脚本 |
| 网络 | - | ConnectX-7 (200 Gbps) | 支持后续多机扩展 |

## 二、当前项目状态快照

### 2.1 代码库

- **最新 commit**：`8d22492 feat(stage2e): 皮层层级+丘脑门控+树突区室化+LCCC真实文本输入`
- **代码位置**：`src/stage2e/`（完全独立，不依赖 stage0/1/2）
- **源文件**（9 个）：
  - `memory_allocator.cu` / `network_init.cu` / `neuron_kernels.cu`
  - `synapse_kernels.cu` / `input_encoding.cu` / `scheduler.cu`
  - `modulatory_kernels.cu` / `thalamic_gate.cu` / `main.cpp`
- **依赖**：仅 CUDA Toolkit，无外部库

### 2.2 网络规模（当前）

| 参数 | 值 | 显存占用 |
|------|-----|---------|
| 神经元总数 | 55,000 | 2.94 MB |
| 突触总数 | 10,700,000 | 816 MB（d_synapses） |
| 持久显存总占用 | - | **1,401 MB** |
| 显存预算 | 1,500 MB | - |
| 实际利用 | 93.4% | 6GB 显存限制 |

### 2.3 训练配置（当前）

| 参数 | 值 |
|------|-----|
| 总步数 | 100,000（已验证） |
| 设计目标步数 | 3,000,000（5 个发育阶段） |
| 发育阶段 | EMBRYO(0-5K) → SYNAPTO(5K-200K) → CRITICAL(200K-800K) → PRUNE(800K-1.5M) → MATURE(1.5M-3M) |
| 训练数据 | `data/lccc_sample_1mb.txt`（1MB LCCC 中文对话子集） |
| 日志间隔 | 1,000 步 |
| 检查点间隔 | 50,000 步 |

### 2.4 最新训练成果（100K 步，RTX 3060）

- 累计脉冲 1.02 亿，avg 1021 spikes/step
- 21,178 个神经元具备显著字节选择性
- 柱间分化 js_mean=0.65（合成数据，达理论上限 94%）
- 四层皮层（L4/L2-3/L5/L6）全部激活，chi2 线性增长
- 20/22 判据通过

---

## 三、迁移所需修改

### 3.1 CMakeLists.txt 修改

**文件**：`src/stage2e/CMakeLists.txt`

**修改点**：

```cmake
# 1. CUDA 架构：sm_86 (Ampere) → sm_120 (Blackwell GB10)
#    同时支持本地调试，改为多架构兼容
set_target_properties(snn_stage2e_p1 PROPERTIES
    CUDA_ARCHITECTURES "86;120"   # 兼容 Ampere + Blackwell
)

# 2. 移除 MSVC 特定的 UTF-8 选项（Linux GCC 默认 UTF-8）
# 保留 if(MSVC) 分支即可，无需删除

# 3. Linux 下 GCC 警告选项已存在 (-Wall -Wextra -Wno-unused-parameter)
# 无需修改

# 4. 可选：启用 Blackwell 特定的优化
target_compile_options(snn_stage2e_p1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    # $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>  # 显式指定
)
```

### 3.2 构建脚本替换

**当前**：`build_p1.ps1`（PowerShell，依赖 VS DevShell）

**新增**：`build_p1.sh`（Bash，Linux 原生）

```bash
#!/bin/bash
# DGX Spark / Linux 构建脚本
set -e

cd "$(dirname "$0")"
mkdir -p build

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found in PATH"
    exit 1
fi

# CMake 配置（首次）
if [ ! -f build/CMakeCache.txt ]; then
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
fi

# 编译
ninja -C build snn_stage2e_p1

# 验证
if [ -f build/snn_stage2e_p1 ]; then
    echo "Build OK: $(ls -lh build/snn_stage2e_p1)"
else
    echo "FAILED: snn_stage2e_p1 not found"
    exit 1
fi
```

### 3.3 运行脚本

**新增**：`run_train.sh`

```bash
#!/bin/bash
# DGX Spark 训练启动脚本
set -e
cd "$(dirname "$0")/.."

# 确保数据文件存在
if [ ! -f data/lccc_sample_1mb.txt ]; then
    echo "ERROR: data/lccc_sample_1mb.txt not found"
    exit 1
fi

# 确保检查点目录存在
mkdir -p src/stage2e/checkpoints

# 启动训练（3M 步完整发育）
cd src/stage2e
./build/snn_stage2e_p1 --steps 3000000 2>&1 | tee training_dgxspark_3m.log
```

### 3.4 无需修改的部分

- **源代码**：所有 `.cu` / `.cuh` / `.cpp` / `.h` 文件无需任何修改（已用标准 C++17/CUDA）
- **UTF-8 编码**：Linux GCC 默认 UTF-8，无需 `/utf-8` 标志
- **路径分隔符**：代码中使用相对路径（`data/...`、`checkpoints/...`），Linux 兼容
- **文件 I/O**：使用标准 `std::ifstream`，跨平台兼容

---

## 四、DGX Spark 上的网络规模扩展建议

DGX Spark 的 128GB 统一内存（vs 当前 6GB）允许大幅扩展网络规模。以下是三档扩展方案：

### 4.1 方案 A：保守迁移（推荐首选）

**目标**：先验证迁移正确性，规模不变

| 参数 | 值 | 说明 |
|------|-----|------|
| 神经元 | 55,000 | 与当前一致 |
| 突触 | 10.7M | 与当前一致 |
| 显存占用 | ~1.4 GB | 占 128GB 的 1.1% |
| 训练步数 | 3,000,000 | 完整 5 阶段发育 |
| 预计训练时间 | 数小时 | GB10 算力远超 RTX 3060 |

**价值**：验证代码在 Blackwell 上的正确性，跑通完整 3M 步发育（当前笔记本无法完成）。

### 4.2 方案 B：中等扩展（验证后尝试）

**目标**：扩大网络规模，提升语义涌现潜力

| 参数 | 值 | 扩展倍数 |
|------|-----|---------|
| 神经元 | 550,000 | 10× |
| 突触 | 107M | 10× |
| 柱数 | 500 | 10× |
| 显存占用 | ~14 GB | 占 128GB 的 11% |
| 训练步数 | 3,000,000 | 同上 |

**需要修改的 config.h 参数**：
```c
#define N_COLUMNS_2E               500      // 50 → 500
#define NEURONS_PER_COLUMN_2E      1000     // 不变
#define N_TOTAL_SYNAPSES_2E        107000000  // 10.7M → 107M
#define VRAM_BUDGET_BYTES          (20LL * 1024 * 1024 * 1024)  // 20 GB
```

### 4.3 方案 C：激进扩展（长期目标）

**目标**：接近小鼠皮层规模

| 参数 | 值 | 扩展倍数 |
|------|-----|---------|
| 神经元 | 5,500,000 | 100× |
| 突触 | 1.07B | 100× |
| 柱数 | 5,000 | 100× |
| 显存占用 | ~140 GB | 占 128GB 的 109%（需统一内存溢出至系统内存） |

**注意**：方案 C 接近 128GB 上限，需启用统一内存溢出（CUDA managed memory），可能影响性能。建议先跑方案 A/B 验证。

---

## 五、部署步骤

### 5.1 前置准备（在 DGX Spark 上）

```bash
# 1. 验证环境
nvcc --version          # CUDA 13.x
gcc --version           # GCC 11+
cmake --version         # CMake 3.18+
ninja --version         # Ninja 1.10+

# 2. 验证 GPU
nvidia-smi              # 应显示 GB10 / 128GB 统一内存
```

### 5.2 代码传输

**方式 1：Git 克隆（推荐）**

```bash
git clone <repo-url> the-true-ai
cd the-true-ai
git checkout master
```

**方式 2：rsync 传输（含未提交的数据文件）**

```bash
# 从笔记本同步到 DGX Spark
rsync -avz --exclude='build/' --exclude='*.exe' --exclude='*.bin' \
    /f/项目/THE\ TRUE\ AI/ user@dgx-spark:~/the-true-ai/
```

### 5.3 构建与测试

```bash
cd the-true-ai/src/stage2e
chmod +x build_p1.sh run_train.sh

# 1. 构建
./build_p1.sh

# 2. 烟雾测试（10K 步，验证迁移正确性）
./build/snn_stage2e_p1 --steps 10000

# 3. 中等测试（100K 步，对比笔记本结果）
./build/snn_stage2e_p1 --steps 100000 2>&1 | tee test_100k.log

# 4. 完整训练（3M 步）
./run_train.sh
```

### 5.4 后台运行（长训练）

```bash
# 使用 nohup 后台运行，避免 SSH 断开中断训练
nohup ./build/snn_stage2e_p1 --steps 3000000 > training_3m.log 2>&1 &
echo $! > training.pid

# 查看进度
tail -f training_3m.log

# 检查进程
ps -p $(cat training.pid)
```

---

## 六、验证清单

### 6.1 构建验证

- [ ] `nvcc --version` 显示 CUDA 13.x
- [ ] `./build_p1.sh` 无错误完成
- [ ] `build/snn_stage2e_p1` 可执行文件生成
- [ ] `./build/snn_stage2e_p1 --help` 显示用法

### 6.2 烟雾测试验证（10K 步）

- [ ] 无 CUDA 错误（kernel launch failure 等）
- [ ] 显存占用 < 2GB（`nvidia-smi`）
- [ ] 累计脉冲 > 100 万
- [ ] burst% > 0.5%
- [ ] 四层皮层（L4/L2-3/L5/L6）均有发放

### 6.3 100K 步对比验证

与笔记本结果对比（应基本一致，允许浮点误差）：

| 指标 | 笔记本（RTX 3060） | DGX Spark 预期 |
|------|-------------------|---------------|
| 累计脉冲 | 102,155,029 | ±5% |
| avg spikes/step | 1,021 | ±5% |
| chi2_mean | 44,721 | ±10% |
| js_mean（合成数据） | 0.65 | ±5% |
| 字节选择性显著神经元 | 21,178 | ±5% |

### 6.4 完整训练验证（3M 步）

- [ ] 跨越所有 5 个发育阶段（EMBRYO→SYNAPTO→CRITICAL→PRUNE→MATURE）
- [ ] PSW 成熟率 > 50%（当前笔记本 100K 步仅 0%）
- [ ] js_mean 在真实文本下 > 0.30（当前 0.19）
- [ ] 检查点文件正常生成（每 50K 步）

---

## 七、风险与注意事项

### 7.1 潜在风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| sm_120 架构兼容性 | 中 | 编译失败 | CUDA 13.x 已支持 Blackwell；保留 sm_86 回退 |
| 统一内存性能 | 低 | 训练变慢 | 1.4GB 占用远小于 128GB，不会溢出 |
| 浮点精度差异 | 低 | 指标小幅偏差 | 允许 ±10% 误差，关注趋势而非绝对值 |
| LCCC 语料路径 | 低 | 加载失败 | 确认 `data/lccc_sample_1mb.txt` 已传输 |
| 长训练中断 | 中 | 数据丢失 | 每 50K 步检查点；使用 `nohup` 或 `tmux` |

### 7.2 关键约束（来自项目记忆，迁移后必须保持）

- STDP kernel 先计算 delta_w 再更新 last_spike
- 抑制性突触 [-W_MAX, 0] 区间
- 80/20 兴奋/抑制比例（每层内独立维持）
- Checkpoint 保存完整 d_synapses_（含 STDP 状态）
- 树突区室化：前馈连接专用 Ca²⁺ 动力学（NMDA_CA_TAU_FEEDFORWARD=10.0f, CA_MAX_FEEDFORWARD=0.12f）
- 文本编解码函数必须运行在 host（`__host__`）

---

## 八、后续工作（部署完成后）

1. **跑通方案 A（保守迁移）**：3M 步完整发育训练，验证 PSW 成熟与语义涌现
2. **实现解码器**：基于 3M 步训练结果，编写语言运动皮层（L6 脉冲→字节映射）
3. **尝试方案 B（10× 扩展）**：500 柱 / 550K 神经元，观察规模效应
4. **多字节序列编码**：迁移 stage1 的 text_codec，实现词级输入
5. **多机扩展**：利用 ConnectX-7 网络连接两台 DGX Spark，256GB 组合内存跑 100× 扩展

---

## 附录：关键文件清单

### 需要传输的文件

| 文件/目录 | 说明 | 大小 |
|----------|------|------|
| `src/stage2e/*.cu` / `*.cuh` / `*.cpp` / `*.h` | 源代码 | ~500 KB |
| `src/stage2e/CMakeLists.txt` | 构建配置 | 4 KB |
| `src/stage2e/build_p1.sh`（新增） | Linux 构建脚本 | 1 KB |
| `src/stage2e/run_train.sh`（新增） | 训练启动脚本 | 1 KB |
| `data/lccc_sample_1mb.txt` | LCCC 中文语料 | 1 MB |
| `docs/` | 设计文档 | < 1 MB |
| `.trae/specs/` | 11 个 spec 设计规格 | < 1 MB |

### 不需要传输的文件

| 文件/目录 | 原因 |
|----------|------|
| `src/stage2e/build/` | Linux 上重新构建 |
| `src/stage2e/*.exe` | Windows 可执行文件 |
| `src/stage2e/checkpoints/*.bin` | 训练产物，重新生成 |
| `src/stage2e/*.log` | 训练日志，重新生成 |
| `src/stage0/` / `stage1/` / `stage2/` | stage2e 完全独立，不依赖 |

---

**报告结束**

本报告基于截至 2026-07-25 的项目状态生成。下一步：在 DGX Spark 上执行方案 A（保守迁移），跑通完整 3M 步发育训练，验证 PSW 成熟与语义涌现是否达成。

参考来源：
- [NVIDIA DGX Spark 产品介绍](https://www.nvidia.cn/products/workstations/dgx-spark/)
- [NVIDIA DGX Spark 软件优化](https://developer.nvidia.cn/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
