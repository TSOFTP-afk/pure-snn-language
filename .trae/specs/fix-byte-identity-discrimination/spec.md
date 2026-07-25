# 修复字节身份区分问题 Spec

## Why

100K 步 LCCC 真实文本训练发现核心问题：**网络响应跟随字节频率而非字节身份**。

证据：所有字节的"响应占比/语料占比"比值在 1.69-2.12 之间几乎均匀，说明网络对每个字节的"额外响应"是均匀的（~1.8× 频率），**没有对特定字节产生身份级的差异化响应**。

诊断确认两个根因：
1. **频率淹没效应**：高频字节（空格 22.6%）在非偏好柱的激活（5.42），远高于低频字节（0.1%）在偏好柱的激活（0.16），比值 34:1
2. **柱偏好范围不匹配**：当前柱偏好连续范围 `[c×5, c×5+5)` 与 UTF-8 字节离散聚集分布（头字节 E0-EF、续字节 80-BF）不匹配，导致有些柱从不激活、有些柱总是激活

## What Changes

### 阶段 1：频率归一化输入增益（方案 A）
- 在 `load_text_corpus` 中统计字节频率直方图 `g_byte_freq[256]`
- 新增 `compute_freq_norm_gain(byte)` 函数：`gain = clamp(sqrt(mean_freq/freq), 0.3, 3.0)`
- 在 `input_inject_kernel` 中乘上频率归一化增益
- 未加载文本时 gain=1.0（向后兼容 step%256 模式）

### 阶段 2：柱偏好重映射
- 当前：柱 c 偏好字节 `[c×5, c×5+5)`（连续范围，与 UTF-8 分布不匹配）
- 改进：按字节在语料中的频率排序后轮询分配，确保每柱都有高频和低频字节
  - 柱 i 偏好的 5 字节 = 排序后第 `{i, i+50, i+100, i+150, i+200}` 位
- 柱偏好增益保持现状：`gain_in=2.0, gain_out=0.3`（6.67:1 已足够强）
- 新增 `init_column_pref_mapping()` 函数在 `load_text_corpus` 后调用

### 阶段 3：离线解码器（零侵入评估指标）
- 新增 `decoder.cu` / `decoder.cuh`（完全独立，不改训练流程）
- 实现"最大响应字节"算法：对每个 L6 神经元找其响应最强的字节（argmax）
- 对测试文本段，统计 L6 神经元响应，用多数投票解码字节
- 输出解码准确率作为"是否学到字节身份"的硬指标
- 新增独立的 `decoder_main.cpp` 入口或 `--decode` 参数

### 阶段 4：本地分步验证
- 10K 步快速验证：响应比值标准差从 ~0.1 → >0.5
- 100K 步完整验证：js_mean（真实文本）从 0.19 → >0.30
- 解码准确率：作为客观评估指标

### 阶段 5：DGX Spark 部署
- 确认本地验证通过后，部署到 DGX Spark 跑 3M 步完整发育训练
- 验证 PSW 成熟（需 CRITICAL 阶段 800K+ 步）

## Impact

- **Affected specs**:
  - `add-dendritic-compartmentalization`（树突区室化，保持不变）
  - `boost-column-differentiation`（柱间分化，本 spec 增强其效果）
  - `add-thalamic-gating`（丘脑门控，保持不变）
- **Affected code**:
  - `src/stage2e/input_encoding.cu`（核心修改：频率统计 + 归一化增益 + 柱重映射）
  - `src/stage2e/input_encoding.cuh`（新增函数声明）
  - `src/stage2e/config.h`（新增频率归一化参数）
  - `src/stage2e/decoder.cu`（新增文件）
  - `src/stage2e/decoder.cuh`（新增文件）
  - `src/stage2e/decoder_main.cpp`（新增文件，独立入口）
  - `src/stage2e/CMakeLists.txt`（添加 decoder 源文件和 target）

## ADDED Requirements

### Requirement: 字节频率归一化输入增益

系统 SHALL 在加载文本语料时统计字节频率直方图，并在输入注入时对每个字节应用归一化增益，使所有字节的有效激活强度接近。

#### Scenario: 文本已加载
- **WHEN** `load_text_corpus` 成功加载文本
- **THEN** 字节频率直方图 `g_byte_freq[256]` 被填充
- **AND** `compute_freq_norm_gain(byte)` 返回 `clamp(sqrt(mean_freq/freq), 0.3, 3.0)`
- **AND** `input_inject_kernel` 中输入电流 = `POP_CODING_GAIN × freq_norm_gain × column_pref_gain × gate`

#### Scenario: 文本未加载（回退模式）
- **WHEN** 未调用 `load_text_corpus` 或加载失败
- **THEN** `compute_freq_norm_gain(byte)` 返回 1.0
- **AND** 行为与当前完全一致（step%256 循环模式）

#### Scenario: 极端频率字节
- **WHEN** 字节频率为 0（语料中未出现）
- **THEN** 归一化增益 = 3.0（上限 clamp）
- **AND** 不会因除零导致 NaN

### Requirement: 柱偏好重映射

系统 SHALL 在文本加载后重新计算柱偏好映射，按字节频率排序轮询分配，确保每柱都有高频和低频字节偏好。

#### Scenario: 文本已加载
- **WHEN** `load_text_corpus` 成功加载
- **THEN** 调用 `init_column_pref_mapping()` 计算新的柱偏好映射
- **AND** 柱 i 偏好的 5 字节 = 按频率排序后的第 `{i, i+50, i+100, i+150, i+200}` 位
- **AND** 每柱都包含至少 1 个高频字节和至少 1 个低频字节

#### Scenario: 文本未加载
- **WHEN** 未加载文本
- **THEN** 保持当前的连续范围映射 `[c×5, c×5+5)` 作为回退

### Requirement: 离线解码器

系统 SHALL 提供独立的离线解码器，从训练后的 `d_neuron_byte_counts` 读取统计，对测试文本段解码字节，输出解码准确率。

#### Scenario: 解码器运行
- **WHEN** 用户运行 `snn_stage2e_decoder --ckpt PATH --text PATH`
- **THEN** 加载 checkpoint 中的 `d_neuron_byte_counts`（55K×256 矩阵）
- **AND** 对每个 L6 神经元找其响应最强的字节（argmax）
- **AND** 对测试文本的每个字节，统计 L6 神经元响应模式
- **AND** 用多数投票解码预测字节
- **AND** 输出解码准确率（预测正确字节数 / 总字节数）

#### Scenario: 零侵入保证
- **WHEN** 解码器运行
- **THEN** 不修改任何训练流程代码
- **AND** 不修改 scheduler.cu / synapse_kernels.cu / neuron_kernels.cu
- **AND** 仅 read-only 访问 checkpoint 文件

## MODIFIED Requirements

### Requirement: 输入注入公式

原公式：
```
input_current[neuron_idx] += POP_CODING_GAIN × column_pref_gain × gate
```

修改为：
```
input_current[neuron_idx] += POP_CODING_GAIN × freq_norm_gain(byte) × column_pref_gain(byte, col) × gate
```

其中：
- `freq_norm_gain(byte) = clamp(sqrt(mean_freq/freq[byte]), 0.3, 3.0)`（新增）
- `column_pref_gain(byte, col) = gain_in if byte in col_prefs[col] else gain_out`（修改为重映射后的偏好表）

## REMOVED Requirements

### Requirement: 柱偏好连续范围映射

**Reason**: 连续范围 `[c×5, c×5+5)` 与 UTF-8 字节离散聚集分布不匹配，导致柱偏好"有用性"严重不均
**Migration**: 改为按频率排序轮询分配的重映射方案；未加载文本时保持连续范围作为回退
