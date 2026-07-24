# 架构结构性问题修复 Spec

## Why
当前 stage2e 架构在 200K 长测中暴露出 3 个结构性问题：(1) 权重未按平衡态网络理论 1/√K 缩放，导致兴奋/抑制无法动态平衡，神经元活动高度相关；(2) 输入全柱广播（8 感官/50 柱/2.0 增益）导致柱同质化，col_ratio 停滞在 1.23（目标 >2.0）；(3) Ca²⁺ 回弹 LTD 阈值过高（0.5），实际 max_ca 仅 0.35，机制无法触发。这三个问题叠加导致 silhouette 长期停滞在 -0.0254（目标 >0.15），语义聚类无法涌现。

PSW 和 Ca²⁺ 回弹 LTD 已实现但只在突触层面起作用，无法解决网络拓扑层面的对称性破缺问题。需要从架构层面修复才能让局部机制发挥作用。

## What Changes
- 在 `network_init.cu` 中实现 1/√K 权重缩放，兴奋/抑制权重按 `1/sqrt(K_avg)` 衰减，激活平衡态网络的去相关活动机制
- 在 `input_encoding.cu` 中将"全柱广播"改为"柱特异性字节偏好分配"，不同柱对字节空间的不同子集产生强响应
- 在 `config.h` 中调整 Ca²⁺ 回弹 LTD 阈值（0.5 → 0.15），让机制在当前低活动水平下可被触发
- 在 `config.h` 中调整输入增益（POP_CODING_GAIN 2.0 → 30.0）以补偿 1/√K 缩放后的活动衰减（5.0 实测不足以维持网络活动，3 spikes/step；提升至 30.0 后达 132 spikes/step）
- 在 `config.h` 中将非偏好柱增益从 0.3 下调至 0.1（实测 0.3 仍使非偏好柱驱动过强，col_ratio 从 2.61 降至 1.13；0.1 增强柱间对比）
- 在 `main.cpp` 中新增 col_ratio、柱间响应分布熵、平衡态指标（活动相关性）的统计输出，用于验证修复效果
- 在 `main.cpp` 中将 `scheduler.run_semantic_eval(total_steps)` 调用从 P3 判据区移至 [10d] 平衡态判据之前，确保 [10d] 能读到最新的 p3_column_ratio_ 值（原顺序导致 col_ratio_now=0）

## Impact
- Affected specs: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md（不修改设计文档，但实现偏离原设计的均匀随机稀疏连接）
- Affected code:
  - `src/stage2e/config.h` — 新增 1/√K 缩放常量、调整 POP_CODING_GAIN 和 CA_REBOUND_THRESHOLD
  - `src/stage2e/network_init.cu` — 在 4 处突触初始化位置应用 1/√K 缩放因子
  - `src/stage2e/input_encoding.cu` — 重写 input_inject_kernel，实现柱特异性字节偏好
  - `src/stage2e/main.cpp` — 新增平衡态指标统计输出
- **BREAKING**: 权重数值范围大幅变化（兴奋 [0.4,1.0]→[0.029,0.072]），现有检查点不兼容
- **BREAKING**: 输入注入逻辑从全柱广播改为柱特异性分配，需重新初始化网络

## ADDED Requirements

### Requirement: 1/√K 平衡态权重缩放
系统 SHALL 在突触初始化时对所有突触权重乘以缩放因子 `1/sqrt(K_avg)`，其中 K_avg 为平均入度（约 195）。这激活了平衡态网络理论（van Vreeswijk & Sompolinsky 1996）的核心数学条件，使兴奋电流与抑制电流动态抵消到 leading order，残差涨落 O(1) 在神经元间近似独立，从而产生弱相关的脉冲活动——这是响应多样性涌现的前提。

#### Scenario: 权重范围符合 1/√K 缩放
- **WHEN** 网络初始化完成
- **THEN** 兴奋性突触权重 |w| ∈ [0.4/sqrt(195), 1.0/sqrt(195)] ≈ [0.029, 0.072]
- **AND** 抑制性突触权重 |w| ∈ [0.4/sqrt(195), 1.0/sqrt(195)] ≈ [0.029, 0.072]
- **AND** PSW 的 alpha/beta 仍按 |w|/W_MAX 比例分配，W_MAX 保持不变（避免 PSW 机制失效）

#### Scenario: 活动去相关
- **WHEN** 运行 10K 步烟雾测试
- **THEN** 神经元发放率分布的变异系数 CV > 0.5（当前为 ~0.3 的均匀分布）
- **AND** 柱间字节响应 col_ratio > 1.5（当前停滞在 1.23）

### Requirement: 柱特异性字节偏好分配
系统 SHALL 在输入注入时为每个柱分配一个字节偏好子集，使不同柱对字节空间的不同区域产生强响应。柱 c 的偏好字节范围由 `c * (256 / N_COLUMNS) ~ (c+1) * (256 / N_COLUMNS)` 决定，当前字节在柱偏好范围内时增益增强，在范围外时增益衰减。这打破了柱间输入对称性，为 STDP 提供对称破缺源。

#### Scenario: 柱间输入差异化
- **WHEN** 注入字节 b
- **THEN** 偏好该字节的柱（c 满足 b ∈ [c*5, (c+1)*5)）接收增益 ×2.0
- **AND** 非偏好柱接收增益 ×0.3（弱信号保持基础活动，避免静默）
- **AND** 50 个柱的输入信号强度呈双峰分布（10% 强响应 + 90% 弱响应）

#### Scenario: 对称破缺触发分化
- **WHEN** 运行 10K 步后统计柱字节响应
- **THEN** 至少 20 个柱展现出明确的字节偏好（在该字节上的响应 > 平均响应的 1.5 倍）
- **AND** col_ratio > 1.8

### Requirement: Ca²⁺ 回弹 LTD 阈值下调
系统 SHALL 将 CA_REBOUND_THRESHOLD 从 0.5 下调至 0.15，使回弹 LTD 机制在当前低活动水平（max_ca ≈ 0.35）下可被触发。原阈值 0.5 远高于实际 max_ca，导致机制形同虚设。

#### Scenario: 回弹 LTD 实际触发
- **WHEN** 运行 10K 步后统计高 Ca²⁺ 突触
- **THEN** max_ca > 0.15（机制可触发）
- **AND** high_ca_ratio > 0.1%（至少部分突触触发回弹 LTD）
- **AND** 回弹 LTD 有效限制权重过强增长

## MODIFIED Requirements

### Requirement: 输入增益配置
原 POP_CODING_GAIN = 2.0f 是基于权重 [0.4, 1.0] 标定的。1/√K 缩放后权重降至 [0.029, 0.072]，等效驱动电流降低约 14 倍。为维持网络基本活动水平，POP_CODING_GAIN 从 2.0 提升至 30.0。实测：5.0 不足以维持网络活动（3 spikes/step，全网络静默），30.0 配合柱特异性增益后达 132 spikes/step（偏好柱 30×2.0=60，非偏好柱 30×0.1=3）。

#### Scenario: 网络活动维持
- **WHEN** 应用 1/√K 缩放和新增益 30.0
- **THEN** 平均每步 spike 数 ∈ [50, 200]（实测 132，处于正常范围）
- **AND** 不出现全网络静默（spike 数 > 10）或爆发（spike 数 < 1000）

### Requirement: 非偏好柱增益下调
系统 SHALL 将 COLUMN_BYTE_PREF_GAIN_OUT 从 0.3 下调至 0.1，使非偏好柱接收的输入电流进一步衰减。实测 0.3 时非偏好柱驱动仍过强，导致柱间响应趋同（col_ratio 从 2.61 降至 1.13）。0.1 让偏好柱与非偏好柱的输入差异从 2.0/0.3≈6.7 倍提升到 2.0/0.1=20 倍，为 STDP 对称破缺提供更强的对比信号。

#### Scenario: 柱间对比增强
- **WHEN** 应用 GAIN_OUT=0.1
- **THEN** 偏好柱与非偏好柱输入比 ≥ 20:1
- **AND** col_ratio > 1.5（实测需验证）

### Requirement: [10d] 判据 col_ratio 读取时机修复
系统 SHALL 在 [10d] 平衡态网络验证判据读取 `scheduler.p3_column_ratio()` 之前，先调用 `scheduler.run_semantic_eval(total_steps)` 以触发 P3-C 语义聚类评估并更新 `p3_column_ratio_`。原代码中 `run_semantic_eval` 在 P3 判据区（[17] 前）才执行，[10d] 在其之前，导致读到初值 0.0。

#### Scenario: [10d] 判据读取到真实 col_ratio
- **WHEN** 程序执行到 [10d] 判据
- **THEN** `scheduler.p3_column_ratio()` 返回最近一次 run_semantic_eval 计算的真实值
- **AND** 不再为 0.0

## REMOVED Requirements
（无删除项）
