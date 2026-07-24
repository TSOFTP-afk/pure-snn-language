# 树突区室化 Spec（方案 A）

## Why

前一轮修复（`fix-l5-l6-chi2-stagnation`）失败，证明 **PSW 饱和不是 L5/L6 chi2 停滞的根因**。重新诊断发现真正根因：

**Ca²⁺ 回弹 LTD 在 SYNAPTOGENIC 阶段启动后（step 5000）快速摧毁前馈权重**：
- `ca_concentration` 被 clamp 到 1.0（[synapse_kernels.cu:87](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu#L87)）
- ca_excess = 1.0 - 0.15 = 0.85
- 前馈专用 rebound_evidence = 2.0 × 0.85 × 0.5 × 1.0 × 0.5 = **0.425/次**
- 1000 步内 beta 增量 ≈ 141.5，前馈权重从 0.75 跌到 ~0.00053（被完全摧毁）
- **step 6000 起 L5/L6 完全不发放**（l6_spikes=0, l6_ema=0.00）
- chi2 停滞是因为 neuron_byte_counts 在 step 6000 后不再增长

**生物学原理**：真实皮层中，前馈连接（L4→L2/3, L2/3→L5）和反馈连接（L5→L2/3, L6→L4）的 Ca²⁺ 动力学是**区室化隔离**的：
- **基底树突**（接收前馈输入）：Ca²⁺ 信号快且小，富含 calbindin 缓冲蛋白，快速清除 Ca²⁺，不易触发回弹 LTD
- **顶端树突**（接收反馈输入）：Ca²⁺ 信号慢且大，缓冲能力弱，易累积到回弹 LTD 阈值

当前代码中所有突触共用同一个 `ca_concentration` 变量且上限为 1.0，忽略了树突区室化，导致前馈连接被回弹 LTD 错误摧毁。

## What Changes

- **新增** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 中前馈连接专用 Ca²⁺ 动力学参数：
  - `NMDA_CA_TAU_FEEDFORWARD 10.0f`（基底树突快速衰减，原 50.0f 的 1/5）
  - `CA_MAX_FEEDFORWARD 0.12f`（基底树突 Ca²⁺ 上限，低于回弹 LTD 阈值 0.15）
- **修改** [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `synapse_nmda_kernel`（第 76-87 行），为前馈连接使用独立的 Ca²⁺ 衰减率和上限
- **不动** `stdp_dual_trace_kernel` 和 `stdp_arrival_pre_kernel` 的回弹 LTD 触发条件（Ca²⁺ 不会超过阈值，自然不触发）
- **不动** 非前馈连接的 Ca²⁺ 动力学（仍用 NMDA_CA_TAU=50.0, 上限 1.0）

## Impact

- Affected specs: `fix-l5-l6-chi2-stagnation`（前置，已完成但失败）、`fix-stp-continuous-recovery`（前置，已完成）、`fix-calcium-rebound-ltd`（前置，已完成）
- Affected code: [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)（新增 2 个宏）、[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu)（`synapse_nmda_kernel` 第 76-87 行）
- Affected behaviors: 前馈连接的 Ca²⁺ 上限从 1.0 降到 0.12（低于回弹 LTD 阈值 0.15），衰减从 τ=50 加快到 τ=10。前馈权重不再被回弹 LTD 摧毁，L5/L6 预期在 SYNAPTOGENIC 阶段后持续发放，chi2_mean 持续增长。

## ADDED Requirements

### Requirement: 前馈连接树突区室化 Ca²⁺ 动力学

系统 SHALL 为前馈连接（标记为 `RECEPTOR_FLAG_FEEDFORWARD`）使用独立的 Ca²⁺ 衰减时间常数和上限，模拟基底树突的区室化特征。

前馈连接 Ca²⁺ 参数：
- `NMDA_CA_TAU_FEEDFORWARD = 10.0f`（基底树突快速衰减，原 50.0f 的 1/5）
- `CA_MAX_FEEDFORWARD = 0.12f`（基底树突 Ca²⁺ 上限，低于 CA_REBOUND_THRESHOLD=0.15）

**数值验证**：
- 前馈 Ca²⁺ 上限 0.12 < CA_REBOUND_THRESHOLD 0.15 → 回弹 LTD 永不触发
- 前馈 Ca²⁺ 衰减 exp(-1/10) ≈ 0.905（原 exp(-1/50) ≈ 0.980），衰减速度 5 倍
- Ca²⁺ 信号仍保留动态变化（用于 CaMKII 等其他机制），只是幅度受限

#### Scenario: 前馈连接 Ca²⁺ 不触发回弹 LTD
- **WHEN** 前馈连接的 post 神经元发放，synapse_nmda_kernel 更新 ca_concentration
- **THEN** ca_concentration 被限制在 [0, 0.12] 范围内（低于回弹 LTD 阈值 0.15）
- **AND** stdp_dual_trace_kernel 的回弹 LTD 条件 `ca_concentration > 0.15` 永远为 false
- **AND** 前馈权重不被回弹 LTD 削弱，保持稳定传递

#### Scenario: 非前馈连接 Ca²⁺ 动力学不变
- **WHEN** 反馈连接、横向连接、跨柱连接的 post 神经元发放
- **THEN** ca_concentration 仍使用 NMDA_CA_TAU=50.0 衰减，上限 1.0
- **AND** 高 Ca²⁺ 时仍触发回弹 LTD（保留生物学防饱和机制）

## MODIFIED Requirements

### Requirement: synapse_nmda_kernel 前馈连接 Ca²⁺ 更新

`synapse_nmda_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 76-87 行）SHALL 根据前馈标志选择不同的 Ca²⁺ 衰减率和上限：

```cpp
// 前馈连接使用独立的 Ca²⁺ 动力学 (模拟基底树突区室化)
// 生物学原理: 基底树突富含 calbindin 缓冲蛋白, Ca²⁺ 快速清除, 不易触发回弹 LTD
bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
float ca_decay_ff = is_feedforward ? expf(-1.0f / NMDA_CA_TAU_FEEDFORWARD) : ca_decay;
float ca_max_ff = is_feedforward ? CA_MAX_FEEDFORWARD : 1.0f;

s.ca_concentration *= ca_decay_ff;
float ca_inflow = s.nmda_conductance * mg_factor * 1000.0f;
s.ca_concentration += ca_inflow;
if (s.ca_concentration > ca_max_ff) s.ca_concentration = ca_max_ff;
```

## REMOVED Requirements

无。
