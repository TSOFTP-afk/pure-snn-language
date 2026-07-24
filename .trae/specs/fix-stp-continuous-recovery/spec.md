# 修复前馈连接 STP 连续恢复 Spec

## Why

易化型 STP 修复后（见 `fix-stp-depression`），L2/3 层仅出现 1 个神经元激活（初始 resource=1.0 的暂态），STP 达到稳态后该神经元不再发放（chi2 停滞在 399.60）。评估发现 spec 中的稳态分析有误——**仅改 STP 参数无效，因为 STP kernel 只在 pre 脉冲到达（arrival）时调用，resource 恢复也只在 arrival 时发生**。

**STP 稳态重新分析**（L4 每 3 步 arrival，STP kernel 只在 arrival 时调用）：
- 无论 τ_rec=500 还是 τ_rec=50，resource 恢复只在 arrival 时执行一次
- 每次 arrival 恢复量：τ_rec=500 → 0.002，τ_rec=50 → 0.020（10 倍提升）
- 但消耗也同步增加（u 稳态略高），净效果几乎抵消
- resource 稳态 ≈ 0.005（两种参数下几乎相同）
- 有效因子 u×r ≈ 0.004（无改善）

**根本问题**：STP 的 resource 恢复在生物学上是连续过程，但当前实现只在 arrival 时离散恢复，导致 τ_rec 参数失效。生物学上 STP 恢复是每毫秒都在发生的连续过程（Markram et al. 1998），应该在每步都执行恢复。

## What Changes

- **修改** [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `synapse_nmda_kernel`（每步对所有突触执行），为前馈连接增加每步 resource 恢复
- **修改** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 的 `STP_TAU_REC_FEEDFORWARD` 从 50.0f 改为 3.0f（配合每步恢复，让 resource 稳态达到 ~0.13）
- **不动** `stdp_stp_kernel` 中 arrival 时的 STP 更新逻辑（仍处理 utilization 跳变和 resource 消耗）
- **不动** 非前馈连接的 STP 行为（仍只在 arrival 时恢复）

## Impact

- Affected specs: `fix-stp-depression`（前置修复，已完成但效果不足）、`fix-feedforward-layer-weights`（前置修复，已完成）、`fix-calcium-rebound-ltd`（前置修复，已完成）
- Affected code: [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu)（`synapse_nmda_kernel`）、[config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)（`STP_TAU_REC_FEEDFORWARD`）
- Affected behaviors: 前馈连接的 resource 恢复从离散（arrival 时）改为连续（每步），resource 稳态从 ~0.005 提升到 ~0.13，有效信号增强约 26 倍

## ADDED Requirements

### Requirement: 前馈连接 STP 连续恢复

系统 SHALL 为前馈连接（L4→L2/3, L2/3→L5, L5→L6，标记为 `RECEPTOR_FLAG_FEEDFORWARD`）在每步执行 resource 恢复，模拟生物学上 STP 恢复的连续特性。

前馈连接每步 resource 恢复公式：
```
s.resource += (1.0 - s.resource) * (1.0 - expf(-1.0f / STP_TAU_REC_FEEDFORWARD))
```

恢复在 `synapse_nmda_kernel` 中执行（该 kernel 每步对所有突触调用），仅对 `receptor_flags & RECEPTOR_FLAG_FEEDFORWARD` 的突触生效。

`STP_TAU_REC_FEEDFORWARD = 3.0f`（配合每步恢复，让 3 步内恢复 ~63%）

**连续恢复稳态分析**（L4 每 3 步 arrival，每步恢复 α=1-exp(-1/3)≈0.283）：
- 3 步恢复：`r' = 1-(1-r)×(1-0.283)³ = 1-(1-r)×0.368 = 0.632+0.368r`
- arrival 消耗：`r_new = r' × (1-u)`，u 稳态 ≈ 0.807
- 稳态：`r = (0.632+0.368r) × 0.193 → r ≈ 0.131`
- 有效因子 u×r ≈ 0.807 × 0.131 ≈ 0.106（比原 0.004 提升 26 倍）
- `delay_ring_current = 0.42 × 0.131 = 0.055`
- 每个 L2/3 神经元接收 43 个突触：`I_inject = 43 × 0.055 = 2.37`
- 稳态 synaptic_current ≈ 1.42，dV ≈ 0.15/步，3 步累积 V ≈ 0.45（加 NMDA 可达阈值）

#### Scenario: L4 高频输入时前馈突触 resource 维持高位
- **WHEN** 偏好柱 L4 以 ~16Hz 发放（每 3 步一次注入），前馈突触每步恢复 resource
- **THEN** resource 稳态 ≈ 0.131（每步恢复 28.3%，3 步内恢复 63.2%，arrival 消耗 80.7%×r'）
- **AND** 有效电流 I_inject ≈ 2.37，配合 NMDA 电流能驱动 L2/3 发放
- **AND** L2/3 神经元持续发放（非暂态），STDP LTP 持续触发，突触持续强化

#### Scenario: 非前馈连接 STP 行为不变
- **WHEN** 横向连接（L2/3→L2/3）、反馈连接（L5→L2/3）、跨柱连接的 pre 脉冲到达
- **THEN** 这些突触的 resource 仍只在 arrival 时恢复（`stdp_stp_kernel` 逻辑不变）
- **AND** resource 在高频下仍会抑郁，保持局部竞争和抑制控制

## MODIFIED Requirements

### Requirement: synapse_nmda_kernel 前馈连接 resource 恢复

`synapse_nmda_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 45-103 行）SHALL 在每步对前馈连接执行 resource 恢复，紧接在电导衰减之后、钙浓度更新之前：

```cpp
// 前馈连接: 每步连续恢复 resource (生物学: STP 恢复是连续过程)
// 修复: 原只在 arrival 时恢复导致 resource 稳态≈0.005, 信号被削弱~200倍
// 每步恢复让 resource 稳态≈0.13, 有效信号增强~26倍
if (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD) {
    float rec_recovery = 1.0f - expf(-1.0f / STP_TAU_REC_FEEDFORWARD);
    s.resource += (1.0f - s.resource) * rec_recovery;
}
```

### Requirement: STP_TAU_REC_FEEDFORWARD 参数

[config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 中 `STP_TAU_REC_FEEDFORWARD` 从 50.0f 改为 3.0f，配合每步恢复让 resource 稳态达到 ~0.131。

## REMOVED Requirements

无。
