# 修复前馈连接 STP 抑郁 Spec

## Why

前馈权重修复后（见 `fix-feedforward-layer-weights`），L4 层保持健康（100% chi2 显著，chi2_mean=3175），但 L2/3/L5/L6 仍然完全不活跃。根因是 STP（短期可塑性）的 resource 在高频输入下急剧衰减，有效电流被削弱约 200 倍，前馈权重提升 3.5 倍无法补偿。

**STP 抑郁稳态分析**（当前抑郁型参数 U_SE=0.2, τ_rec=500, L4 每 3 步发放）：
- utilization 稳态 ≈ 0.75
- resource 稳态 ≈ 0.005
- 有效因子 u×r ≈ 0.75 × 0.005 ≈ 0.004
- `delay_ring_current = weight × resource = 0.42 × 0.005 = 0.0021`
- 每个 L2/3 神经元接收 43 个突触事件：`I_inject = 43 × 0.0021 = 0.09`
- 稳态 synaptic_current ≈ 0.053，dV ≈ 0.006/步，3 步累积 V ≈ 0.02 << 阈值 1.0

**对比 L4**：L4 接收外部输入（群体编码注入，增益 80），不经过 STP，所以不受影响。

生物学依据：皮层前馈连接（L4→L2/3, L2/3→L5, L5→L6）多为**易化型**（facilitating）突触，能在高频输入下维持或增强信号传递（Markram et al. 1998, Thomson & Bannister 2003）。当前实现统一使用抑郁型参数，不符合生物学前馈连接特性。

## What Changes

- **新增** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 中前馈连接 STP 参数宏 `STP_U_FEEDFORWARD`、`STP_TAU_FAC_FEEDFORWARD`、`STP_TAU_REC_FEEDFORWARD`
- **修改** [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `stdp_stp_kernel`，为前馈连接使用易化型 STP 参数
- **修改** [network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu) 突触初始化，为前馈连接标记 STP 类型（复用 `receptor_flags` 高位或新增字段）
- **不动** 抑制性突触（FS/LTS/SOM 亚型）的 STP 参数
- **不动** 横向连接（L2/3→L2/3, L6→L6）、反馈连接（L5→L2/3, L6→L4）、跨柱连接的 STP 参数

## Impact

- Affected specs: `fix-feedforward-layer-weights`（前置修复，已完成）、`fix-calcium-rebound-ltd`（前置修复，已完成）
- Affected code: [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)（新增宏）、[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu)（`stdp_stp_kernel`）、[network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu)（突触初始化标记前馈类型）
- Affected behaviors: 前馈连接（L4→L2/3, L2/3→L5, L5→L6）的 STP 从抑郁型改为易化型，resource 稳态从 ~0.005 提升到 ~0.19，有效信号增强约 32 倍，打破 L2/3 静默困境

## ADDED Requirements

### Requirement: 前馈连接易化型 STP

系统 SHALL 为皮层前馈连接（L4→L2/3, L2/3→L5, L5→L6）使用易化型 STP 参数，使前馈信号在高频输入下能维持传递。

前馈连接的 STP 参数（易化型）：
- `STP_U_FEEDFORWARD = 0.02f`（低初始利用率，易化型特征）
- `STP_TAU_FAC_FEEDFORWARD = 200.0f`（易化时间常数，长于 τ_rec 实现易化）
- `STP_TAU_REC_FEEDFORWARD = 50.0f`（快速恢复，保证 resource 不耗尽）

**易化型 STP 稳态分析**（L4 每 3 步发放）：
- utilization 稳态 ≈ 0.67（利用率随高频上升）
- resource 稳态 ≈ 0.19（快速恢复保证资源不耗尽）
- 有效因子 u×r ≈ 0.67 × 0.19 ≈ 0.13（比抑郁型 0.004 提升 32 倍）
- `delay_ring_current = weight × resource = 0.42 × 0.19 = 0.080`
- 每个 L2/3 神经元接收 43 个突触事件：`I_inject = 43 × 0.080 = 3.44`
- 稳态 synaptic_current ≈ 2.06，dV ≈ 0.22/步，3 步累积 V ≈ 0.66（接近阈值，加上 NMDA 可达阈值）

#### Scenario: L4 高频输入时 L2/3 通过易化型 STP 维持信号
- **WHEN** 偏好柱 L4 以 ~16Hz 发放（每 3 步一次注入，~100 神经元发放）
- **THEN** 前馈突触的 utilization 随高频上升至 ~0.67，resource 因快速恢复维持在 ~0.19
- **AND** L2/3 神经元接收有效电流 I_inject ≈ 3.44，配合 NMDA 电流能越过阈值
- **AND** L2/3 神经元发放，STDP LTP 触发，突触进一步强化

#### Scenario: 抑制性和非前馈连接不受影响
- **WHEN** 抑制性突触（FS/LTS/SOM 亚型）或横向/反馈连接的 pre 脉冲到达
- **THEN** 这些突触仍使用原抑郁型 STP 参数（STP_U_SE/STP_U_SI, STP_TAU_FAC=200, STP_TAU_REC=500）
- **AND** resource 在高频下仍会抑郁，保持局部竞争和抑制控制

## MODIFIED Requirements

### Requirement: stdp_stp_kernel 参数选择

`stdp_stp_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 311-339 行）SHALL 根据突触的 STP 类型标记选择参数：

1. **前馈连接**（标记为易化型）：使用 `STP_U_FEEDFORWARD`、`STP_TAU_FAC_FEEDFORWARD`、`STP_TAU_REC_FEEDFORWARD`
2. **其他兴奋性连接**：使用原 `STP_U_SE`、`STP_TAU_FAC`、`STP_TAU_REC`（抑郁型）
3. **抑制性连接**：使用原 `STP_U_SI`、`STP_TAU_FAC`、`STP_TAU_REC`（抑郁型）

STP 类型标记方案：复用 `BioSynapse.receptor_flags` 的 bit2（0x04）作为前馈连接标志位。`init_syn_fields` 时根据 `is_feedforward` 设置该位。

修改后的 STP kernel 伪代码：
```cpp
// 判断 STP 类型
bool is_feedforward = (s.receptor_flags & 0x04);  // bit2 = 前馈连接标志
float baseline_u, tau_fac, tau_rec;
if (is_feedforward) {
    baseline_u = STP_U_FEEDFORWARD;
    tau_fac = STP_TAU_FAC_FEEDFORWARD;
    tau_rec = STP_TAU_REC_FEEDFORWARD;
} else if (s.receptor_flags & 0x03) {  // 兴奋性 (AMPA|NMDA)
    baseline_u = STP_U_SE;
    tau_fac = STP_TAU_FAC;
    tau_rec = STP_TAU_REC;
} else {  // 抑制性
    baseline_u = STP_U_SI;
    tau_fac = STP_TAU_FAC;
    tau_rec = STP_TAU_REC;
}
float fac_decay = expf(-1.0f / tau_fac);
float rec_recovery = 1.0f - expf(-1.0f / tau_rec);
```

## REMOVED Requirements

无。抑郁型 STP 仍用于非前馈连接，不删除。
