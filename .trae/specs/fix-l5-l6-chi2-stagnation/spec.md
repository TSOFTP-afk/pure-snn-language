# 修复 L5/L6 chi2 停滞 Spec

## Why

STP 连续恢复修复后（见 `fix-stp-continuous-recovery`），四层全部激活且 L4/L2/3 的 chi2_mean 每 10K 步持续翻倍增长（L4: 3150→31802, L2/3: 611→3833）。但 **L5/L6 的 chi2_mean 从 step 10000 到 100000 完全停滞**（L5=767.25 不变, L6=599.23 不变），表明这两层的字节选择性学习在 10K 步后就停止了。

**根因诊断**：PSW 前馈权重在 10K 步内饱和到 W_MAX=1.5，之后 STDP 无法改变有效权重，L5/L6 的 spike 模式固化。

PSW 饱和的数值验证：
- 初始 α=β=0.05（α+β=0.1）
- 单次 LTP evidence = PSW_ETA_ALPHA × delta_w × plast_gain × M_ij × plast_factor
  = 20.0 × 0.0009 × 1.0 × 0.5 × 0.5 ≈ 0.0045/次
- L5 每注入步发放（每 3 步一次），10K 步内约 3333 次 LTP
- α 增量 ≈ 0.0045 × 3333 ≈ 15
- 权重 w = W_MAX × α/(α+β) = 1.5 × 15/15.05 ≈ 1.495（饱和到 W_MAX）
- 饱和后 dw/dα = W_MAX × β/(α+β)² ≈ 0.00033，每次 LTP 权重变化 ≈ 1.5e-6（几乎为零）

**为何 L4/L2/3 不受影响**：L4 接收外部输入（POP_CODING_GAIN=80），spike 模式由输入+阈值适应决定；L2/3 由 L4 的动态模式驱动。L5/L6 完全依赖前馈权重（不接收外部输入），权重饱和后模式固化。

## What Changes

- **新增** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 中前馈连接专用 PSW 学习率宏 `PSW_ETA_ALPHA_FEEDFORWARD`、`PSW_ETA_BETA_FEEDFORWARD`
- **修改** [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `stdp_dual_trace_kernel` 和 `stdp_arrival_pre_kernel`，为前馈连接使用专用 PSW 学习率
- **修改** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 的 `structural_plasticity` 衰减因子从 0.999 增强到 0.95，定期重置前馈权重饱和
- **不动** 非前馈连接的 PSW 学习率（仍用 PSW_ETA_ALPHA/BETA=20.0）
- **不动** PSW 其他参数（ALPHA_INIT, BETA_INIT, W_MAX 等）

## Impact

- Affected specs: `fix-stp-continuous-recovery`（前置修复，已完成）、`fix-stp-depression`（前置修复，已完成）、`fix-feedforward-layer-weights`（前置修复，已完成）、`fix-calcium-rebound-ltd`（前置修复，已完成）
- Affected code: [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)（新增宏 + 修改衰减因子）、[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu)（`stdp_dual_trace_kernel` 和 `stdp_arrival_pre_kernel` 的 PSW 证据计算）、[scheduler.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu)（`launch_structural_plasticity` 的 decay_factor）
- Affected behaviors: 前馈连接的 PSW 学习率从 20.0 降到 2.0（10 倍减慢饱和），structural_plasticity 衰减从 0.999 增强到 0.95（每千步衰减 5%，重获可塑性）。L5/L6 的 chi2_mean 预期在 100K 步内持续增长而非停滞。

## ADDED Requirements

### Requirement: 前馈连接专用 PSW 学习率

系统 SHALL 为前馈连接（L4→L2/3, L2/3→L5, L5→L6，标记为 `RECEPTOR_FLAG_FEEDFORWARD`）使用独立的 PSW 学习率，减慢权重饱和速度。

前馈连接 PSW 学习率：
- `PSW_ETA_ALPHA_FEEDFORWARD = 2.0f`（原 20.0 的 1/10）
- `PSW_ETA_BETA_FEEDFORWARD = 2.0f`（原 20.0 的 1/10）

**饱和时间估算**（PSW_ETA_ALPHA_FEEDFORWARD=2.0）：
- 单次 LTP evidence = 2.0 × 0.0009 × 0.5 × 0.5 ≈ 0.00045/次
- 10K 步内 α 增量 ≈ 0.00045 × 3333 ≈ 1.5（原 15，减慢 10 倍）
- 权重 w = 1.5 × 1.55/1.6 ≈ 1.45（接近 W_MAX 但未完全饱和）
- 100K 步后 α ≈ 15，但此时 structural_plasticity 衰减已介入

#### Scenario: 前馈权重在 100K 步内不完全饱和
- **WHEN** L5/L6 神经元每 3 步发放（注入步），前馈连接累积 LTP 证据
- **THEN** 10K 步内 α 增量 ≈ 1.5（原 15），权重 ≈ 1.45（接近但不饱和）
- **AND** 100K 步内 structural_plasticity 每 1000 步衰减 5%，定期重置饱和
- **AND** L5/L6 的 chi2_mean 在 100K 步内持续增长（非停滞在 767/599）

#### Scenario: 非前馈连接 PSW 学习率不变
- **WHEN** 横向连接（L2/3→L2/3）、反馈连接（L5→L2/3）、跨柱连接的 STDP 事件发生
- **THEN** 这些突触仍使用原 `PSW_ETA_ALPHA=20.0`、`PSW_ETA_BETA=20.0`
- **AND** 学习速度保持不变，维持快速适应性

## MODIFIED Requirements

### Requirement: stdp_dual_trace_kernel 前馈连接 PSW 证据计算

`stdp_dual_trace_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 135-232 行）SHALL 在计算 PSW 证据时根据前馈标志选择学习率：

```cpp
// 根据前馈标志选择 PSW 学习率
bool is_feedforward = (s.receptor_flags & RECEPTOR_FLAG_FEEDFORWARD);
float eta_alpha = is_feedforward ? PSW_ETA_ALPHA_FEEDFORWARD : PSW_ETA_ALPHA;
float eta_beta  = is_feedforward ? PSW_ETA_BETA_FEEDFORWARD  : PSW_ETA_BETA;

// PSW: delta_w 拆分为 LTP (累加 α) 和 LTD (累加 β)
if (delta_w > 0.0f) {
    float evidence = eta_alpha * delta_w * plasticity_gain * M_ij * plasticity_factor;
    synapse_alpha[i] += evidence;
} else if (delta_w < 0.0f) {
    float evidence = eta_beta * (-delta_w) * plasticity_gain * M_ij * plasticity_factor;
    synapse_beta[i] += evidence;
}

// Ca²⁺ 回弹 LTD 也使用前馈专用 eta_beta
if (post_spike && s.ca_concentration > CA_REBOUND_THRESHOLD) {
    float ca_excess = s.ca_concentration - CA_REBOUND_THRESHOLD;
    float rebound_evidence = eta_beta * ca_excess * CA_REBOUND_LTD_GAIN
                             * plasticity_gain * M_ij;
    synapse_beta[i] += rebound_evidence;
}
```

### Requirement: stdp_arrival_pre_kernel 前馈连接 PSW 证据计算

`stdp_arrival_pre_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 234-306 行）SHALL 同样根据前馈标志选择学习率（与 `stdp_dual_trace_kernel` 一致）。

### Requirement: structural_plasticity 衰减因子增强

[scheduler.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu) 第 1051 行的 `decay_factor` 从 0.999f 改为 0.95f，每 1000 步衰减 5%（原 0.1%），定期重置前馈权重饱和。

**衰减效果**：
- 1000 步后 α/β 衰减 5%（α×0.95, β×0.95，保持 α/(α+β) 比例不变但减小证据强度）
- 10000 步后衰减 40%（0.95^10 ≈ 0.60）
- 100000 步后衰减 99.4%（0.95^100 ≈ 0.006），完全重获可塑性

## REMOVED Requirements

无。
