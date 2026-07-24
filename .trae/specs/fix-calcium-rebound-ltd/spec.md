# 修复 Ca²⁺ 回弹 LTD 双重触发 Spec

## Why

模块 C（皮层层级结构 L4/L2-3/L5/L6）实施后，100K 步训练在 40K 步因 silhouette、js_mean 停滞而停止，L2/3/L5/L6 层完全不活跃。诊断表明根因是 `stdp_arrival_pre_kernel` 中存在**无条件触发**的 Ca²⁺ 回弹 LTD（不要求 `post_spike`），与 `stdp_dual_trace_kernel` 中**有条件触发**的回弹 LTD（要求 `post_spike`）形成双重 LTD 风暴。

数值验证：单次 `rebound_evidence = PSW_ETA_BETA × ca_excess × CA_REBOUND_LTD_GAIN × plasticity_gain × M_ij ≈ 20.0 × 0.20 × 0.5 × 1.0 × 0.5 ≈ 1.0`，是初始证据 α+β=0.1 的 **10 倍**，导致 L4→L2/3 突触权重一次性崩塌（从 ~0.75 跌到 ~0.068），L2/3 永久静默，进而 L5（依赖 L2/3）和 L6（依赖 L5）级联失效。

## What Changes

- **移除** `stdp_arrival_pre_kernel`（[synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 285-292 行）中的 Ca²⁺ 回弹 LTD 代码块
- **保留** `stdp_dual_trace_kernel`（第 210-215 行）中已有的 Ca²⁺ 回弹 LTD（含 `post_spike` 条件，符合 BCM 理论）
- **不动** 其他 STDP/PSW 参数（PSW_ETA_BETA、CA_REBOUND_LTD_GAIN、CA_REBOUND_THRESHOLD 保持不变）

## Impact

- Affected specs: 无（独立 bug 修复）
- Affected code: [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 中 `stdp_arrival_pre_kernel` 的回弹 LTD 分支
- Affected behaviors: L4→L2/3 突触不再被 pre 到达时的额外 LTD 削弱，L2/3/L5/L6 层应恢复激活

## ADDED Requirements

无新增需求。

## MODIFIED Requirements

### Requirement: Ca²⁺ 回弹 LTD 触发条件

Ca²⁺ 回弹 LTD（BCM 理论的分子实现）**仅由 post 端发放事件触发**。当突触后神经元发放脉冲（`post_spike == true`）且突触局部 Ca²⁺ 浓度超过阈值（`ca_concentration > CA_REBOUND_THRESHOLD`）时，按 `rebound_evidence = PSW_ETA_BETA × ca_excess × CA_REBOUND_LTD_GAIN × plasticity_gain × M_ij` 累积 β（LTD 证据）。

pre 脉冲到达事件（`stdp_arrival_pre_kernel`）**不再触发** Ca²⁺ 回弹 LTD。pre 到达仅处理标准 STDP 的 LTD 分量（`delta_w = -x_post_trace × STDP_A_MINUS_2E`），这是 pre-before-post 时序的 LTD 表达，与 Ca²⁺ 驱动的回弹 LTD 是两个独立机制，不应叠加。

#### Scenario: L4 高频输入时 L2/3 突触不被过度削弱
- **WHEN** L4 偏好柱以 ~16Hz 频率发放（每 3 步一次注入），L2/3 偶尔发放导致局部 Ca²⁺ 上升至 0.35
- **THEN** pre 脉冲到达时仅应用标准 STDP 的 LTD 分量（`delta_w = -x_post × A_minus ≈ -0.03`），单次 evidence ≈ 0.6（PSW_ETA_BETA × 0.03 × M_ij），与初始证据 0.1 同量级，权重渐进衰减而非一次性崩塌
- **AND** 仅当 L2/3 神经元实际发放（`post_spike == true`）且 ca > 0.15 时，才触发额外的回弹 LTD（防过强机制）

#### Scenario: L2/3/L5/L6 层级联激活
- **WHEN** 修复后跑 100K 步训练
- **THEN** L2/3 层出现非零激活（`layer_active[1] > 0`），L5 层依赖 L2/3 输入激活（`layer_active[2] > 0`），L6 层依赖 L5 输入激活（`layer_active[3] > 0`）
- **AND** 层间激活延迟指标显示 L4→L2/3→L5→L6 的级联顺序（`layer_delay` 单调递增）

## REMOVED Requirements

### Requirement: pre 到达时无条件触发 Ca²⁺ 回弹 LTD

**Reason**: 该机制破坏了 BCM 理论的双向阈值语义——Ca²⁺ 回弹 LTD 是 post 端 Ca²⁺ 超载的反应，应由 post 端事件（post_spike）触发，而非 pre 脉冲到达。pre 到达时无条件触发导致：①与 post 发放时的回弹 LTD 形成双重 LTD；②Ca²⁺ 衰减期间（τ_ca=50 步）每个 pre 到达都触发 LTD，单突触在 50 步内被反复削弱；③单次 evidence（~1.0）远超初始证据（0.1），权重一次性崩塌。

**Migration**: 直接删除该代码块。pre 到达的标准 LTD 分量（`delta_w = -x_post × A_minus`）已由 `stdp_arrival_pre_kernel` 第 258 行处理，移除回弹 LTD 不影响 STDP 的 LTD 表达。
