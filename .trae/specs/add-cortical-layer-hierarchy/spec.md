# 真实皮层层级结构 L4/L2-3/L5/L6 Spec

对应 [人脑差距模块评估](../../../docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md) §4 模块 C（优先级：很高），属于 Phase R2。

## Why

当前柱内是简化的 `sensory/association/motor` 三段式结构（200/600/200），缺少真实新皮层的层级流向和反馈通路。最关键的缺失是 **L6 → 丘脑反馈**：这使已实施的模块 D（丘脑-皮层门控）只能单向调制输入，无法形成皮层-丘脑闭环。模块 C 将柱内结构升级为生物合理的 L4 / L2-3 / L5 / L6 四层，并建立 L6 → 丘脑门控的反馈通路，把单向门控升级为双向闭环。这是评估文档中明确"比继续扩大规模更重要"的方向，也是 Phase R1（跨柱权重消融）失败后转向假设审查的核心候选根因。

## What Changes

- **BREAKING**：重定义柱内布局，从 3 段（200 sensory + 600 association + 200 motor）升级为 4 层（200 L4 + 350 L2/3 + 200 L5 + 250 L6 = 1000），每柱神经元总数不变
- **BREAKING**：重定义 `NeuronStateAdEx.region` 枚举语义：`0=L4, 1=L2/3, 2=L5, 3=L6, 4=前额叶`（原 `3=prefrontal` 改为 `4`）
- 重写 `init_synapses_host` 突触拓扑生成：按生物合理流向规则（L4→L2/3, L2/3→L5, L5→L6, L6→L4 反馈, L2/3 横向, L2/3 跨柱同层）替代当前的 `sensory→assoc→motor→assoc` 循环
- 跨柱突触约束到 **L2/3 同层**（替代当前"同流向"任意层），符合生物皮层"跨柱连接主要发生在 L2/3"的事实
- 修改 `input_inject_kernel`：注入目标从柱首 200 神经元改为 L4 层（仍是柱首 200 神经元，但语义上是 L4）
- 新增 **L6 → 丘脑门控反馈**：在 `thalamic_gate_update_kernel` 中加入 L6 活动项，高 L6 活动→门控关闭（皮层"已收到"），低 L6 活动→门控打开（皮层"还要输入"），形成预测编码式闭环
- 新增层间指标：层间激活顺序（L4→L2/3→L5/6 延迟）、层间字节选择性差异、层间互信息

## Impact

- **Affected specs**：[add-thalamic-gating](../add-thalamic-gating/spec.md)（门控从单向升级为双向闭环）、[fix-architectural-issues](../fix-architectural-issues/spec.md)（柱内布局重定义）、[boost-column-differentiation](../boost-column-differentiation/spec.md)（跨柱突触约束到 L2/3 同层）
- **Affected code**：
  - [src/stage2e/config.h](../../../src/stage2e/config.h)：柱内层尺寸宏、跨柱层级约束
  - [src/stage2e/types.h](../../../src/stage2e/types.h)：`region` 枚举注释更新（字段大小不变）
  - [src/stage2e/network_init.cu](../../../src/stage2e/network_init.cu)：神经元层分配 + 突触拓扑流向规则（核心改动）
  - [src/stage2e/input_encoding.cu](../../../src/stage2e/input_encoding.cu)：注入目标注释/语义对齐 L4
  - [src/stage2e/thalamic_gate.cu](../../../src/stage2e/thalamic_gate.cu) + `.cuh`：新增 L6 反馈输入接口
  - [src/stage2e/scheduler.cu](../../../src/stage2e/scheduler.cu)：层间指标统计 + L6 spike count 计算
- **硬约束（项目记忆，不可违反）**：
  - 80/20 兴奋/抑制比例在每层内独立维持
  - 抑制性突触权重 ∈ [-W_MAX, 0]
  - 1/√K 平衡态权重缩放
  - 不修改 stage0/1/2 代码
  - 总突触数维持 ~10.7M（显存预算 1.5GB 不变）
  - STDP kernel 先计算 delta_w 再更新 last_spike

## ADDED Requirements

### Requirement: L4 丘脑输入层

柱内 L4 层（每柱前 200 神经元） SHALL 作为丘脑输入的唯一入口，`input_inject_kernel` 仅向 L4 神经元注入电流。

#### Scenario: 输入注入到达 L4
- **WHEN** `is_inject_step == true` 且 `launch_input_inject` 被调用
- **THEN** 仅 `region == 0`（L4）的神经元接收 `atomicAdd(&input_current[idx], GAIN * gain * gate)`
- **AND** L2/3 / L5 / L6 神经元不直接接收外部输入电流

### Requirement: L2/3 局部与跨柱整合层

L2/3 层（每柱 200..550 神经元，共 350 个）SHALL 承担柱内横向连接和跨柱连接。跨柱突触 SHALL 约束到 L2/3 同层（pre 和 post 均为 L2/3）。

#### Scenario: 跨柱连接约束到 L2/3
- **WHEN** `init_synapses_host` 生成跨柱突触（`n_inter` 个）
- **THEN** pre 神经元的 `region == 1`（L2/3）且 post 神经元的 `region == 1`（L2/3）
- **AND** 不存在 L4→L4、L5→L5、L6→L6 的跨柱突触

### Requirement: L5 输出层

L5 层（每柱 550..750 神经元，共 200 个）SHALL 作为柱输出层，接收 L2/3 投射并投射到 L6。前额叶投射 SHALL 改为从 L5 发起（替代当前从 association 层发起）。

#### Scenario: L5 输出投射
- **WHEN** pre 神经元 `region == 2`（L5）且属于联合皮层
- **THEN** 其柱内突触目标为 L6（`region == 3`）
- **AND** 其前额叶投射目标为前额叶神经元（`region == 4`）

### Requirement: L6 丘脑反馈层

L6 层（每柱 750..1000 神经元，共 250 个）SHALL 投射到同柱 L4（反馈），并通过 L6 活动调制丘脑门控信号。

#### Scenario: L6 → L4 柱内反馈
- **WHEN** pre 神经元 `region == 3`（L6）
- **THEN** 其柱内突触目标包含 L4（`region == 0`）神经元
- **AND** 该反馈突触权重可被 STDP 调制

### Requirement: L6 → 丘脑门控闭环反馈

`thalamic_gate_update_kernel` SHALL 接收每柱 L6 spike count 作为额外输入，按预测编码逻辑调制 `gate_signal`：高 L6 活动→门控关闭，低 L6 活动→门控打开。

#### Scenario: L6 活动调制门控
- **WHEN** 柱 c 的 L6 spike count 高于其 L6 活动 EMA
- **THEN** `gate_target` 减小（门控关闭，皮层"已收到足够输入"）
- **WHEN** 柱 c 的 L6 spike count 低于其 L6 活动 EMA
- **THEN** `gate_target` 增大（门控打开，皮层"还要输入"）

#### Scenario: 闭环稳定
- **GIVEN** 10K 步烟雾测试
- **WHEN** L6 反馈启用
- **THEN** `spike/step` 仍 ∈ [50, 200]（活动区间不回归）
- **AND** `gate_mean` ∈ [0.3, 0.9]（门控不饱和到 0 或 1）

### Requirement: 层间激活顺序指标

scheduler SHALL 每 10000 步输出层间激活顺序统计，验证 L4 → L2/3 → L5/6 的层级流向。

#### Scenario: 层级激活顺序
- **WHEN** `launch_semantic_eval(step)` 被调用且 `step % 10000 == 0`
- **THEN** 输出每层平均 spike 时间（相对注入步的延迟）
- **AND** 期望顺序为 L4 延迟 < L2/3 延迟 < L5 延迟 < L6 延迟

### Requirement: 层间字节选择性差异

scheduler SHALL 输出每层的字节选择性（卡方显著神经元比例），验证不同层形成不同选择性。

#### Scenario: 层间选择性分化
- **WHEN** 10K 步训练后
- **THEN** L4 / L2/3 / L5 / L6 的卡方显著神经元比例 SHALL 分别报告
- **AND** 至少 2 个层的选择性比例差异 > 0.5%（即使绝对值都低）

## MODIFIED Requirements

### Requirement: 柱内突触流向规则

`init_synapses_host` 中的柱内突触目标选择 SHALL 按以下流向规则（替代当前的 `sensory→assoc, assoc→motor, motor→assoc` 循环）：

| pre 层 | 允许的 post 层（柱内） | 权重范围 |
| --- | --- | --- |
| L4 | L2/3 | [0.057, 0.143]（1/√K 缩放） |
| L2/3 | L2/3（横向）, L5 | [0.057, 0.143] |
| L5 | L6, L2/3（反馈） | [0.057, 0.143] |
| L6 | L4（反馈）, L6（横向） | [0.057, 0.143] |

抑制性突触按 80/20 比例在每层内分配，权重 ∈ [-0.143, -0.057]。

### Requirement: 跨柱突触层级约束

`init_synapses_host` 中的跨柱突触（`n_inter` 个）SHALL 仅在 L2/3 层之间生成（pre 和 post 均为 `region == 1`），替代当前的"同流向任意层"。

### Requirement: region 枚举语义

`NeuronStateAdEx.region` 字段（uint8_t，偏移 33）语义重定义：

| 值 | 旧语义 | 新语义 |
| --- | --- | --- |
| 0 | sensory | **L4** |
| 1 | association | **L2/3** |
| 2 | motor | **L5** |
| 3 | prefrontal | **L6** |
| 4 | — | **prefrontal**（新增） |

字段大小不变（uint8_t），`sizeof(NeuronStateAdEx) == 56` 静态断言不破坏。所有读取 `region` 的代码 SHALL 更新枚举常量。

### Requirement: 前额叶投射起源

前额叶投射（`n_pf` 个突触）SHALL 从 L5 层发起（替代当前从 association 层发起），符合生物皮层"L5 是主要皮层下输出层"的事实。

## REMOVED Requirements

### Requirement: sensory→assoc→motor 循环流向

**Reason**：被新的 L4→L2/3→L5→L6→L4 生物合理流向替代。原循环流向（motor→assoc）无生物学对应，且加剧信号在 association 层内的自我维持。
**Migration**：`init_synapses_host` 中 `pre_layer` 判断逻辑从 3 层（0/1/2）改为 4 层（0/1/2/3），`target_layer` 选择按新流向规则表。

### Requirement: 跨柱突触"同流向"约束

**Reason**：被"L2/3 同层"约束替代。原"同流向"允许 L4→L4、L5→L5 等跨柱连接，违背生物皮层"跨柱连接主要发生在 L2/3"的事实，且加剧跨柱信号平均化（col_ratio 恶化的候选根因之一）。
**Migration**：`init_synapses_host` 中跨柱目标选择从"同 layer 的目标柱"改为"目标柱的 L2/3 层"。
