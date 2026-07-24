# 丘脑-皮层门控 Spec (add-thalamic-gating)

## Why
当前输入注入是固定节奏（每 `INPUT_INJECT_INTERVAL` 步注入一字节）+ 固定增益（`gain_in/gain_out` 常量），不符合真实大脑的丘脑门控机制。真实感知中，丘脑根据内部状态（唤醒、注意、预测误差）动态决定"什么输入进入、什么时候进入、以什么增益进入"。

固定注入导致三个问题：
1. **输入与网络状态脱耦**：网络高活跃时仍强行注入，低活跃时仍静默，无法形成状态依赖学习。
2. **缺乏注意力雏形**：所有字节被同等对待，无法对"意外"或"任务相关"输入增强响应。
3. **调质系统无输入端接口**：DA/ACh 变量已存在但未接入输入门控，无法形成"预测误差→注意增强→学习强化"闭环。

本 spec 引入丘脑门控模块，让输入增益由网络内部状态动态调制，为后续注意力、预测编码、状态依赖学习提供基础。

## What Changes
- 新增 `thalamic_gate.cu/.cuh`：实现丘脑门控状态更新和输入增益调制
- 在 `config.h` 中新增门控相关参数（门控更新率、增益范围、novelty/DA 耦合系数）
- 在 `scheduler.cu` 流水线中插入门控更新步骤（在 input_inject 之前）
- 修改 `input_encoding.cu` 的 `input_inject_kernel`：接受门控增益因子，替代固定 `gain_in/gain_out`
- 在 `main.cpp` 中新增门控指标输出（gate_mean、gate_open_ratio）
- 不修改 `NeuronStateAdEx` 结构（保持 56B 对齐和 checkpoint 兼容性）
- 不修改 `BioSynapse` 结构

## Impact
- Affected specs:
  - `boost-activity-and-column-ratio/spec.md` — 输入增益从固定改为动态，需验证活动不回归
  - `boost-column-differentiation/spec.md` — 门控可能影响柱间分化，需联合验证
  - `docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md` — 实现设计文档 §1.1 中 `thalamic_gate` 概念
- Affected code:
  - `src/stage2e/config.h` — 新增 ~8 个门控参数宏
  - `src/stage2e/thalamic_gate.cu/.cuh` — 新增文件
  - `src/stage2e/scheduler.cu` — 插入门控更新步骤，修改 input_inject 调用
  - `src/stage2e/input_encoding.cu` — kernel 签名增加门控增益参数
  - `src/stage2e/main.cpp` — 新增门控指标输出
  - `src/stage2e/CMakeLists.txt` — 添加新源文件
- **BREAKING**: 输入注入行为变化，现有检查点不兼容（门控状态未保存），需重新初始化
- **风险**: 门控过强可能导致输入被抑制、活动回归；门控过弱可能无效。需设置合理初始参数确保 A 不回归

## ADDED Requirements

### Requirement: 丘脑门控状态结构
系统 SHALL 为每柱维护一个丘脑门控状态 `ThalamicGateState`，包含门控信号 `gate_signal ∈ [0, 1]`（0=完全闭门，1=完全开门）、慢速活动估计 `activity_ema`、novelty 估计 `novelty_ema`。结构大小 SHALL ≤ 16B 以降低显存开销（50 柱 × 16B = 800B，可忽略）。门控状态 SHALL 存储在独立设备缓冲区，不嵌入 NeuronStateAdEx。

#### Scenario: 门控状态初始化
- **WHEN** 网络初始化完成
- **THEN** 每柱 gate_signal = 0.5（半开，中性初始态）
- **AND** activity_ema = 0.0、novelty_ema = 0.0
- **AND** 门控缓冲区大小 = N_COLUMNS_2E × sizeof(ThalamicGateState)

### Requirement: 门控信号由内部状态驱动
系统 SHALL 每步更新门控信号，驱动因素包括：
1. **活动水平**：当前 spike count 相对于滑动平均，活动过低时门控开大（补偿），活动过高时门控关小（保护）
2. **novelty（ACh 代理）**：当前字节与历史字节分布的偏离度，novelty 高时门控开大（注意新输入）
3. **预测误差（DA 代理）**：预留接口，当前阶段用 0 占位，后续接 DA 信号

更新公式：
```
activity_norm = clamp((activity_ema - current_spikes) / activity_ema, -1, 1)
novelty_norm = clamp(novelty_ema, 0, 1)
gate_target = 0.5 + GATE_ACTIVITY_COUP * activity_norm + GATE_NOVELTY_COUP * novelty_norm
gate_target = clamp(gate_target, GATE_MIN, GATE_MAX)
gate_signal += GATE_UPDATE_RATE * (gate_target - gate_signal)
```

#### Scenario: 活动补偿
- **WHEN** 当前 spike count 远低于 activity_ema
- **THEN** activity_norm > 0，gate_target 提升，门控开大，输入增益增加
- **AND** 后续活动应回升

#### Scenario: 活动保护
- **WHEN** 当前 spike count 远高于 activity_ema
- **THEN** activity_norm < 0，gate_target 下降，门控关小，输入增益降低
- **AND** 后续活动应回落

#### Scenario: novelty 增强
- **WHEN** 当前字节 novelty_ema 较高
- **THEN** gate_target 提升，门控开大，新输入被增强
- **AND** novelty 低时门控回归中性

### Requirement: 门控调制输入增益
系统 SHALL 在 input_inject_kernel 中将固定增益替换为门控调制增益：
```
effective_gain_in = POP_CODING_GAIN * COLUMN_BYTE_PREF_GAIN_IN * gate_signal[col]
effective_gain_out = POP_CODING_GAIN * COLUMN_BYTE_PREF_GAIN_OUT * gate_signal[col]
```
门控信号按柱独立调制，偏好柱和非偏好柱同步缩放，保持柱间输入比（GAIN_IN/GAIN_OUT）不变。

#### Scenario: 门控全开
- **WHEN** gate_signal[col] = 1.0
- **THEN** effective_gain = POP_CODING_GAIN × GAIN_IN/OUT（与原行为一致）

#### Scenario: 门控全闭
- **WHEN** gate_signal[col] = 0.0
- **THEN** effective_gain = 0，该柱无输入注入

#### Scenario: 门控半开
- **WHEN** gate_signal[col] = 0.5
- **THEN** effective_gain = 原增益 × 0.5

### Requirement: 活动不回归
系统 SHALL 在引入门控后维持 spike/step ∈ [50, 200]。门控初始态为 0.5（半开），且门控更新率较慢（GATE_UPDATE_RATE ≈ 0.01），确保短期活动不剧烈波动。若实测 spike/step < 50，SHALL 提升 GATE_MIN 至 0.3 或提升初始 gate_signal 至 0.7。

#### Scenario: 活动维持
- **WHEN** 应用门控后运行 10K 步
- **THEN** spike/step ∈ [50, 200]
- **AND** `[7] 发放活动正常` PASS
- **AND** 门控指标 gate_mean ∈ [0.2, 0.9]（不应长期全闭或全开）

## MODIFIED Requirements

### Requirement: 输入注入（更新自 boost-activity-and-column-ratio）
原 input_inject_kernel 使用固定 `gain_in/gain_out`。本 spec 修改 kernel 签名，接受 `const float* gate_signal` 数组（每柱一个），将增益改为 `POP_CODING_GAIN * gain * gate_signal[col]`。`launch_input_inject` 函数 SHALL 额外接收门控信号设备指针。

## REMOVED Requirements
（无删除项）
