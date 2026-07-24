# 提升网络活动与柱间分化 Spec

## Why
前序 `fix-architectural-issues` spec 完成了 1/√K 权重缩放与柱特异性输入，但 10K 烟雾测试暴露两个硬阻塞：
1. **活动稀疏**：spike/step=24.8（目标 [50, 200]），mean_fr=0.0007，导致 STDP 事件稀少、PSW mature 仅 1.36%，所有上层指标因统计信号不足而无法评估。
2. **柱间分化不足**：col_ratio=1.33（目标 >2.0），即使偏好/非偏好柱输入增益比已达 20:1，响应比仅 1.33:1，说明输入信号在网络传播中被噪声淹没。

这两个问题是当前距离语义涌现的唯一硬阻塞，必须先解决才能进入 200K/800K 长测。

## What Changes
- 在 `config.h` 中放宽 1/√K 缩放强度：从 `1/sqrt(K)` 改为 `1/sqrt(K/4)`（即缩放因子从 0.0707 提升至 0.1414，约 2 倍），保留平衡态网络的去相关特性但提升单突触驱动强度
- 在 `config.h` 中提升 `POP_CODING_GAIN` 从 30.0 → 80.0（配合缩放放宽，使偏好柱等效输入达 80×2.0=160，非偏好柱 80×0.1=8）
- 在 `config.h` 中提升 `POP_CODING_K_PER_COLUMN` 从 50 → 100（每柱激活神经元数翻倍，增强群体编码信号）
- 在 `config.h` 中将 `COLUMN_BYTE_PREF_GAIN_OUT` 从 0.1 → 0.03（进一步压低非偏好柱，使输入比从 20:1 提升到 66:1）
- 在 `config.h` 中将 `INPUT_INJECT_INTERVAL` 从 5 → 3（提升注入密度，减少静默步）
- 在 `input_encoding.cu` 中验证 kernel 是否需调整以支持 K=100（COL_SENSORY_SIZE_2E=200，100<200 无需改动）
- 在 `main.cpp` 中无新增判据，复用已有 `[10d]`、`[7]`、`[5]`、`[21]` 判据验证

## Impact
- Affected specs: 
  - `fix-architectural-issues/spec.md` — 本 spec 是其后续调参，不修改其内容
  - `docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md` — 不修改设计文档
- Affected code:
  - `src/stage2e/config.h` — 调整 6 个参数常量
  - `src/stage2e/network_init.cu` — 缩放因子计算公式调整（1 处）
  - `src/stage2e/input_encoding.cu` — 无需修改（K 参数化，已支持任意值）
- **BREAKING**: 权重数值范围变化（[0.029, 0.072] → [0.058, 0.144]），现有检查点不兼容
- **BREAKING**: 输入驱动强度大幅提升，需重新初始化网络

## ADDED Requirements

### Requirement: 放宽平衡态权重缩放强度
系统 SHALL 将权重缩放因子从 `1/sqrt(K_avg)` 放宽至 `1/sqrt(K_avg/4)`，即 `2/sqrt(K_avg)`。这保留了平衡态网络理论的核心特性（兴奋/抑制动态平衡、活动去相关），但将单突触驱动强度提升约 2 倍，缓解活动稀疏问题。K_avg/4 的分母修正对应"有效入度"概念：在稀疏激活（每步约 25% 突触活跃）下，实际参与驱动神经元的突触数约为 K_avg/4。

#### Scenario: 权重范围符合新缩放
- **WHEN** 网络初始化完成
- **THEN** 兴奋性突触权重 |w| ∈ [0.4×2/sqrt(195), 1.0×2/sqrt(195)] ≈ [0.057, 0.143]
- **AND** 抑制性突触权重 |w| ∈ [0.4×2/sqrt(195), 1.0×2/sqrt(195)] ≈ [0.057, 0.143]
- **AND** 平衡态去相关特性仍成立（CV > 0.5）

#### Scenario: 网络活动提升
- **WHEN** 运行 10K 步烟雾测试
- **THEN** 平均每步 spike 数 ∈ [50, 200]（当前 24.8，目标提升 2-4 倍）
- **AND** mean_fr > 0.002（当前 0.0007，目标提升约 3 倍）
- **AND** 不出现爆发（spike/step < 1000）

### Requirement: 强化柱间输入对比
系统 SHALL 将 `COLUMN_BYTE_PREF_GAIN_OUT` 从 0.1 下调至 0.03，使偏好柱与非偏好柱的输入比从 20:1 提升至 66:1（2.0/0.03）。同时 `POP_CODING_GAIN` 从 30.0 提升至 80.0，配合缩放宽放后，偏好柱等效驱动 = 80×2.0×0.143 ≈ 22.9（接近单步发放阈值 9.37 的 2.4 倍），非偏好柱 = 80×0.03×0.143 ≈ 0.34（远低于阈值，保持静默）。

#### Scenario: 柱间分化显著
- **WHEN** 运行 10K 步后统计柱字节响应
- **THEN** col_ratio > 2.0（当前 1.33）
- **AND** 偏好柱响应占总响应 > 70%（当前约 55%）
- **AND** 至少 30 个柱展现明确字节偏好（强响应柱数量，当前未测量）

### Requirement: 提升注入密度与群体编码强度
系统 SHALL 将 `POP_CODING_K_PER_COLUMN` 从 50 提升至 100（每柱激活神经元数翻倍，群体编码信号增强 2 倍），并将 `INPUT_INJECT_INTERVAL` 从 5 降至 3（注入密度从 20% 提升至 33%）。

#### Scenario: 注入信号增强
- **WHEN** 注入字节 b
- **THEN** 每柱激活 100 个 sensory 神经元（当前 50），占柱 sensory 层 50%
- **AND** 每 3 步注入一次（当前 5 步），静默步比例从 80% 降至 67%

## MODIFIED Requirements

### Requirement: 网络活动维持（更新自 fix-architectural-issues）
原 spec 要求 spike/step ∈ [50, 200]，实测 24.8 未达成。本 spec 通过 4 项参数调整（缩放放宽、增益提升、K 提升、注入密度提升）协同作用，目标将活动提升至 [50, 200] 范围。预估提升倍数：缩放 2× × 增益 2.67× × K 2× × 注入密度 1.67× ≈ 17.8×，但实际受神经元不应期、抑制反馈等非线性因素限制，预期提升 3-5× 至 75-125 spikes/step。

#### Scenario: 活动达标
- **WHEN** 应用所有参数调整
- **THEN** spike/step ∈ [50, 200]
- **AND** `[7] 发放活动正常 (avg > 10)` PASS
- **AND** `[5] spike count 极差 > 100` PASS（当前 60，活动提升后极差应随之扩大）

## REMOVED Requirements
（无删除项）
