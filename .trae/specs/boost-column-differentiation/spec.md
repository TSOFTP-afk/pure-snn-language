# 提升柱间分化 Spec (boost-column-differentiation)

## Why
前序 `boost-activity-and-column-ratio` spec 成功解决了活动稀疏（A: spike/step 24.8→127.5），但柱间分化（B）反而恶化（col_ratio 1.33→1.13）。根因分析确认：跨柱兴奋性突触权重 `randf(0.5f, 1.0f) * w_scale` 与柱内 `randf(0.4f, 1.0f) * w_scale` 几乎一样强，活动提升后偏好柱的强响应通过跨柱兴奋传播淹没所有柱，柱间差异被抹平。Task 6.1 尝试将输入比从 66:1 提升至 100:1 仍无效（col_ratio=1.12），证明问题不在输入端而在突触传播端。

本 spec 通过降低跨柱突触权重（约 3 倍），让柱内处理强、跨柱传播弱，从架构层面恢复柱间分化能力。这符合真实皮层"柱内强连接、跨柱弱连接"的小世界拓扑（Braitenberg & Schüz 1998, Bassett & Bullmore 2006）。

## What Changes
- 在 `config.h` 中新增 4 个跨柱突触权重范围宏：`CROSS_COL_W_EXC_MIN/MAX`、`CROSS_COL_W_INH_MIN/MAX`（参数化便于消融切换）
- 在 `network_init.cu` 跨柱突触初始化段（行 392-397）将权重范围从硬编码 `randf(0.5f, 1.0f)`/`randf(-1.0f, -0.5f)` 改为使用新宏
- 柱内突触权重（行 322-326）保持不变（`[0.4, 1.0]`/`[-1.0, -0.4]`），确保柱内处理能力
- 前额叶长程投射权重（行 457-461）保持不变，前额叶独立于柱分化逻辑
- 不修改 `input_encoding.cu`、`synapse_kernels.cu`、`scheduler.cu` 等其他文件

## 消融实验设计 (R1)
根据人脑差距模块评估文档的"防止无限调参"原则，本 spec 采用**三档预注册消融**，失败后停止搜索权重，转向假设审查：

| 档位 | 跨柱兴奋性 | 跨柱抑制性 | 生物学依据 | 目的 |
|------|-----------|-----------|-----------|------|
| Baseline (已测) | [0.5, 1.0] | [-1.0, -0.5] | 原始（非生物合理） | 对照（已知 col_ratio=1.17） |
| **Weak (本 spec 实施)** | **[0.1, 0.3]** | **[-0.3, -0.1]** | 弱跨柱连接（Braitenberg & Schüz 1998） | 主要假设验证 |
| VeryWeak (条件触发) | [0.05, 0.15] | [-0.15, -0.05] | 极弱跨柱连接边界 | 边界探索 |

**停止规则**：Weak 档测试后：
- 若 col_ratio > 1.5 → 成功，不继续 VeryWeak
- 若 1.17 < col_ratio ≤ 1.5 → 部分改善，测试 VeryWeak
- 若 col_ratio ≤ 1.17 → 无改善，停止搜索权重，转向假设审查（可能需检查 col_ratio 指标定义或竞争机制）

## Impact
- Affected specs:
  - `boost-activity-and-column-ratio/spec.md` — 本 spec 是其后续，保留其全部参数调整（A 的解决方案），仅补充 B 的架构修复
  - `fix-architectural-issues/spec.md` — 不修改
  - `docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md` — 不修改设计文档
- Affected code:
  - `src/stage2e/config.h` — 新增 4 个宏定义
  - `src/stage2e/network_init.cu` — 修改跨柱突触权重初始化（1 处，行 392-397）
- **BREAKING**: 跨柱突触权重数值范围变化（[0.5, 1.0]→[0.1, 0.3]），现有检查点不兼容，需重新初始化网络
- **风险**: 跨柱突触约占总突触 15%（30/195），权重降低 3 倍预计使总突触驱动下降约 11%，spike/step 可能从 127.5 降至 ~110-115，仍处于 [50, 200] 目标范围内（A 不回归）

## ADDED Requirements

### Requirement: 跨柱突触权重参数化与大幅降低
系统 SHALL 将跨柱突触权重范围从 `[0.5, 1.0] * w_scale`（兴奋性）/ `[-1.0, -0.5] * w_scale`（抑制性）降低至 `[0.1, 0.3] * w_scale` / `[-0.3, -0.1] * w_scale`，约 3 倍降低。权重范围 SHALL 通过 `config.h` 宏参数化（`CROSS_COL_W_EXC_MIN/MAX`、`CROSS_COL_W_INH_MIN/MAX`），便于后续调参。柱内突触权重保持 `[0.4, 1.0]` / `[-1.0, -0.4]` 不变，确保柱内处理能力不受影响。

生物学依据：真实皮层柱内连接强（0.5-1.0 倍最大值）、跨柱长程连接弱（0.1-0.3 倍），这是小世界拓扑的核心特征（Braitenberg & Schüz 1998）。跨柱连接主要用于区间协调，而非初级驱动。

#### Scenario: 跨柱权重范围符合新值
- **WHEN** 网络初始化完成
- **THEN** 跨柱兴奋性突触权重 |w| ∈ [0.1×w_scale, 0.3×w_scale] ≈ [0.014, 0.042]
- **AND** 跨柱抑制性突触权重 |w| ∈ [0.1×w_scale, 0.3×w_scale] ≈ [0.014, 0.042]
- **AND** 柱内突触权重范围不变（|w| ∈ [0.057, 0.143]）
- **AND** 跨柱/柱内权重比 < 0.35（原 ≈ 1.0，目标 < 0.35）

#### Scenario: 柱间分化显著提升
- **WHEN** 运行 10K 步烟雾测试
- **THEN** col_ratio > 2.0（当前 1.13，目标提升约 2 倍）
- **AND** `[10d] 平衡态网络验证` PASS（CV > 0.5 且 col_ratio > 1.5）
- **AND** `[21] P3-C 语义聚类评估` 中 col_ratio > 2.0
- **AND** 偏好柱响应占总响应 > 70%

### Requirement: 活动水平不回归（A 保持）
系统 SHALL 在降低跨柱权重后维持 spike/step ∈ [50, 200]。跨柱突触仅占总突触约 15%，权重降低 3 倍预计使总驱动下降约 11%，预期 spike/step 从 127.5 降至 ~110-115，仍在目标范围内。若实测 spike/step < 50，SHALL 通过微调 `POP_CODING_GAIN`（+10~20）补偿，但不回退跨柱权重降低。

#### Scenario: 活动维持
- **WHEN** 应用跨柱权重降低后运行 10K 步
- **THEN** spike/step ∈ [50, 200]（预期 ~110-115，当前基线 127.5）
- **AND** `[7] 发放活动正常 (avg > 10)` PASS
- **AND** `[5] spike count 极差 > 100` PASS
- **AND** 不出现爆发（spike/step < 1000）

## MODIFIED Requirements

### Requirement: 柱间分化（更新自 boost-activity-and-column-ratio）
原 spec 通过提升 `COLUMN_BYTE_PREF_GAIN_IN` 2.0→3.0 试图改善 col_ratio，实测无效（1.12），证明输入端调整无法解决突触传播端的对称性破缺问题。本 spec 从突触传播端入手，降低跨柱突触权重 3 倍，使偏好柱的强响应不再通过跨柱兴奋传播淹没其他柱，从而恢复柱间分化能力。

#### Scenario: 跨柱传播衰减
- **WHEN** 偏好柱接收到强输入并产生强响应
- **THEN** 该响应通过跨柱突触传播到其他柱时衰减约 3 倍
- **AND** 非偏好柱接收到的跨柱驱动远低于其自身柱内驱动
- **AND** 各柱保持各自字节偏好（由输入端 GAIN_IN/GAIN_OUT = 66:1 决定）

## REMOVED Requirements
（无删除项）
