# 修复前馈层级突触权重不足 Spec

## Why

Ca²⁺ 回弹 LTD 修复后（见 `fix-calcium-rebound-ltd`），L4 层显著改善（100% chi2 显著，chi2_mean 从 3197→6393 翻倍），但 L2/3/L5/L6 仍然完全不活跃（20K 步内 0 个活跃神经元）。

根因是 L4→L2/3 前馈突触初始权重不足以驱动 L2/3 发放，形成"鸡生蛋"困境：
- 当前权重 = `[0.4, 1.0] × w_scale = [0.057, 0.141]`（`w_scale = 2/√200 ≈ 0.141`）
- 每个 L2/3 神经元接收 ~43 个活跃 L4 突触（偏好柱 100 个 L4 发放 × 150 出度 / 350 L2/3）
- 稳态 synaptic_current ≈ `0.588 × 43 × weight × resource = 12.6 × weight`（resource≈0.5）
- 稳态 dV ≈ `12.6 × weight / 9.37 = 1.35 × weight`
- 3 步累积 V ≈ `3.63 × weight`（注入步 + 2 衰减步）
- 要 V ≥ 1.0 → **weight ≥ 0.275**
- 当前中位数 weight ≈ 0.10，3 步累积 V ≈ 0.36 << 阈值 1.0

L2/3 无法发放 → STDP LTP 无法触发（需 `post_spike`）→ 突触无法加强 → 永久静默。级联导致 L5（依赖 L2/3）和 L6（依赖 L5）同样静默。

## What Changes

- **新增** [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 中前馈层级突触权重范围宏 `FEEDFORWARD_W_EXC_MIN/MAX` 和 `FEEDFORWARD_W_INH_MIN/MAX`
- **修改** [network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu) 柱内突触生成（第 364-387 行主循环）和剩余突触补足（第 435-452 行），对前馈连接（L4→L2/3, L2/3→L5, L5→L6）使用更强的权重范围 `[2.5, 3.5] × w_scale = [0.354, 0.495]`
- **不动** 跨柱突触（L2/3→L2/3）、前额叶投射、L5→前额叶、反馈连接（L5→L2/3, L6→L4）、横向连接（L2/3→L2/3, L6→L6）的权重范围
- **不动** PSW/STDP 参数（权重仍由 PSW 动态调整，仅修改初始值范围）

## Impact

- Affected specs: `fix-calcium-rebound-ltd`（前置修复，已完成）
- Affected code: [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)（新增宏）、[network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu)（柱内突触权重生成逻辑）
- Affected behaviors: L4→L2/3、L2/3→L5、L5→L6 前馈权重提升约 3.5 倍，打破"鸡生蛋"困境，使深层能够发放

## ADDED Requirements

### Requirement: 前馈层级突触权重范围

系统 SHALL 为皮层前馈层级连接（L4→L2/3, L2/3→L5, L5→L6）使用比横向/反馈连接更强的初始权重范围，确保信号能从 L4 逐级传播到 L6。

前馈连接定义（基于 `pick_target_layer` 流向规则）：
- `pre_layer == REGION_L4 && target_layer == REGION_L23`
- `pre_layer == REGION_L23 && target_layer == REGION_L5`
- `pre_layer == REGION_L5 && target_layer == REGION_L6`

兴奋性前馈权重范围：`[FEEDFORWARD_W_EXC_MIN, FEEDFORWARD_W_EXC_MAX] × w_scale = [2.5, 3.5] × 0.141 = [0.354, 0.495]`
抑制性前馈权重范围：`[FEEDFORWARD_W_INH_MIN, FEEDFORWARD_W_INH_MAX] × w_scale = [-3.5, -2.5] × 0.141 = [-0.495, -0.354]`

非前馈连接（横向 L2/3→L2/3、反馈 L5→L2/3、L6→L4、跨柱、前额叶投射）保持原范围 `[0.4, 1.0] × w_scale`。

#### Scenario: L4 发放时 L2/3 能被驱动发放
- **WHEN** 偏好柱 L4 以 ~16Hz 发放（每 3 步一次注入，~100 神经元发放）
- **THEN** 每个 L2/3 神经元接收 ~43 个活跃 L4 突触，单突触权重 ≈ 0.42（中位数）
- **AND** 稳态 synaptic_current ≈ `12.6 × 0.42 = 5.3`，3 步累积 V ≈ `3.63 × 0.42 = 1.52 > 阈值 1.0`
- **AND** L2/3 神经元能发放，STDP LTP 触发，突触进一步强化

#### Scenario: L2/3 发放时 L5 能被驱动发放
- **WHEN** L2/3 部分神经元发放（≥10 个/柱）
- **THEN** L5 神经元通过 L2/3→L5 前馈突触接收足够电流（同样使用 [2.5, 3.5] × w_scale 范围）
- **AND** L5 神经元能发放，信号传播到 L6

#### Scenario: 反馈和横向连接不受影响
- **WHEN** L5 神经元发放投射回 L2/3（反馈连接）
- **THEN** L5→L2/3 突触仍使用原范围 `[0.4, 1.0] × w_scale`，不会过度兴奋 L2/3
- **AND** L2/3→L2/3 横向连接仍使用原范围，保持局部竞争

## MODIFIED Requirements

### Requirement: 柱内突触权重生成

柱内突触权重生成逻辑（[network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu) 第 364-387 行主循环和第 435-452 行补足循环）SHALL 根据 `pre_layer` 和 `target_layer` 判断连接类型：

1. **前馈连接**（L4→L2/3, L2/3→L5, L5→L6）：使用 `FEEDFORWARD_W_*` 范围
2. **其他柱内连接**（L2/3→L2/3 横向、L5→L2/3 反馈、L6→L4 反馈、L6→L6 横向）：使用原 `[0.4, 1.0]` / `[-1.0, -0.4]` 范围

权重生成伪代码：
```cpp
int target_layer = pick_target_layer(pre_layer);
bool is_feedforward = (pre_layer == REGION_L4  && target_layer == REGION_L23) ||
                       (pre_layer == REGION_L23 && target_layer == REGION_L5)  ||
                       (pre_layer == REGION_L5  && target_layer == REGION_L6);
float w_exc_min = is_feedforward ? FEEDFORWARD_W_EXC_MIN : 0.4f;
float w_exc_max = is_feedforward ? FEEDFORWARD_W_EXC_MAX : 1.0f;
float w = pre_is_exc ? randf(w_exc_min, w_exc_max) * w_scale
                      : randf(-w_exc_max, -w_exc_min) * w_scale;
```

## REMOVED Requirements

无。
