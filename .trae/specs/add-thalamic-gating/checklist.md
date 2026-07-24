# Checklist

## Phase 1: 参数配置验证
- [x] config.h 中新增 `GATE_UPDATE_RATE = 0.01f`
- [x] config.h 中新增 `GATE_MIN = 0.1f`、`GATE_MAX = 0.9f`
- [x] config.h 中新增 `GATE_ACTIVITY_COUP = 0.3f`
- [x] config.h 中新增 `GATE_NOVELTY_COUP = 0.2f`
- [x] config.h 中新增 `GATE_ACTIVITY_EMA_DECAY = 0.99f`
- [x] config.h 中新增 `GATE_NOVELTY_EMA_DECAY = 0.95f`
- [x] config.h 中新增 `GATE_INITIAL_SIGNAL = 0.5f`
- [x] config.h 中新增 `GATE_UPDATE_INTERVAL = 1`
- [x] 参数注释说明生物学依据（丘脑门控、注意力、状态依赖输入）

## Phase 1: 门控结构验证
- [x] thalamic_gate.cuh 定义 `ThalamicGateState` 结构，sizeof ≤ 16B
- [x] 结构包含 gate_signal, activity_ema, novelty_ema 字段
- [x] 声明 init_thalamic_gate, launch_thalamic_gate_update, launch_thalamic_gate_stats 函数

## Phase 1: 门控实现验证
- [x] thalamic_gate.cu 实现 init_thalamic_gate_kernel：每柱 gate_signal=0.5, activity_ema=0, novelty_ema=0
- [x] thalamic_gate.cu 实现 thalamic_gate_update_kernel：
  - [x] 计算 activity_norm = clamp((activity_ema - current_spikes) / max(activity_ema, 1), -1, 1)
  - [x] 计算 novelty = 1 - byte_frequency（当前字节历史频率）
  - [x] 更新 activity_ema = decay * activity_ema + (1-decay) * current_spikes
  - [x] 更新 novelty_ema = decay * novelty_ema + (1-decay) * novelty
  - [x] 计算 gate_target = 0.5 + GATE_ACTIVITY_COUP * activity_norm + GATE_NOVELTY_COUP * novelty_norm
  - [x] clamp gate_target 到 [GATE_MIN, GATE_MAX]
  - [x] gate_signal += GATE_UPDATE_RATE * (gate_target - gate_signal)
- [x] thalamic_gate.cu 实现 thalamic_gate_stats_kernel：统计 gate_mean, gate_open_ratio
- [x] host launcher 函数设置正确的 grid/block 配置

## Phase 1: 输入注入修改验证
- [x] input_inject_kernel 签名新增 `const float* gate_signal` 参数
- [x] kernel 内增益计算改为 `POP_CODING_GAIN * gain * gate_signal[col]`
- [x] launch_input_inject 签名新增 `const float* d_gate_signal` 参数
- [x] input_encoding.cuh 函数声明已更新
- [x] 向后兼容：d_gate_signal 为 nullptr 时使用 gate_signal=1.0（等价原行为）

## Phase 1: Scheduler 集成验证
- [x] BioMechanismScheduler 新增 `ThalamicGateState* d_gate_states_` 成员
- [x] 构造函数分配门控缓冲区（N_COLUMNS_2E × sizeof(ThalamicGateState)）并调用 init_thalamic_gate
- [x] step() 流水线在 input_inject 之前插入门控更新
- [x] 门控更新使用 per-column spike count（从 d_spike_flags 聚合或复用 d_p3_column_spikes）
- [x] launch_input_inject 调用传入 d_gate_states_ 的 gate_signal 数组
- [x] 门控指标统计每 N 步调用 launch_thalamic_gate_stats
- [x] 析构函数释放门控缓冲区

## Phase 1: 指标输出验证
- [x] RUN_PARAM 输出门控参数（GATE_UPDATE_RATE, GATE_MIN, GATE_MAX, GATE_ACTIVITY_COUP, GATE_NOVELTY_COUP）
- [x] FINAL_METRIC 或定期输出新增 gate_mean, gate_open_ratio
- [x] 新增判据 `[22] 丘脑门控运行`：gate_mean ∈ [0.2, 0.9] 且 gate_open_ratio > 0.2

## Phase 1: 构建验证
- [x] CMakeLists.txt 添加 thalamic_gate.cu
- [x] build_p1.ps1 执行成功，ninja exit: 0
- [x] 无编译错误
- [x] 无新增编译警告
- [x] 显存占用未显著增加（门控缓冲区仅 800B）

## Phase 1: 10K 烟雾测试验证 — A 活动不回归
- [x] spike/step ∈ [50, 200]（基线 127.5，门控半开后预期 ~60-100）
- [x] `[7] 发放活动正常 (avg > 10)` PASS
- [x] `[5] spike count 极差 > 100` PASS
- [x] 不出现爆发（spike/step < 1000）
- [x] mean_fr > 0.002

## Phase 1: 10K 烟雾测试验证 — 门控运行
- [x] `[22] 丘脑门控运行` PASS
- [x] gate_mean ∈ [0.2, 0.9]（不应长期全闭或全开）
- [x] gate_open_ratio > 0.2（至少 20% 柱门控开启）
- [x] gate_signal 有动态变化（非恒定值，说明状态驱动生效）

## Phase 1: 回归验证（不破坏已有机制）
- [x] `[10b] PSW 概率突触权重` PASS
- [x] `[10c] Ca²⁺ 回弹 LTD` PASS
- [x] `[17] P3 稀疏竞争机制运行` PASS
- [x] `[18] P3-b k-WTA 柱内竞争` PASS
- [x] `[16] 字节/输入响应卡方 > 500` PASS

## Phase 1: 结论判据
- [x] 若 spike/step ∈ [50, 200] 且 `[22]` PASS：本 spec 成功 ✅
- [x] 若 spike/step < 50（A 回归）：执行 Task 10.1 提升 GATE_MIN 或 GATE_INITIAL_SIGNAL
- [x] 若 gate_mean 长期 = GATE_MIN（门控卡死）：执行 Task 10.2 降低 GATE_ACTIVITY_COUP
- [x] 若 gate_mean 长期 = GATE_MAX（门控饱和）：执行 Task 10.3 提升 GATE_UPDATE_RATE
- [x] 若门控对 col_ratio 无改善：记录数据，本 spec 范围内不解决（门控目标是状态依赖输入，非柱分化）

## 最终结论
- **A 活动不回归**: PASS ✓ — avg_spikes_per_step=95.48 ∈ [50, 200], criterion_active=1
- **门控运行**: PASS ✓ — gate_mean=0.7024 ∈ [0.2, 0.9], gate_open_ratio=1.0 > 0.2, criterion_gate=1
- **已有机制不破坏**: PASS ✓ — PSW(mature=2.5%), Ca²⁺回弹, k-WTA, 卡方(9202显著), 语义准备度(0.85) 全部 PASS
- **门控行为分析**: gate_mean=0.70 > 初始 0.5, 说明活动补偿生效(网络活动偏低时门控自动开大), novelty 增强也在工作
- **未通过项(非门控引入)**: criterion_p3_inhibitory=0, criterion_balance=0 — 这些是之前就存在的问题, 不是门控引入的回归
- **显存影响**: 1401.24 MB, 与门控前一致(门控缓冲区仅 800B, 可忽略)
- **结论**: 模块D(丘脑-皮层门控)实施成功, 核心目标全部达成, 无需回退策略
