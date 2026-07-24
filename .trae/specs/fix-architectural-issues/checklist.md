# Checklist

## Phase 1: 参数配置验证
- [x] config.h 中新增 `BALANCED_NETWORK_K_AVG = 195`
- [x] config.h 中 `POP_CODING_GAIN` 改为 30.0f（实测 5.0 不足，3 spikes/step）
- [x] config.h 中新增 `COLUMN_BYTE_PREF_RANGE = 256 / N_COLUMNS_2E`（= 5）
- [x] config.h 中 `COLUMN_BYTE_PREF_GAIN_IN = 2.0f` 和 `COLUMN_BYTE_PREF_GAIN_OUT = 0.1f`（实测 0.3 致 col_ratio=1.13，0.1 强化对比）
- [x] config.h 中 `CA_REBOUND_THRESHOLD` 从 0.5f 改为 0.15f
- [x] `STDP_W_MAX_2E` 保持原值不变（PSW 机制不失效）

## Phase 1: 权重缩放实现验证
- [x] network_init.cu 中 init_synapses_host 顶部计算 `w_scale = 1.0f / sqrtf((float)SYNAPSES_PER_NEURON_2E)`
- [x] 柱内突触（n_intra 循环）：兴奋性 `randf(0.4f, 1.0f) * w_scale`，抑制性 `randf(-1.0f, -0.4f) * w_scale`
- [x] 跨柱突触（n_inter 循环）：同上缩放
- [x] 前额叶投射突触（n_pf 循环）：同上缩放
- [x] 前额叶自反馈突触（is_pf 分支）：同上缩放
- [x] Grep 验证：8 处 `* w_scale` 应用（已确认 9 行匹配）
- [x] PSW 初始化保持 `w_ratio = fabsf(s.weight) / STDP_W_MAX_2E`

## Phase 1: 柱特异性输入实现验证
- [x] input_encoding.cu 中 `input_inject_kernel` 新增参数 `byte_pref_range, gain_in, gain_out`
- [x] kernel 内计算柱 c 的偏好范围 `[c * byte_pref_range, (c+1) * byte_pref_range)`
- [x] 判断 byte 是否在偏好范围，选择 `gain = in_range ? gain_in : gain_out`
- [x] atomicAdd 使用 `POP_CODING_GAIN * gain` 而非 `POP_CODING_GAIN`
- [x] `launch_input_inject` 传入新参数

## Phase 1: 验证指标实现验证
- [x] main.cpp 新增 `sample_balance_state_stats` 函数（计算 CV = std/mean of firing rates）
- [x] 判据 `[10d] 平衡态网络验证`：CV > 0.5 且 col_ratio > 1.5
- [x] FINAL_METRIC 输出 `balance_cv`、`balance_mean_fr`、`balance_std_fr`、`criterion_balance`

## Phase 1: [10d] 判据读取时机修复（Task 7）
- [x] main.cpp 中在 [10d] 判据代码块之前插入 `scheduler.run_semantic_eval(total_steps)` 调用
- [x] 验证 P3 判据区原 `run_semantic_eval` 调用是否需要移除/保留（幂等性分析，已移除避免重复计算）
- [x] 程序执行到 [10d] 时 `scheduler.p3_column_ratio()` 返回值不为 0（实测 1.3328，原为 0）

## Phase 1: 构建验证
- [x] `build_p1.ps1` 执行成功，ninja exit: 0
- [x] 无编译错误
- [x] 无新增警告

## Phase 1: 10K 烟雾测试验证
- [x] 所有原判据仍通过（PSW PASS mature=1.36%、Ca²⁺ 回弹 PASS high_ratio=0.036%、k-WTA PASS、卡方 PASS）
- [ ] `[10d] 平衡态网络验证` 通过：CV > 0.5 且 col_ratio > 1.5
   - CV=4.9162 ✓（远超 0.5，但 mean_fr=0.0007 极低，CV 高主要因稀疏活动）
   - col_ratio=1.3328 ✗（< 1.5，柱间分化仍不足，需进一步调参 — 超出本 spec 范围）
- [ ] 平均每步 spike 数 ∈ [50, 200]（实测 24.8，低于 50，活动过稀疏）
- [ ] col_ratio > 1.5（实测 1.33，需进一步调参）
- [x] Ca²⁺ 回弹 LTD 机制已触发：high_ca_ratio=0.036%（< 0.1% 子判据阈值，但机制本身已运行，36 个突触 ca>0.15）

## Phase 1: 结论
- ✅ Task 7 核心目标达成：[10d] 判据读取时机 bug 已修复，col_ratio 从 0 变为 1.33
- ⚠️ [10d] 判据仍 FAIL，但原因是 col_ratio 本身数值不够（1.33 < 1.5），非时机 bug
- ⚠️ 网络活动稀疏（24.8 spikes/step，mean_fr=0.0007），需进一步调整输入增益或权重缩放
- 📋 后续调参（提升 col_ratio 至 >1.5、提升活动水平至 >50 spikes/step）超出本 spec 范围，建议在新 spec 中处理
