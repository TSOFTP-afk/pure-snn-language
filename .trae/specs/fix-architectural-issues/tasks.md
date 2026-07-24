# Tasks

## Phase 1: 最小改动验证（高 ROI 修复）

- [x] Task 1: 在 config.h 中新增架构修复参数
  - [x] SubTask 1.1: 新增 `BALANCED_NETWORK_K_AVG`（平均入度 195）常量
  - [x] SubTask 1.2: 调整 `POP_CODING_GAIN` 从 2.0f → 30.0f（实测 5.0 不足以维持活动，3 spikes/step；30.0 达 132 spikes/step）
  - [x] SubTask 1.3: 新增柱特异性参数 `COLUMN_BYTE_PREF_RANGE = 256 / N_COLUMNS_2E`（每柱偏好字节数 = 5）、`COLUMN_BYTE_PREF_GAIN_IN = 2.0f`（偏好柱增益倍数）、`COLUMN_BYTE_PREF_GAIN_OUT = 0.1f`（非偏好柱增益倍数，实测 0.3 过强致 col_ratio 1.13，0.1 强化柱间对比）
  - [x] SubTask 1.4: 调整 `CA_REBOUND_THRESHOLD` 从 0.5f → 0.15f
  - [x] SubTask 1.5: `STDP_W_MAX_2E` 保持不变（PSW 机制不变），仅在 network_init 中对初始权重应用缩放

- [x] Task 2: 在 network_init.cu 中实现 1/√K 权重缩放
  - [x] SubTask 2.1: 在 init_synapses_host 函数顶部计算缩放因子 `const float w_scale = 1.0f / sqrtf((float)SYNAPSES_PER_NEURON_2E);`
  - [x] SubTask 2.2: 在 4 处突触权重初始化位置（柱内/跨柱/前额叶自反馈/前额叶接收）将 `s.weight = randf(0.4f, 1.0f)` 改为 `s.weight = randf(0.4f, 1.0f) * w_scale`
  - [x] SubTask 2.3: 同样对抑制性权重 `s.weight = randf(-1.0f, -0.4f) * w_scale`
  - [x] SubTask 2.4: PSW 的 alpha/beta 分配保持 `w_ratio = fabsf(s.weight) / STDP_W_MAX_2E`（W_MAX 不变，PSW 机制自适应）
  - [x] SubTask 2.5: 验证：所有 4 处位置都被修改（已 Grep 确认 9 行 w_scale 引用）

- [x] Task 3: 在 input_encoding.cu 中实现柱特异性字节偏好
  - [x] SubTask 3.1: 修改 `input_inject_kernel` 签名，新增参数 `int byte_pref_range, float gain_in, float gain_out`
  - [x] SubTask 3.2: 在 kernel 内计算柱 c 的字节偏好范围 `[c * byte_pref_range, (c+1) * byte_pref_range)`
  - [x] SubTask 3.3: 判断当前 byte 是否在偏好范围内，选择 `gain = (in_range) ? gain_in : gain_out`
  - [x] SubTask 3.4: 将 `atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN)` 改为 `atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN * gain)`
  - [x] SubTask 3.5: 修改 `launch_input_inject` 调用，传入新增参数（从 config.h 读取）
  - [x] SubTask 3.6: 验证：不同柱对同一字节的响应应有明显差异

- [x] Task 4: 在 main.cpp 中新增架构修复验证指标
  - [x] SubTask 4.1: 新增 `sample_balance_state_stats` 函数，计算 CV = std/mean of firing rates
  - [x] SubTask 4.2: 新增 `[10d] 平衡态网络验证` 判据
  - [x] SubTask 4.3: 在 FINAL_METRIC 中输出 `balance_cv`、`balance_mean_fr`、`balance_std_fr`、`criterion_balance`

- [x] Task 5: 构建验证（已完成）

- [x] Task 7: 修复 [10d] 判据读取 col_ratio 时机问题（CRITICAL）
  - [x] SubTask 7.1: 在 main.cpp 的 [10d] 判据代码块之前插入 `scheduler.run_semantic_eval(total_steps)` 调用，触发 P3-C 语义聚类评估并更新 p3_column_ratio_
  - [x] SubTask 7.2: 验证 P3 判据区原 `scheduler.run_semantic_eval(total_steps)` 调用是否仍需要保留（run_semantic_eval 完全幂等，已移除原位置调用避免重复计算）
  - [x] SubTask 7.3: 验证：[10d] 判据中 `col_ratio_now` 不再为 0（实测 1.3328）

- [x] Task 8: 重新构建并跑 10K 烟雾测试
  - [x] SubTask 8.1: 运行 build_p1.ps1 重新构建（ninja exit: 0）
  - [x] SubTask 8.2: 运行 `snn_stage2e_p1.exe --steps 10000`（成功完成）
  - [x] SubTask 8.3: 检查 [10d] 平衡态判据：CV=4.9162 ✓（>0.5），col_ratio=1.3328 ✗（<1.5）
  - [x] SubTask 8.4: 关键指标记录：col_ratio=1.33, CV=4.92, mean_fr=0.0007, std_fr=0.0036, spike/step=24.8

# Task Dependencies
- Task 7 依赖 Task 1-5（已有参数和实现）
- Task 8 依赖 Task 7

# Notes
- Task 1-5 已在前序会话中完成实现，本次主要完成 Task 7（时机修复）和 Task 8（验证）
- 本 spec 不包含 Phase 2（预测编码反馈）和 Phase 3（空间拓扑约束）
- PSW 和 Ca²⁺ 回弹 LTD 已实现的机制保持不变，本 spec 仅调整其参数和前置条件
- W_MAX (STDP_W_MAX_2E) 保持不变，避免 PSW 的 alpha/beta 比例计算失效
