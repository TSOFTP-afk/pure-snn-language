# Tasks

## Phase 1: 参数调参（最小改动验证）

- [x] Task 1: 在 config.h 中调整 6 个参数
  - [x] SubTask 1.1: `POP_CODING_GAIN` 从 30.0f → 80.0f（行 143）
  - [x] SubTask 1.2: `POP_CODING_K_PER_COLUMN` 从 50 → 100（行 136）
  - [x] SubTask 1.3: `COLUMN_BYTE_PREF_GAIN_OUT` 从 0.1f → 0.03f（行 153）
  - [x] SubTask 1.4: `INPUT_INJECT_INTERVAL` 从 5 → 3（行 146）
  - [x] SubTask 1.5: 新增 `BALANCED_WEIGHT_SCALE_DIVISOR = 4`（行 57）

- [x] Task 2: 在 network_init.cu 中放宽缩放因子
  - [x] SubTask 2.1: w_scale 从 `1.0f / sqrtf(...)` 改为 `2.0f / sqrtf(...)`（行 172）
  - [x] SubTask 2.2: 注释更新：缩放因子从 ≈0.0707 → ≈0.1414
  - [x] SubTask 2.3: 8 处 `* w_scale` 引用未改动（自动生效）

- [x] Task 3: 验证 input_encoding.cu 兼容性
  - [x] SubTask 3.1: `POP_CODING_K_PER_COLUMN=100` ≤ `COL_SENSORY_SIZE_2E=200` ✓
  - [x] SubTask 3.2: kernel 循环 `for (int k = 0; k < POP_CODING_K_PER_COLUMN; ++k)` 自动适配
  - [x] SubTask 3.3: `INPUT_INJECT_INTERVAL=3` 在 main.cpp 行 677 `step % INPUT_INJECT_INTERVAL == 0` 自动生效

- [x] Task 4: 构建验证
  - [x] SubTask 4.1: 运行 build_p1.ps1，确认无编译错误（ninja exit: 0）
  - [x] SubTask 4.2: 无编译警告或错误

- [x] Task 5: 10K 烟雾测试验证
  - [x] SubTask 5.1: 运行 `snn_stage2e_p1.exe --steps 10000`（GAIN_IN=2.0, GAIN_OUT=0.03）
  - [x] SubTask 5.2: 检查关键判据：
    - `[7] 发放活动正常 (avg > 10)`：✅ PASS (127.5 spikes/step，目标 [50,200] 达成)
    - `[5] spike count 极差 > 100`：✅ PASS (388，当前 60→388)
    - `[10d] 平衡态网络验证`：❌ FAIL (CV=3.38 ✓, col_ratio=1.13 ✗ < 1.5)
    - `[21] P3-C 语义聚类评估`：col_ratio=1.13 ✗ < 2.0
  - [x] SubTask 5.3: 关键指标记录：spike/step=127.5, col_ratio=1.13, CV=3.38, mean_fr=0.0163, PSW mature=5.0%, silhouette=-0.0578
  - [x] SubTask 5.4: spike/step=127.5 ∈ [50,200]，无需回退 GAIN

- [x] Task 6: 失败回退策略（条件执行）
  - [x] SubTask 6.1: col_ratio=1.13 < 2.0，已尝试 GAIN_IN 2.0→3.0（输入比 66:1→100:1），col_ratio 仍 1.12（v2 测试），**确认单纯调输入增益无效**
  - [x] SubTask 6.2: 活动未爆发（spike/step=140.3 < 500），无需回退
  - [x] SubTask 6.3: silhouette=-0.0623 无改善（仍 ≈ 0 甚至更负），**确认问题超出参数层面，需新 spec 处理跨柱突触权重**

## 根因分析（Task 6 执行后得出）
活动提升后 col_ratio 反而下降（1.33→1.13），根因是**跨柱兴奋性突触传播**：
- 跨柱突触权重 `randf(0.5f, 1.0f) * w_scale` 与柱内 `randf(0.4f, 1.0f) * w_scale` 几乎一样强
- 活动提升后，偏好柱的强响应通过跨柱兴奋性突触传播到所有柱
- 所有柱都被"淹没"到相似活动水平，柱间差异被抹平
- **解决方案**：需新 spec 降低跨柱突触权重（如 `randf(0.1f, 0.3f) * w_scale`），让柱内处理强、跨柱传播弱

# Task Dependencies
- Task 2 依赖 Task 1（参数定义）
- Task 3 独立，可与 Task 2 并行
- Task 4 依赖 Task 1, 2, 3
- Task 5 依赖 Task 4
- Task 6 条件执行，依赖 Task 5 结果

# Notes
- 本 spec 仅调参，不引入新机制
- 预期通过 4 项参数协同提升活动 3-5 倍至 [50, 200] 范围
- 若调参后 silhouette 仍停滞，说明问题超出参数层面，需评估是否引入预测编码/拓扑约束等新机制（新 spec）
- PSW、Ca²⁺ 回弹、k-WTA 等已实现机制保持不变
- W_MAX (STDP_W_MAX_2E) 保持不变
