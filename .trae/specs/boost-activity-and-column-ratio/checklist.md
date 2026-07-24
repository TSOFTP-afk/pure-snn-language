# Checklist

## Phase 1: 参数配置验证
- [x] config.h 中 `POP_CODING_GAIN` 从 30.0f 改为 80.0f
- [x] config.h 中 `POP_CODING_K_PER_COLUMN` 从 50 改为 100
- [x] config.h 中 `COLUMN_BYTE_PREF_GAIN_OUT` 从 0.1f 改为 0.03f
- [x] config.h 中 `INPUT_INJECT_INTERVAL` 从 5 改为 3
- [x] config.h 中新增 `BALANCED_WEIGHT_SCALE_DIVISOR = 4`（注释说明有效入度概念）
- [x] `STDP_W_MAX_2E` 保持原值不变（PSW 机制不失效）
- [x] `COLUMN_BYTE_PREF_GAIN_IN` 调参后为 3.0f（Task 6.1 回退尝试，最终保留 2.0f 或 3.0f 均无效，待新 spec）
- [x] `CA_REBOUND_THRESHOLD` 保持 0.15f 不变

## Phase 1: 权重缩放放宽验证
- [x] network_init.cu 中 w_scale 计算从 `1.0f / sqrtf(SYNAPSES_PER_NEURON_2E)` 改为 `2.0f / sqrtf(SYNAPSES_PER_NEURON_2E)`
- [x] 注释更新：缩放因子从 ≈0.0707 → ≈0.1414（约 2 倍）
- [x] 8 处 `* w_scale` 引用无需改动（自动生效）
- [x] 权重范围预期：[0.057, 0.143]（实测 p25=0.0608, p75=0.1181 ✓）

## Phase 1: input_encoding.cu 兼容性验证
- [x] `POP_CODING_K_PER_COLUMN=100` ≤ `COL_SENSORY_SIZE_2E=200` ✓
- [x] kernel 循环 `for (int k = 0; k < POP_CODING_K_PER_COLUMN; ++k)` 自动适配
- [x] `INPUT_INJECT_INTERVAL=3` 在 main.cpp 行 677 `step % INPUT_INJECT_INTERVAL == 0` 自动生效

## Phase 1: 构建验证
- [x] `build_p1.ps1` 执行成功，ninja exit: 0
- [x] 无编译错误
- [x] 无新增警告

## Phase 1: 10K 烟雾测试验证 — 活动稀疏解决 ✅
- [x] `[7] 发放活动正常 (avg > 10)` PASS (127.5, 远超 10)
- [x] spike/step ∈ [50, 200]（实测 127.5，目标达成 ✅）
- [x] `[5] spike count 极差 > 100` PASS (388，当前 60→388)
- [x] mean_fr > 0.002（实测 0.0163，提升 23 倍 ✅）
- [x] 不出现爆发（spike/step=127.5 < 1000 ✅）
- [x] PSW mature_ratio 提升（1.36% → 5.0%，提升 3.7 倍 ✅）
- [x] `[12] eligibility trace 非零` 从 FAIL → PASS（e1_nz=197，原为 0）
- [x] 卡方显著神经元从 2046 → 9491（4.6 倍提升 ✅）

## Phase 1: 10K 烟雾测试验证 — 柱间分化提升 ❌
- [ ] `[10d] 平衡态网络验证` 通过：CV > 0.5 且 col_ratio > 1.5
   - CV=3.38 ✓（>0.5）
   - col_ratio=1.13 ✗（< 1.5，反而从 1.33 下降）
- [ ] col_ratio > 2.0（实测 1.13，目标未达成）
- [ ] `[21] P3-C 语义聚类评估` 中 col_ratio > 2.0（实测 1.13 ✗）
- [x] 偏好柱与非偏好柱输入比 ≥ 66:1（GAIN_IN/GAIN_OUT = 2.0/0.03 = 66:1，或 3.0/0.03=100:1）

## Phase 1: 回归验证（不破坏已有机制）✅
- [x] `[10b] PSW 概率突触权重` PASS (mature=5.0%)
- [x] `[10c] Ca²⁺ 回弹 LTD` PASS (high_ratio=0.072%，max_ca=0.479)
- [x] `[17] P3 稀疏竞争机制运行` PASS
- [x] `[18] P3-b k-WTA 柱内竞争` PASS (active_cols=18, suppressed=130)
- [x] `[16] 字节/输入响应卡方 > 500` PASS (9491 显著)

## Phase 1: 结论判据
- [x] 若 spike/step ∈ [50, 200] 且 col_ratio > 2.0：本 spec 成功
   - **部分成功**：spike/step=127.5 ✅，col_ratio=1.13 ❌
- [x] 若 spike/step > 200：执行 Task 6.2 回退 GAIN — **未触发**（127.5 ∈ [50,200]）
- [x] 若 spike/step < 50：执行 Task 6.2 提升 GAIN — **未触发**
- [x] 若 col_ratio 仍 < 2.0：执行 Task 6.1 进一步调整 GAIN_OUT/GAIN_IN
   - 已执行：GAIN_IN 2.0→3.0，col_ratio 仍 1.12，**确认无效**
- [x] 若 silhouette 无改善（仍 ≈ 0）：记录数据，本 spec 范围内不解决，建议新 spec
   - silhouette=-0.0623（更负），**确认需新 spec 处理跨柱突触权重**

## 最终结论
### ✅ 已达成（A 活动稀疏）
本 spec 成功解决网络活动稀疏问题：
- spike/step: 24.8 → 127.5（5.1 倍提升，进入 [50,200] 目标范围）
- 极差: 60 → 388（6.5 倍）
- PSW mature: 1.36% → 5.0%（3.7 倍）
- 卡方显著神经元: 2046 → 9491（4.6 倍）
- eligibility [12]: FAIL → PASS
- mean_fr: 0.0007 → 0.0163（23 倍）

### ❌ 未达成（B 柱间分化）
col_ratio 反而从 1.33 降至 1.13，根因是**跨柱兴奋性突触传播**：
- 跨柱突触权重 `randf(0.5f, 1.0f) * w_scale` 与柱内 `randf(0.4f, 1.0f) * w_scale` 几乎一样强
- 活动提升后，偏好柱响应通过跨柱兴奋传播到所有柱，柱间差异被抹平
- Task 6.1 尝试 GAIN_IN 2.0→3.0（输入比 66:1→100:1）仍无效，col_ratio=1.12

### 📋 后续工作（需新 spec）
需新 spec 降低跨柱突触权重，让柱内处理强、跨柱传播弱：
- 跨柱兴奋性: `randf(0.5f, 1.0f) * w_scale` → `randf(0.1f, 0.3f) * w_scale`（降低约 3 倍）
- 跨柱抑制性: `randf(-1.0f, -0.5f) * w_scale` → `randf(-0.3f, -0.1f) * w_scale`（同步降低）
- 这符合生物学"柱内强连接、跨柱弱连接"的真实皮层拓扑
