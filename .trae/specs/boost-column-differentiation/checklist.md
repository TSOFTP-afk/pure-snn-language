# Checklist

## Phase 1: 参数配置验证
- [ ] config.h 中新增 `CROSS_COL_W_EXC_MIN = 0.1f`
- [ ] config.h 中新增 `CROSS_COL_W_EXC_MAX = 0.3f`
- [ ] config.h 中新增 `CROSS_COL_W_INH_MIN = -0.3f`
- [ ] config.h 中新增 `CROSS_COL_W_INH_MAX = -0.1f`
- [ ] 宏定义包含注释说明生物学依据（柱内强/跨柱弱）和调参原因（约 3 倍降低）
- [ ] 前序 `boost-activity-and-column-ratio` spec 的参数保持不变（POP_CODING_GAIN=80, K=100, GAIN_OUT=0.03, INTERVAL=3, w_scale=2/√K）

## Phase 1: 跨柱突触权重修改验证
- [ ] network_init.cu 行 392-397 跨柱兴奋性权重改为 `randf(CROSS_COL_W_EXC_MIN, CROSS_COL_W_EXC_MAX) * w_scale`
- [ ] network_init.cu 行 392-397 跨柱抑制性权重改为 `randf(CROSS_COL_W_INH_MIN, CROSS_COL_W_INH_MAX) * w_scale`
- [ ] 跨柱权重注释更新，说明"约 3 倍降低，抑制跨柱兴奋传播"
- [ ] 柱内突触权重（行 322-326）未被改动（仍为 `randf(0.4f, 1.0f)`/`randf(-1.0f, -0.4f)`）
- [ ] 前额叶长程投射权重（行 457-461）未被改动（仍为 `randf(0.6f, 1.2f)`/`randf(-1.2f, -0.6f)`）
- [ ] PSW 初始化逻辑（w_ratio 计算）未被改动，自动适配新权重值

## Phase 1: 构建验证
- [ ] `build_p1.ps1` 执行成功，ninja exit: 0
- [ ] 无编译错误（特别是宏未定义错误）
- [ ] 无新增编译警告

## Phase 1: 10K 烟雾测试验证 — A 活动不回归
- [ ] spike/step ∈ [50, 200]（基线 127.5，预期 ~110-115）
- [ ] `[7] 发放活动正常 (avg > 10)` PASS
- [ ] `[5] spike count 极差 > 100` PASS
- [ ] 不出现爆发（spike/step < 1000）
- [ ] mean_fr > 0.002（基线 0.0163）
- [ ] PSW mature_ratio 不显著下降（基线 5.0%）

## Phase 1: 10K 烟雾测试验证 — B 柱间分化提升
- [ ] col_ratio > 2.0（基线 1.13，目标 > 2.0）
- [ ] `[10d] 平衡态网络验证` PASS（CV > 0.5 且 col_ratio > 1.5）
- [ ] `[21] P3-C 语义聚类评估` 中 col_ratio > 2.0
- [ ] CV > 0.5（基线 3.38，平衡态去相关特性保持）
- [ ] 偏好柱响应占总响应 > 70%

## Phase 1: 回归验证（不破坏已有机制）
- [ ] `[10b] PSW 概率突触权重` PASS（mature_ratio > 1%）
- [ ] `[10c] Ca²⁺ 回弹 LTD` PASS
- [ ] `[17] P3 稀疏竞争机制运行` PASS
- [ ] `[18] P3-b k-WTA 柱内竞争` PASS
- [ ] `[16] 字节/输入响应卡方 > 500` PASS

## Phase 1: 结论判据
- [ ] 若 spike/step ∈ [50, 200] 且 col_ratio > 2.0：本 spec 成功 ✅
- [ ] 若 spike/step < 50（A 回归）：执行 Task 5.1 微调 POP_CODING_GAIN +10~20
- [ ] 若 col_ratio ∈ [1.5, 2.0)（B 部分改善）：执行 Task 5.2 进一步降低跨柱权重
- [ ] 若 col_ratio < 1.5（B 未改善）：记录数据，评估是否需更深层架构调整（新 spec）
- [ ] 若 silhouette 无改善（仍 ≈ 0 或更负）：记录数据，本 spec 范围内不解决，建议后续 spec

## 最终结论
（待 10K 烟雾测试后填写）
