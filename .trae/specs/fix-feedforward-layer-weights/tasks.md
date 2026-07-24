# Tasks

- [x] Task 1: 在 config.h 中新增前馈权重范围宏
  - [x] SubTask 1.1: 在 [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 第 71-79 行新增 `FEEDFORWARD_W_EXC_MIN 2.5f`、`FEEDFORWARD_W_EXC_MAX 3.5f`、`FEEDFORWARD_W_INH_MIN -3.5f`、`FEEDFORWARD_W_INH_MAX -2.5f` 四个宏
- [x] Task 2: 修改 network_init.cu 柱内突触权重生成逻辑
  - [x] SubTask 2.1: 修改主循环（第 364-394 行），在 `target_layer = pick_target_layer(pre_layer)` 后判断 `is_feedforward`，根据判断结果选择权重范围
  - [x] SubTask 2.2: 修改补足循环（第 441-466 行），同样判断 `is_feedforward` 并选择权重范围
  - [x] SubTask 2.3: 确认跨柱突触（第 396-413 行）和前额叶投射（第 416-438 行）的权重未受影响
- [x] Task 3: 重新构建并跑 100K 步测试
  - [x] SubTask 3.1: 运行 `build_p1.ps1` 重新构建，确认无编译错误
  - [x] SubTask 3.2: 启动 100K 步训练，在 step 12000 停止（L2/3 仍不活跃，确认 STP 抑郁是更深瓶颈）
- [x] Task 4: 验证层级联激活
  - [x] SubTask 4.1: 检查 step 10000 的 semantic eval — **L2/3/L5/L6 均为 0，前馈权重提升不足以克服 STP 抑郁**
  - [x] SubTask 4.2: 检查 `layer_delay[0..3]` — L4=0.04（正常），L2/3/L5/L6=-1（未激活）
  - [x] SubTask 4.3: 检查网络总活动 — 注入步 spikes=255-288（正常），非注入步 0-53
  - [x] SubTask 4.4: 检查 L4 chi2 显著性 — 100% 显著，chi2_mean=3175（保持健康）

## 验证结论

**前馈权重修复方向正确但不足以解决问题**：L4 层保持健康（100% chi2 显著），证明权重修改未破坏 L4 学习。但 L2/3 仍然完全不活跃，根因是 STP 抑郁将有效电流削弱约 200 倍（resource 稳态≈0.005），前馈权重提升 3.5 倍无法补偿。

**下一步需要新 spec 修复 STP 抑郁**：为前馈连接减弱 STP 抑郁效应（增大 τ_rec、减小 U_SE、或使用易化型 STP）。

# Task Dependencies

- Task 2 依赖 Task 1（先定义宏再使用）
- Task 3 依赖 Task 2（先改代码再构建）
- Task 4 依赖 Task 3（先跑测试再验证）
