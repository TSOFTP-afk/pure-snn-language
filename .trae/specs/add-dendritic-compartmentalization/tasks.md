# Tasks

- [ ] Task 1: 在 config.h 中新增前馈连接专用 Ca²⁺ 动力学参数
  - [ ] SubTask 1.1: 在 [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 第 121 行 `NMDA_CA_TAU` 附近新增 `NMDA_CA_TAU_FEEDFORWARD 10.0f` 和 `CA_MAX_FEEDFORWARD 0.12f`
- [ ] Task 2: 修改 synapse_nmda_kernel 实现树突区室化
  - [ ] SubTask 2.1: 修改 [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `synapse_nmda_kernel`（第 76-87 行），根据 `receptor_flags & RECEPTOR_FLAG_FEEDFORWARD` 选择不同的 ca_decay 和 ca_max
- [ ] Task 3: 重新构建并跑 100K 步测试
  - [ ] SubTask 3.1: 运行 `build_p1.ps1` 重新构建，确认无编译错误
  - [ ] SubTask 3.2: 启动 100K 步训练，监控 L5/L6 的 l6_spikes 和 chi2_mean 变化
- [ ] Task 4: 验证 L5/L6 在 SYNAPTOGENIC 阶段后持续发放
  - [ ] SubTask 4.1: 检查 step 6000+ 的 l6_spikes 是否 > 0（修复前从 step 6000 起为 0）
  - [ ] SubTask 4.2: 检查 step 10000 的 L5/L6 chi2_mean 是否比修复前增长
  - [ ] SubTask 4.3: 检查 step 50000/100000 的 L5/L6 chi2_mean 是否持续增长（非停滞）
  - [ ] SubTask 4.4: 检查 L4/L2/3 chi2 增长趋势保持
  - [ ] SubTask 4.5: 检查网络活动 spikes/step 维持在 [50, 200] 区间

# Task Dependencies

- Task 2 依赖 Task 1（需要前馈专用 Ca²⁺ 参数宏）
- Task 3 依赖 Task 1-2（代码修改完成）
- Task 4 依赖 Task 3（先跑测试再验证）
