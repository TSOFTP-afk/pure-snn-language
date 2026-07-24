# Tasks

- [x] Task 1: 在 config.h 中新增前馈连接专用 PSW 学习率宏
  - [x] SubTask 1.1: 在 [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 第 262-267 行新增 `PSW_ETA_ALPHA_FEEDFORWARD 2.0f` 和 `PSW_ETA_BETA_FEEDFORWARD 2.0f` 两个宏
- [x] Task 2: 修改 stdp_dual_trace_kernel 使用前馈专用 PSW 学习率
  - [x] SubTask 2.1: 修改 [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `stdp_dual_trace_kernel`（第 204-228 行），根据前馈标志选择学习率，Ca²⁺ 回弹 LTD 也用前馈专用 eta_beta
- [x] Task 3: 修改 stdp_arrival_pre_kernel 使用前馈专用 PSW 学习率
  - [x] SubTask 3.1: 修改 [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `stdp_arrival_pre_kernel`（第 288-301 行），同样根据前馈标志选择学习率
- [x] Task 4: 增强 structural_plasticity 衰减因子
  - [x] SubTask 4.1: 修改 [scheduler.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu) 第 1052 行的 `decay_factor` 从 0.999f 改为 0.95f
- [ ] Task 5: 重新构建并跑 100K 步测试
  - [ ] SubTask 5.1: 运行 `build_p1.ps1` 重新构建，确认无编译错误
  - [ ] SubTask 5.2: 启动 100K 步训练，监控 L5/L6 chi2_mean 变化
- [ ] Task 6: 验证 L5/L6 chi2 持续增长
  - [ ] SubTask 6.1: 检查 step 10000 的 L5 chi2_mean（基线值）
  - [ ] SubTask 6.2: 检查 step 50000 的 L5/L6 chi2_mean 是否比 step 10000 增长（非停滞）
  - [ ] SubTask 6.3: 检查 step 100000 的 L5/L6 chi2_mean 是否持续增长
  - [ ] SubTask 6.4: 检查 L4/L2/3 chi2 增长趋势保持（不应因前馈学习率降低而受损）
  - [ ] SubTask 6.5: 检查网络活动 `spikes/step` 维持在 [50, 200] 区间

# Task Dependencies

- Task 2, 3 依赖 Task 1（需要前馈专用 PSW 学习率宏）
- Task 4 独立（与 Task 1-3 并行）
- Task 5 依赖 Task 1-4（全部代码修改完成）
- Task 6 依赖 Task 5（先跑测试再验证）
