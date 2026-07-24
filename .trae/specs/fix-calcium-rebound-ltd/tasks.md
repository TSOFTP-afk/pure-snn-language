# Tasks

- [x] Task 1: 移除 `stdp_arrival_pre_kernel` 中的 Ca²⁺ 回弹 LTD 代码块
  - [x] SubTask 1.1: 删除 [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 第 285-292 行的 `if (s.ca_concentration > CA_REBOUND_THRESHOLD) { ... }` 代码块
  - [x] SubTask 1.2: 保留该 kernel 中标准 STDP 的 LTD 分量（`delta_w = -x_post × A_minus`，第 258 行）和 PSW 权重更新逻辑
  - [x] SubTask 1.3: 确认 `stdp_dual_trace_kernel`（第 210-215 行）的回弹 LTD 代码块未受影响（仍含 `post_spike` 条件）
- [x] Task 2: 重新构建 stage2e 并跑 100K 步测试
  - [x] SubTask 2.1: 运行 `build_p1.ps1` 重新构建，确认生成 `snn_stage2e_p1.exe` 无编译错误
  - [x] SubTask 2.2: 启动 100K 步训练，在 step 20000 停止（L2/3 仍不活跃，确认存在第二瓶颈）
- [x] Task 3: 验证 L2/3/L5/L6 层激活与层间级联
  - [x] SubTask 3.1: 检查训练日志中 `layer_active[1..3]`（L2/3/L5/L6）— **均为 0，L2/3/L5/L6 未激活**
  - [x] SubTask 3.2: 检查 `layer_delay[0..3]` — L4 延迟 0.14（正常），L2/3/L5/L6 延迟 -1（未激活）
  - [x] SubTask 3.3: 检查网络总活动 — 注入步 spikes=255-283（正常），非注入步 spikes=0-49
  - [x] SubTask 3.4: L4 chi2_mean 从 3197→6393 翻倍（L4 学习健康），但 L2/3 完全静默

## 验证结论

**Ca²⁺ 回弹 LTD 修复成功**：L4 层显著改善（100% chi2 显著，chi2_mean 翻倍，295 活跃神经元），证明移除双重 LTD 后 L4→L4 自连接和 L4 的字节选择性学习正常工作。

**但暴露了第二瓶颈**：L2/3/L5/L6 仍然完全不活跃。根因是 L4→L2/3 初始突触权重不足以驱动 L2/3 发放（稳态 V≈0.36 << 阈值 1.0），形成"鸡生蛋"困境——L2/3 需要强突触才能发放，但 STDP LTP 需要 L2/3 发放才能加强突触。此问题需要新的 spec 解决。

# Task Dependencies

- Task 2 依赖 Task 1（先改代码再构建）
- Task 3 依赖 Task 2（先跑测试再验证）
