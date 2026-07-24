- [x] config.h 中新增 `PSW_ETA_ALPHA_FEEDFORWARD 2.0f` 和 `PSW_ETA_BETA_FEEDFORWARD 2.0f` 两个宏
- [x] `stdp_dual_trace_kernel` 根据 `receptor_flags & RECEPTOR_FLAG_FEEDFORWARD` 选择前馈专用 PSW 学习率
- [x] `stdp_arrival_pre_kernel` 同样根据前馈标志选择学习率
- [x] `stdp_dual_trace_kernel` 的 Ca²⁺ 回弹 LTD 也使用前馈专用 `PSW_ETA_BETA_FEEDFORWARD`
- [x] scheduler.cu 的 `structural_plasticity` 衰减因子从 0.999f 改为 0.95f
- [x] 非前馈连接的 PSW 学习率保持不变（仍用 `PSW_ETA_ALPHA/BETA=20.0`）
- [x] `build_p1.ps1` 构建成功，无编译错误
- [x] step 10000 的 L5 chi2_mean 记录为基线值 — **基线: L5=767.71, L6=599.08**
- [ ] step 50000 的 L5 chi2_mean 比 step 10000 增长 — **FAIL: step 20000 L5=767.71 不变**
- [ ] step 50000 的 L6 chi2_mean 比 step 10000 增长 — **FAIL: step 20000 L6=599.08 不变**
- [ ] step 100000 的 L5/L6 chi2_mean 持续增长 — **FAIL: 训练在 27000 停止, 趋势已明确停滞**
- [x] L4/L2/3 chi2 增长趋势保持 — **PASS: L4 3150→6326, L2/3 608→709 (持续增长)**
- [x] 网络总活动 `spikes/step` 维持在 [50, 200] 区间 — **PASS: 注入步 266-315, 非注入步 34-82**

## 验证总结

**修复失败**：L5/L6 chi2_mean 在 step 10000-20000 内完全停滞（L5=767.71 不变, L6=599.08 不变），与前馈学习率降低和衰减增强前的表现完全相同。

**关键发现**：PSW 饱和**不是** L5/L6 chi2 停滞的根因。即使学习率降到 1/10、衰减增强到 5%，L5/L6 的 chi2 仍然停滞。

**重新诊断方向**：
1. L5/L6 神经元在非注入步完全不发放（l6_spikes=0），chi2 统计仅依赖注入步的 spike
2. 注入步的 L5/L6 spike 模式可能在早期就固化（不是权重饱和，而是输入模式固化）
3. 需要检查 L5/L6 的 spike 是否由 L4→L2/3→L5→L6 的级联驱动，还是由其他机制（如 residual input_current）驱动
4. 可能需要检查 neuron_byte_counts 的累积方式——L5/L6 在注入步的 spike 是否真的在变化
