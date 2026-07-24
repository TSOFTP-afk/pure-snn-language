- [x] `stdp_arrival_pre_kernel` 中 Ca²⁺ 回弹 LTD 代码块（第 285-292 行）已删除
- [x] `stdp_arrival_pre_kernel` 中标准 STDP LTD 分量（`delta_w = -x_post × A_minus`，第 258 行）保留
- [x] `stdp_dual_trace_kernel` 中 Ca²⁺ 回弹 LTD（第 210-215 行）保留且仍含 `post_spike` 条件
- [x] 其他 STDP/PSW 参数（PSW_ETA_BETA、CA_REBOUND_LTD_GAIN、CA_REBOUND_THRESHOLD）未改动
- [x] `build_p1.ps1` 构建成功，生成 `snn_stage2e_p1.exe` 无编译错误
- [ ] 100K 步训练中 L2/3 层出现非零激活（`layer_active[1] > 0`）— **FAIL: L2/3 仍为 0，存在第二瓶颈（L4→L2/3 突触权重不足）**
- [ ] 100K 步训练中 L5 层出现非零激活（`layer_active[2] > 0`）— **FAIL: 依赖 L2/3，级联失效**
- [ ] 100K 步训练中 L6 层出现非零激活（`layer_active[3] > 0`）— **FAIL: 依赖 L5，级联失效**
- [ ] 层间激活延迟 `layer_delay[0..3]`（L4/L2/3/L5/L6）单调递增 — **FAIL: 仅 L4 有延迟 0.14，其余 -1**
- [x] 网络总活动 `spikes/step` 维持在 [50, 200] 区间（注入步 255-283，非注入步 0-49）
- [x] silhouette、js_mean 指标不再在 40K 步前早期停滞（L4 chi2_mean 持续增长 3197→6393）

## 验证总结

**通过项（5/11）**：代码修改正确，构建成功，L4 层健康（100% chi2 显著，chi2_mean 翻倍），网络活动正常，L4 学习无停滞。

**未通过项（4/11）**：L2/3/L5/L6 层激活相关检查点未通过。根因是 L4→L2/3 初始突触权重不足（稳态 V≈0.36 << 阈值 1.0），这是独立于 Ca²⁺ 回弹 LTD 的第二瓶颈，需要新的 spec 解决。

**部分通过项（2/11）**：层间激活延迟和网络活动部分符合预期（L4 正常），但深层未激活。
