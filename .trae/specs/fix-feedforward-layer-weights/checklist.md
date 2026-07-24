- [x] config.h 中新增 `FEEDFORWARD_W_EXC_MIN 2.5f`、`FEEDFORWARD_W_EXC_MAX 3.5f`、`FEEDFORWARD_W_INH_MIN -3.5f`、`FEEDFORWARD_W_INH_MAX -2.5f` 四个宏
- [x] network_init.cu 主循环（第 364-394 行）根据 `is_feedforward` 判断选择前馈权重范围
- [x] network_init.cu 补足循环（第 441-466 行）根据 `is_feedforward` 判断选择前馈权重范围
- [x] 跨柱突触（第 396-413 行）权重范围未改动（仍用 `CROSS_COL_W_*`）
- [x] 前额叶投射（第 416-438 行）权重范围未改动（仍用 `[0.6, 1.2]`）
- [x] `build_p1.ps1` 构建成功，无编译错误
- [ ] step 10000 的 semantic eval 中 L2/3 层 `layer_active[1] > 0` — **FAIL: L2/3 仍为 0**
- [ ] step 10000 的 semantic eval 中 L5 层 `layer_active[2] > 0` — **FAIL: L5 仍为 0**
- [ ] step 10000 的 semantic eval 中 L6 层 `layer_active[3] > 0` — **FAIL: L6 仍为 0**
- [ ] `layer_delay[0..3]` 单调递增 — **FAIL: 仅 L4=0.04，L2/3/L5/L6=-1**
- [x] 网络总活动 `spikes/step` 维持在 [50, 200] 区间（注入步 255-288，非注入步 0-53）
- [x] L4 chi2 显著性保持（100% 显著，chi2_mean=3175，与前馈权重提升前相当）

## 验证总结

**通过项（7/11）**：代码修改正确，构建成功，L4 层健康（100% chi2 显著，chi2_mean=3175），网络活动正常，跨柱/前额叶权重未受影响。

**未通过项（4/11）**：L2/3/L5/L6 层激活相关检查点未通过。前馈权重提升 3.5 倍后 L2/3 仍然完全不活跃。

## 根因分析：STP 抑郁是更深的瓶颈

前馈权重提升（[0.4,1.0]→[2.5,3.5]×w_scale，提升 3.5 倍）不足以驱动 L2/3 发放，因为 **STP（短期可塑性）的 resource 在高频输入下急剧衰减**，有效电流被削弱约 200 倍：

**STP 稳态分析**（L4 每 3 步发放一次，STP kernel 只在 arrival 时调用）：
- 每次 arrival：resource 恢复 `1-exp(-1/500) ≈ 0.002`，然后消耗 `r_new = r×(1-u_new)`
- utilization 稳态 ≈ 0.75，resource 稳态 ≈ 0.005
- `delay_ring_current = weight × resource = 0.42 × 0.005 = 0.0021`
- 每个 L2/3 神经元接收 43 个突触事件：`I_inject = 43 × 0.0021 = 0.09`
- 稳态 synaptic_current ≈ 0.053，dV ≈ 0.006/步，3 步累积 V ≈ 0.02 << 阈值 1.0

**对比 L4**：L4 接收外部输入（群体编码注入，`POP_CODING_GAIN=80`），不经过 STP，所以不受影响。

**下一步**：需要新的 spec 修复 STP 抑郁问题，可能方案：
1. 为前馈连接减弱 STP 抑郁（增大 τ_rec 或减小 U_SE）
2. 为前馈连接使用易化型 STP（facilitating，而非抑郁型 depressing）
3. 让 STP resource 恢复每步执行（而非只在 arrival 时）
