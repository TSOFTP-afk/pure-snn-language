# Checklist

## 阶段 1：频率归一化输入增益

- [ ] `load_text_corpus` 成功加载文本后，`g_byte_freq[256]` 被正确填充
- [ ] 日志输出 Top-10 高频字节和 Top-10 低频字节频率，数值合理
- [ ] `compute_freq_norm_gain(byte)` 公式正确：`clamp(sqrt(mean_freq/freq), 0.3, 3.0)`
- [ ] 频率为 0 的字节返回增益 3.0（上限），不产生 NaN
- [ ] 未加载文本时 `compute_freq_norm_gain(byte)` 返回 1.0（向后兼容）
- [ ] `input_inject_kernel` 中 `atomicAdd` 包含 `freq_norm_gain` 因子
- [ ] `config.h` 中新增 `FREQ_NORM_GAIN_MIN=0.3f` 和 `FREQ_NORM_GAIN_MAX=3.0f`

## 阶段 2：柱偏好重映射

- [ ] `g_col_prefs[50][5]` 数组在文本加载后被正确初始化
- [ ] 柱 i 偏好的 5 字节 = 按频率排序后的第 `{i, i+50, i+100, i+150, i+200}` 位
- [ ] 每柱都包含至少 1 个高频字节（前 50 位）和至少 1 个低频字节（后 50 位）
- [ ] `input_inject_kernel` 使用重映射后的偏好表判断 `gain_in`/`gain_out`
- [ ] 未加载文本时回退到连续范围映射 `[c×5, c×5+5)`（向后兼容）
- [ ] 柱偏好增益保持现状：`gain_in=2.0, gain_out=0.3`（6.67:1）

## 阶段 3：离线解码器

- [ ] `decoder.cu` / `decoder.cuh` 文件创建
- [ ] `load_neuron_byte_counts(ckpt_path)` 能正确从 checkpoint 加载 55K×256 矩阵
- [ ] `find_neuron_best_byte()` 对每个 L6 神经元返回其响应最强的字节
- [ ] `decode_text_segment()` 对测试文本输出预测字节序列
- [ ] 解码准确率计算正确：预测正确字节数 / 总字节数
- [ ] `decoder_main.cpp` 解析 `--ckpt` 和 `--text` 参数
- [ ] 解码报告包含：准确率、混淆矩阵、Top-10 字节解码效果
- [ ] `CMakeLists.txt` 中添加 `snn_stage2e_decoder` target
- [ ] 解码器不修改任何训练流程代码（零侵入验证）

## 阶段 4：本地分步验证

### 10K 步快速验证
- [ ] 构建无错误，10K 步训练无 crash
- [ ] 字节解读报告中响应比值标准差 > 0.5（当前 ~0.1）
- [ ] 空格/罕见字节激活比 < 5:1（当前 34:1）
- [ ] 四层皮层（L4/L2-3/L5/L6）全部激活

### 100K 步完整验证
- [ ] 100K 步训练完成无 crash
- [ ] js_mean（真实文本）> 0.30（当前 0.19）
- [ ] 解码器准确率 > 1/256 ≈ 0.39%（随机基线）
- [ ] 解码器准确率 > 5%（初步学习标志）
- [ ] 与当前 100K 步结果对比：响应比值、js_mean、chi2_mean 均有提升

## 阶段 5：DGX Spark 部署

- [ ] 代码已 push 到 GitHub
- [ ] DGX Spark 上 git clone 并构建成功
- [ ] 10K 步烟雾测试结果与本地一致（允许 ±10% 误差）
- [ ] 3M 步完整发育训练已启动（后台 nohup 模式）
- [ ] 训练日志可正常监控（tail -f）

## 零侵入保证

- [ ] `scheduler.cu` 未被修改
- [ ] `synapse_kernels.cu` 未被修改
- [ ] `neuron_kernels.cu` 未被修改
- [ ] `modulatory_kernels.cu` 未被修改
- [ ] `thalamic_gate.cu` 未被修改
- [ ] 解码器仅 read-only 访问 checkpoint 文件
