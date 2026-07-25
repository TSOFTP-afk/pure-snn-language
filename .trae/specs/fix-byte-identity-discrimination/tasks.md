# Tasks

## 阶段 1：频率归一化输入增益（方案 A）

- [x] Task 1: 在 `input_encoding.cu` 中实现字节频率统计
  - [x] SubTask 1.1: 在 `load_text_corpus` 中添加 `g_byte_freq[256]` 全局数组，统计字节频率
  - [x] SubTask 1.2: 计算 `g_mean_freq = total_bytes / 256`，处理 freq=0 的情况（设为 mean_freq 避免除零）
  - [x] SubTask 1.3: 在日志中输出 Top-10 高频字节和 Top-10 低频字节的频率，验证统计正确

- [x] Task 2: 实现 `compute_freq_norm_gain(byte)` 函数
  - [x] SubTask 2.1: 在 `input_encoding.cu` 中添加 `compute_freq_norm_gain(uint8_t byte)` 函数
  - [x] SubTask 2.2: 公式：`gain = clamp(sqrt(g_mean_freq / g_byte_freq[byte]), 0.3f, 3.0f)`
  - [x] SubTask 2.3: 未加载文本时返回 1.0f（向后兼容）
  - [x] SubTask 2.4: 在 `input_encoding.cuh` 中声明该函数

- [x] Task 3: 在 `input_inject_kernel` 中集成频率归一化增益
  - [x] SubTask 3.1: 修改 `input_inject_kernel` 签名，新增 `float freq_norm_gain` 参数
  - [x] SubTask 3.2: 修改 `atomicAdd` 行：`POP_CODING_GAIN × freq_norm_gain × gain × gate`
  - [x] SubTask 3.3: 修改 `launch_input_inject` host launcher，调用 `compute_freq_norm_gain(byte)` 并传入 kernel
  - [x] SubTask 3.4: 在 `config.h` 中新增 `FREQ_NORM_GAIN_MIN=0.3f` 和 `FREQ_NORM_GAIN_MAX=3.0f` 常量

## 阶段 2：柱偏好重映射

- [x] Task 4: 实现柱偏好重映射数据结构
  - [x] SubTask 4.1: 在 `input_encoding.cu` 中添加 `g_col_prefs[50][5]` 全局数组（50 柱 × 5 偏好字节）
  - [x] SubTask 4.2: 添加 `g_col_prefs_initialized` 标志，标识是否已初始化

- [x] Task 5: 实现 `init_column_pref_mapping()` 函数
  - [x] SubTask 5.1: 按 `g_byte_freq` 对 256 字节排序（降序，高频在前）
  - [x] SubTask 5.2: 轮询分配：柱 i 偏好字节 = 排序后第 `{i, i+50, i+100, i+150, i+200}` 位
  - [x] SubTask 5.3: 验证每柱都包含至少 1 个高频字节（前 50 位）和至少 1 个低频字节（后 50 位）
  - [x] SubTask 5.4: 在 `load_text_corpus` 末尾调用 `init_column_pref_mapping()`
  - [x] SubTask 5.5: 在 `input_encoding.cuh` 中声明该函数

- [x] Task 6: 修改 `input_inject_kernel` 使用重映射后的偏好表
  - [x] SubTask 6.1: 修改 kernel 签名，新增 `const uint8_t* col_prefs` 参数（指向 `g_col_prefs` 的 device 拷贝）
  - [x] SubTask 6.2: 修改偏好判断逻辑：遍历柱的 5 个偏好字节，匹配则 `gain_in`，否则 `gain_out`
  - [x] SubTask 6.3: 在 `launch_input_inject` 中将 `g_col_prefs` 拷贝到 device（或使用 `__constant__` 内存）
  - [x] SubTask 6.4: 未加载文本时回退到连续范围映射 `[c×5, c×5+5)`（向后兼容）

## 阶段 3：离线解码器（零侵入）

- [x] Task 7: 创建 `decoder.cu` / `decoder.cuh` 基础框架
  - [x] SubTask 7.1: 创建 `decoder.cuh`，声明解码器接口
  - [x] SubTask 7.2: 创建 `decoder.cu`，实现 `load_neuron_byte_counts(ckpt_path)` 从 checkpoint 加载统计
  - [x] SubTask 7.3: 实现 `find_neuron_best_byte()` 对每个 L6 神经元找其响应最强的字节（argmax）

- [x] Task 8: 实现字节解码算法
  - [x] SubTask 8.1: 实现 `decode_text_segment(text, length)` 函数
  - [x] SubTask 8.2: 对测试文本的每个字节，统计 L6 神经元响应模式
  - [x] SubTask 8.3: 用多数投票解码预测字节
  - [x] SubTask 8.4: 输出解码准确率（预测正确字节数 / 总字节数）

- [x] Task 9: 创建独立解码器入口
  - [x] SubTask 9.1: 创建 `decoder_main.cpp`，解析 `--ckpt` 和 `--text` 参数
  - [x] SubTask 9.2: 调用解码器，输出解码报告（准确率、混淆矩阵、Top-10 字节解码效果）
  - [x] SubTask 9.3: 在 `CMakeLists.txt` 中添加 `snn_stage2e_decoder` target

## 阶段 4：本地分步验证

- [ ] Task 10: 10K 步快速验证
  - [ ] SubTask 10.1: 构建并运行 10K 步训练
  - [ ] SubTask 10.2: 检查字节解读报告中的响应比值标准差（目标 >0.5）
  - [ ] SubTask 10.3: 检查空格/罕见字节激活比（目标 <5:1）
  - [ ] SubTask 10.4: 如未达标，诊断并调整 `FREQ_NORM_GAIN_MIN/MAX` 范围

- [ ] Task 11: 100K 步完整验证
  - [ ] SubTask 11.1: 运行 100K 步训练
  - [ ] SubTask 11.2: 检查 js_mean（真实文本，目标 >0.30）
  - [ ] SubTask 11.3: 运行解码器，记录解码准确率
  - [ ] SubTask 11.4: 与当前 100K 步结果对比（响应比值、js_mean、chi2_mean）

## 阶段 5：DGX Spark 部署（验证通过后）

- [ ] Task 12: 部署到 DGX Spark
  - [ ] SubTask 12.1: git push 最新代码到 GitHub
  - [ ] SubTask 12.2: 在 DGX Spark 上 git clone 并构建
  - [ ] SubTask 12.3: 运行 10K 步烟雾测试，确认与本地结果一致
  - [ ] SubTask 12.4: 启动 3M 步完整发育训练（后台 nohup 模式）

# Task Dependencies

- Task 2 依赖 Task 1（需要频率统计）
- Task 3 依赖 Task 2（需要 `compute_freq_norm_gain` 函数）
- Task 5 依赖 Task 1（需要频率统计用于排序）
- Task 6 依赖 Task 4 和 Task 5（需要重映射后的偏好表）
- Task 8 依赖 Task 7（需要解码器基础框架）
- Task 9 依赖 Task 8（需要解码算法）
- Task 10 依赖 Task 3 和 Task 6（需要频率归一化和柱重映射都完成）
- Task 11 依赖 Task 10（10K 步验证通过后才跑 100K）
- Task 12 依赖 Task 11（100K 步验证通过后才部署 DGX Spark）

# Parallelizable Work

- Task 1-6（输入编码修改）和 Task 7-9（解码器）可以并行开发，因为解码器是零侵入的独立模块
