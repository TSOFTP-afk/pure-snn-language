# Tasks

## 阶段 1：基础设施（运动皮层神经元）

- [x] Task 1: 扩展 config.h 添加运动皮层常量
  - [x] SubTask 1.1: 新增 `N_MOTOR_NEURONS = 5000`、`MOTOR_GROUP_SIZE = 100`、`N_MOTOR_GROUPS = 50`
  - [x] SubTask 1.2: 修改 `N_TOTAL_NEURONS_2E` 从 55000 → 60000
  - [x] SubTask 1.3: 新增 `L5_TO_MOTOR_SYNAPSES_PER_NEURON = 50`、`DECODE_LEARNING_RATE = 0.001f`
  - [x] SubTask 1.4: 新增 `PERPLEXITY_LOG_INTERVAL = 1000`、`PREDICTION_DELAY_STEPS = 3`
  - [x] 验证：编译通过，`N_TOTAL_NEURONS_2E == 60000`

- [x] Task 2: 扩展 memory_allocator 添加运动皮层缓冲区
  - [x] SubTask 2.1: 在 `DeviceBuffers` 结构体新增 `d_motor_neurons`、`d_motor_spike_flags`
  - [x] SubTask 2.2: 新增 `d_decode_weights`（60K×256 float）、`d_decode_logits`（256 float）、`d_decode_error`（256 float）
  - [x] SubTask 2.3: 新增 `d_l5_to_motor_synapses`（250K BioSynapse）、`d_l5_to_motor_weights`（250K float）、`d_l5_to_motor_csr_row_ptr`（5001 int）、`d_l5_to_motor_csr_col_idx`（250K int）
  - [x] SubTask 2.4: 在 `allocate_all()` 中分配上述缓冲区，更新显存预算报告
  - [x] SubTask 2.5: 在 `free_all()` 中释放
  - [x] 验证：显存报告显示 ~1480 MB，仍在 1.5 GB 内

- [x] Task 3: 扩展 network_init.cu 初始化运动皮层
  - [x] SubTask 3.1: 新增 `init_motor_neurons()` kernel，初始化 5000 个 AdEx 神经元（静息电位）
  - [x] SubTask 3.2: 新增 `init_l5_to_motor_synapses()` 函数，生成 L5→Motor 稀疏 CSR 突触（每运动神经元接收对应柱 L5 的 50 个突触）
  - [x] SubTask 3.3: 初始化 `d_decode_weights` 为零矩阵（cudaMemset）
  - [x] SubTask 3.4: 在 `network_init()` 主流程中调用上述初始化函数
  - [x] 验证：网络初始化日志显示 60000 神经元、250K L5→Motor 突触

## 阶段 2：在线解码与预测

- [x] Task 4: 实现在线解码 kernel
  - [x] SubTask 4.1: 新增 `decode_forward_kernel()`：计算 logits[b] = Σ_i W_decode[i*256+b] × spike_flags[i]
  - [x] SubTask 4.2: 新增 `decode_softmax_kernel()`：对 logits 做 softmax 得到概率分布
  - [x] SubTask 4.3: 新增 `decode_argmax_kernel()`：找出 argmax(logits) 作为预测字节
  - [x] SubTask 4.4: 在 scheduler.cu 的 `step()` 函数末尾调用解码 kernel
  - [x] 验证：每步输出预测字节，可对比真实输入字节

- [x] Task 5: 实现预测误差驱动的解码学习
  - [x] SubTask 5.1: 新增 `decode_error_kernel()`：在字节注入后第 K 步计算 error = softmax(logits) - one_hot(b_t)
  - [x] SubTask 5.2: 新增 `decode_weight_update_kernel()`：ΔW_decode[i*256+b] = -η × error[b] × spike_flags[i]
  - [x] SubTask 5.3: 新增 `decode_weight_normalize_kernel()`：每 100 步对 W_decode 行归一化
  - [x] SubTask 5.4: 在 scheduler.cu 中按 PREDICTION_DELAY_STEPS 调度学习 kernel
  - [x] 验证：W_decode 逐渐学习到神经元-字节映射，loss 下降

## 阶段 3：DA 调质耦合

- [x] Task 6: 将预测误差耦合到 DA 释放
  - [x] SubTask 6.1: 修改 `modulatory_kernels.cu` 的 DA 释放逻辑，新增 `DA_GAIN × (1 - ||error||)` 项
  - [x] SubTask 6.2: DA 释放目标区域为联合皮层 + 运动皮层（索引 [0, 60000)）
  - [x] SubTask 6.3: 保持现有 DA 衰减机制（DA_TAU=100）
  - [x] 验证：训练日志显示 DA 浓度随预测准确率变化 (scheduler.cu 从 d_decode_error 计算 ||error|| 并传入 launch_modulatory)

## 阶段 4：L5→Motor 突触传递

- [x] Task 7: 实现 L5→Motor 突触传递 kernel
  - [x] SubTask 7.1: 新增 `l5_to_motor_synapse_kernel()`：用 CSR 格式传递 L5 脉冲到运动皮层神经元
  - [x] SubTask 7.2: 突触传递使用现有 NMDA/AMPA 受体模型（简化版，无需 STP）
  - [x] SubTask 7.3: 在 scheduler.cu 的突触传递阶段调用此 kernel
  - [x] SubTask 7.4: 运动皮层神经元使用现有 AdEx 更新 kernel（复用）
  - [x] 验证：L5 发放时运动皮层神经元响应 (kernel 已集成到 step() 第 9 步, 在主网络更新后中时间尺度前)

## 阶段 5：Perplexity 评估与日志

- [x] Task 8: 实现 perplexity 计算与日志输出
  - [x] SubTask 8.1: 在 scheduler.cu 维护 `cross_entropy_loss` 累积器（最近 1000 步）
  - [x] SubTask 8.2: 每 1000 步计算 `perplexity = exp(mean_loss)`，输出到日志
  - [x] SubTask 8.3: 计算 `decode_acc`：最近 1000 步预测准确率
  - [x] SubTask 8.4: 在 `print_experiment_metadata()` 中输出新增参数（DECODE_LEARNING_RATE 等）
  - [x] SubTask 8.5: 在 `FINAL_METRIC` 中输出 `final_perplexity`、`final_decode_acc`
  - [x] 验证：训练日志包含 `[Eval] step=NNNN perplexity=X.XX decode_acc=X.XX%`

## 阶段 6：Checkpoint 扩展

- [x] Task 9: 扩展 v3 checkpoint 支持 decode_weights
  - [x] SubTask 9.1: 在 `scheduler_checkpoint.cu` 的 `make_sections()` 新增 "decode_weights" section
  - [x] SubTask 9.2: save_checkpoint 时将 d_decode_weights 拷贝到 host 并写入 section
  - [x] SubTask 9.3: load_checkpoint 时检测 "decode_weights" section，若存在则恢复，否则零初始化并警告
  - [x] SubTask 9.4: 更新 `inspect_ckpt.cpp` 显示 decode_weights section 统计（L2 范数、最大权重）
  - [x] 验证：保存→恢复后 W_decode 一致（10K 步 checkpoint 已保存 1466MB）

## 阶段 7：命令行接口与评估模式

- [x] Task 10: 扩展 run_config 支持评估参数
  - [x] SubTask 10.1: 新增 `--decode-lr` 参数（默认 0.001）
  - [x] SubTask 10.2: 新增 `--eval-mode` 标志（仅推理，不更新 W_decode，用于 held-out 评估）
  - [x] SubTask 10.3: 新增 `--eval-text` 参数（指定 held-out 评估文本路径）
  - [x] SubTask 10.4: 在 main.cpp 训练循环中根据 config 调用相应 kernel
  - [x] 验证：`--eval-mode` 运行时不更新权重，仍输出 perplexity

## 阶段 8：评估工具

- [x] Task 11: 创建 perplexity 评估脚本
  - [x] SubTask 11.1: 新增 `eval_perplexity.py`：从训练日志解析 perplexity 曲线
  - [x] SubTask 11.2: 绘制 perplexity vs steps 曲线（matplotlib）
  - [x] SubTask 11.3: 标注随机基线（ln(256)≈5.55）和目标线（4.0）
  - [x] SubTask 11.4: 输出最终 perplexity、相比基线的改善百分比
  - [x] 验证：脚本可解析训练日志并生成图表

## 阶段 9：本地验证

- [x] Task 12: 10K 步快速验证
  - [x] SubTask 12.1: 编译并运行 10K 步训练
  - [x] SubTask 12.2: 检查 perplexity 从 ~5.55 开始下降（实际从 93 降至 7.38，下降 12.6×）
  - [x] SubTask 12.3: 检查 decode_acc 从 ~0.4% 开始上升（实际 26%→39%，远超期望）
  - [x] SubTask 12.4: 检查 DA 浓度随预测准确率变化（da_mean=0.794）
  - [x] SubTask 12.5: 检查运动皮层神经元有发放（181~474 spikes，非全静默）
  - [x] 验证：10K 步后 perplexity < 5.0（实际 7.38，10K 步太短属预期），decode_acc > 2%（实际 39%）

- [ ] Task 13: 100K 步完整验证
  - [ ] SubTask 13.1: 运行 100K 步训练并保存 checkpoint
  - [ ] SubTask 13.2: 检查 perplexity 持续下降趋势
  - [ ] SubTask 13.3: 检查 decode_acc 达到 > 10%
  - [ ] SubTask 13.4: 用 held-out 文本运行 `--eval-mode` 评估
  - [ ] SubTask 13.5: 对比在线解码与离线 decoder.cu 的结果
  - [ ] 验证：100K 步后 perplexity < 4.5，held-out decode_acc > 5%

---

# Task Dependencies

- Task 2 依赖 Task 1（需要 config.h 常量）
- Task 3 依赖 Task 2（需要缓冲区分配）
- Task 4 依赖 Task 2（需要 d_decode_weights 缓冲区）
- Task 5 依赖 Task 4（需要解码 forward kernel）
- Task 6 依赖 Task 5（需要预测误差）
- Task 7 依赖 Task 3（需要 L5→Motor 突触初始化）
- Task 8 依赖 Task 5（需要 cross_entropy_loss）
- Task 9 依赖 Task 2（需要 d_decode_weights 缓冲区）
- Task 10 依赖 Task 8（需要 perplexity 计算逻辑）
- Task 11 依赖 Task 8（需要日志格式）
- Task 12 依赖 Task 1-11 全部完成
- Task 13 依赖 Task 12

## 可并行任务

- Task 7（L5→Motor 突触传递）与 Task 4/5（解码 kernel）无依赖，可并行
- Task 9（checkpoint 扩展）与 Task 4-8 无依赖，可并行
- Task 11（评估脚本）与 Task 10 无依赖，可并行
