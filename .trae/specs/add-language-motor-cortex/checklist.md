# Checklist

## 阶段 1：基础设施（运动皮层神经元）

- [x] config.h 新增 N_MOTOR_NEURONS、MOTOR_GROUP_SIZE 等常量
- [x] N_TOTAL_NEURONS_2E 更新为 60000
- [x] DECODE_LEARNING_RATE、PERPLEXITY_LOG_INTERVAL 等参数定义
- [x] memory_allocator 新增 d_motor_neurons 缓冲区
- [x] memory_allocator 新增 d_decode_weights (60K×256) 缓冲区
- [x] memory_allocator 新增 d_l5_to_motor_synapses 等 CSR 缓冲区
- [x] 显存预算报告显示 ~1480 MB，仍在 1.5 GB 内
- [x] free_all() 释放所有新增缓冲区
- [x] network_init 初始化 5000 个运动皮层 AdEx 神经元
- [x] network_init 生成 L5→Motor 稀疏 CSR 突触（250K 突触）
- [x] d_decode_weights 零初始化
- [x] 网络初始化日志显示 60000 神经元

## 阶段 2：在线解码与预测

- [x] decode_forward_kernel 正确计算 logits = W_decode × spike_flags
- [x] decode_softmax_kernel 对 logits 做 softmax
- [x] decode_argmax_kernel 找出预测字节
- [x] scheduler.step() 末尾调用解码 kernel
- [x] 每步输出预测字节（可对比真实输入）

## 阶段 3：预测误差驱动的解码学习

- [x] decode_error_kernel 计算 error = softmax(logits) - one_hot(b_t)
- [x] decode_weight_update_kernel 用 ΔW = -η × error × spike_flags 更新权重
- [x] decode_weight_normalize_kernel 每 100 步行归一化
- [x] 按 PREDICTION_DELAY_STEPS=3 调度学习 kernel
- [ ] W_decode 逐渐学习到神经元-字节映射 (待训练验证)
- [ ] cross_entropy_loss 随训练下降 (待训练验证)

## 阶段 4：DA 调质耦合

- [x] DA 释放逻辑新增 DA_GAIN × (1 - ||error||) 项
- [x] DA 释放目标区域包含运动皮层
- [x] DA 衰减机制（DA_TAU=100）保持不变
- [ ] 训练日志显示 DA 浓度随预测准确率变化 (待训练验证)

## 阶段 5：L5→Motor 突触传递

- [x] l5_to_motor_synapse_kernel 用 CSR 格式传递脉冲
- [x] 突触传递使用 NMDA/AMPA 受体模型
- [x] scheduler 突触传递阶段调用此 kernel
- [x] 运动皮层神经元使用现有 AdEx 更新 kernel
- [ ] L5 发放时运动皮层神经元有响应 (待训练验证)

## 阶段 6：Perplexity 评估与日志

- [x] cross_entropy_loss 累积器维护最近 1000 步
- [x] 每 1000 步计算 perplexity = exp(mean_loss)
- [x] 训练日志包含 `[Eval] step=NNNN perplexity=X.XX decode_acc=X.XX%`
- [x] print_experiment_metadata 输出新增参数
- [x] FINAL_METRIC 输出 final_perplexity、final_decode_acc

## 阶段 7：Checkpoint 扩展

- [ ] v3 checkpoint 新增 "decode_weights" section
- [ ] save_checkpoint 写入 d_decode_weights
- [ ] load_checkpoint 检测并恢复 decode_weights（旧 checkpoint 零初始化）
- [ ] inspect_ckpt.cpp 显示 decode_weights section 统计
- [ ] 保存→恢复后 W_decode 数值一致

## 阶段 8：命令行接口与评估模式

- [ ] --decode-lr 参数支持（默认 0.001）
- [ ] --eval-mode 标志（仅推理，不更新权重）
- [ ] --eval-text 参数（held-out 评估文本路径）
- [ ] main.cpp 训练循环根据 config 调用相应 kernel
- [ ] --eval-mode 运行时不更新权重但仍输出 perplexity

## 阶段 9：评估工具

- [ ] eval_perplexity.py 可解析训练日志
- [ ] 绘制 perplexity vs steps 曲线
- [ ] 标注随机基线（5.55）和目标线（4.0）
- [ ] 输出最终 perplexity 和改善百分比

## 阶段 10：本地验证

- [ ] 10K 步训练编译并运行成功
- [ ] 10K 步后 perplexity < 5.0
- [ ] 10K 步后 decode_acc > 2%
- [ ] 10K 步后运动皮层有发放（非全静默）
- [ ] DA 浓度随预测准确率变化
- [ ] 100K 步训练并保存 checkpoint 成功
- [ ] 100K 步后 perplexity < 4.5
- [ ] 100K 步后 decode_acc > 10%
- [ ] held-out 文本评估 decode_acc > 5%
- [ ] 在线解码与离线 decoder.cu 结果可对比

## 显存预算验证

- [ ] 实际显存占用 ≤ 1500 MB（RTX 3060 预算上限）
- [ ] 显存报告显示新增项明细（d_decode_weights ~59 MB 等）
- [ ] 训练过程中无 CUDA OOM 错误

## 生物合理性验证

- [ ] 运动皮层神经元在 L5 发放后延迟 1-3 步响应
- [ ] 运动群之间形成字节偏好分化
- [ ] 预测误差与 DA 释放呈负相关（越准 DA 越高）
- [ ] W_decode 矩阵的神经元-字节映射与 neuron_byte_counts 一致性
