# 语言运动皮层与输出解码闭环 Spec

## Why

当前 Stage 2e 网络只有"输入→内部表征"的单向通路，缺少"输入→内部表征→输出预测"的完整闭环。这导致两个核心问题：

1. **无法验证真正的语义理解**：现有"解码器"只是离线查表（从 neuron_byte_counts 矩阵反查偏好字节），不是网络在线生成输出。没有 next-byte 预测就无法计算 perplexity，无法评估"网络是否学到语言结构"。
2. **缺少训练监督信号**：当前训练纯靠无监督 STDP，没有预测误差驱动的学习信号。引入 next-byte 预测任务后，预测误差可作为调质系统的 DA 信号源，形成"预测→误差→调质→学习"闭环。

参考 [human-brain-gap-module-assessment.md](file:///f:/项目/THE%20TRUE%20AI/docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md) §14：模块 M（语言输入编码与输出解码闭环）优先级"很高"，是"真实 SNN 可用于简单对话"的必经阶段。

## What Changes

### 新增：语言运动皮层（Motor Cortex）

- 在 55K 神经元基础上新增 **5K 运动皮层神经元**（独立区域，不挤占联合皮层）
- 神经元总数：55K → **60K**
- 运动皮层组织为 **50 个运动群**（与 50 柱一一对应），每群 100 神经元
- 每个运动群对应一个"候选输出字节"（50 个运动群 → 256 字节的多对一映射）
- 运动皮层接收来自联合皮层 L5 层的**前馈兴奋性连接**（L5 → Motor）

### 新增：解码权重矩阵（在线解码）

- **解码权重矩阵 W_decode**：[60K × 256] 的 float32 矩阵（58.6 MB）
- 训练阶段：用 STDP + 预测误差驱动 W_decode 学习"神经元群体活动 → 输入字节"的映射
- 推理阶段：每步用 W_decode × spike_flags 计算 logits，argmax 得到预测字节
- **解码梯度**：预测误差通过 W_decode 反传到神经元 eligibility trace（三因素学习的第三因素）

### 新增：next-byte 预测训练循环

- 每注入一个字节 b_t 后，延迟 K 步（K=3，与 INPUT_INJECT_INTERVAL 一致）后：
  1. 读取当前步网络输出 logits = W_decode × spike_flags
  2. 计算预测误差：target = one_hot(b_t)，error = softmax(logits) - target
  3. 更新 W_decode（Hebbian：ΔW = η · error · spike_flags）
  4. 释放 DA 调质：DA = -||error||（预测越准 DA 越高）
- 每 1000 步计算并记录 **perplexity** 指标

### 新增：在线解码评估工具

- 新增 `online_decoder.cu`：训练过程中每步计算预测字节
- 新增 `eval_perplexity.py`：从训练日志读取 perplexity 曲线，对比随机基线（ln(256)≈5.55）
- 扩展 checkpoint：新增 "decode_weights" section（保存 W_decode）

### 修改：显存预算

- RTX 3060 显存占用：1.4 GB → **1.49 GB**（+60 MB，仍在 1.5 GB 预算内）
- DGX Spark：无约束

### 修改：网络初始化

- [network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu)：新增运动皮层神经元初始化
- [memory_allocator.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/memory_allocator.cu)：新增 d_motor_neurons、d_decode_weights 缓冲区
- [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h)：新增 N_MOTOR_NEURONS、MOTOR_GROUP_SIZE 等常量

## Impact

- **Affected specs**:
  - [fix-byte-identity-discrimination](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/fix-byte-identity-discrimination/spec.md) — 解码器从离线查表升级为在线预测
  - [add-cortical-layer-hierarchy](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/add-cortical-layer-hierarchy/spec.md) — L5 输出新增到运动皮层的通路
  - [add-thalamic-gating](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/add-thalamic-gating/spec.md) — 丘脑门控可扩展到运动输出选择

- **Affected code**:
  - [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) — 神经元总数、运动皮层参数
  - [memory_allocator.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/memory_allocator.cu) / [memory_allocator.cuh](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/memory_allocator.cuh) — 新增缓冲区
  - [network_init.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/network_init.cu) — 运动皮层神经元 + L5→Motor 突触
  - [scheduler.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu) — 新增解码 kernel 调用
  - [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) — 新增 L5→Motor 突触传递
  - [modulatory_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/modulatory_kernels.cu) — DA 释放与预测误差耦合
  - [main.cpp](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/main.cpp) — 训练日志新增 perplexity 输出
  - [scheduler_checkpoint.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler_checkpoint.cu) — 新增 decode_weights section
  - [run_config.cpp](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/run_config.cpp) / [run_config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/run_config.h) — 新增 --eval-mode 参数

## ADDED Requirements

### Requirement: 运动皮层神经元群

系统 SHALL 在联合皮层之外新增独立的运动皮层区域，包含 5000 个 AdEx 神经元，组织为 50 个运动群（每群 100 神经元），每个运动群对应一个候选输出字节类别。

#### Scenario: 运动皮层初始化
- **WHEN** 网络初始化完成
- **THEN** 存在 5000 个运动皮层神经元，全局索引范围为 [55000, 60000)
- **AND** 50 个运动群各含 100 神经元，群内兴奋性:抑制性 = 80:20
- **AND** 运动皮层神经元初始膜电位为静息电位 (-70mV)

#### Scenario: L5 → Motor 前馈连接
- **WHEN** 网络初始化完成
- **THEN** 每个运动皮层神经元接收来自对应柱 L5 层的 50 个兴奋性突触
- **AND** 突触延迟为 1-3 步（柱内延迟）
- **AND** 初始权重在 [0.3, 0.8] 范围内（强连接，确保信号传递）

### Requirement: 在线解码权重矩阵

系统 SHALL 维护一个 [N_TOTAL_NEURONS × 256] 的解码权重矩阵 W_decode，用于从神经元群体活动预测当前输入字节。

#### Scenario: 解码权重初始化
- **WHEN** 网络初始化完成
- **THEN** W_decode 矩阵为零初始化（避免初始偏置）
- **AND** 矩阵显存占用为 60000 × 256 × 4 = 61.44 MB

#### Scenario: 在线解码预测
- **WHEN** 每步网络更新完成后
- **THEN** 计算 logits[b] = Σ_i W_decode[i*256+b] × spike_flags[i] 对所有 b ∈ [0,256)
- **AND** 预测字节 = argmax(logits)
- **AND** 计算 softmax(logits) 得到字节概率分布

### Requirement: 预测误差驱动的解码学习

系统 SHALL 在每字节注入后延迟 K 步（K=3）计算预测误差，并用误差驱动 W_decode 更新和 DA 释放。

#### Scenario: 预测误差计算
- **WHEN** 字节 b_t 注入后第 K 步
- **THEN** 读取当前步 logits
- **AND** 计算 target = one_hot(b_t)
- **AND** 计算 error = softmax(logits) - target
- **AND** 记录 cross_entropy_loss = -log(softmax(logits)[b_t])

#### Scenario: 解码权重更新
- **WHEN** 预测误差计算完成
- **THEN** 更新 W_decode：ΔW_decode[i*256+b] = -η_decode × error[b] × spike_flags[i]
- **AND** 学习率 η_decode = 0.001（初始值，可配置）
- **AND** W_decode 行归一化（防止权重爆炸）

#### Scenario: DA 调质释放
- **WHEN** 预测误差计算完成
- **THEN** 释放 DA 到联合皮层 + 运动皮层
- **AND** DA 浓度 = DA_BASE + DA_GAIN × (1 - ||error||)
- **AND** 预测越准（||error|| 越小）DA 越高，触发更多 STDP 学习

### Requirement: Perplexity 评估指标

系统 SHALL 每 1000 步计算并记录 next-byte 预测的 perplexity，用于评估网络语言建模能力。

#### Scenario: Perplexity 计算
- **WHEN** 训练步数 % 1000 == 0
- **THEN** 收集最近 1000 步的 cross_entropy_loss
- **AND** 计算 mean_loss = mean(losses)
- **AND** 计算 perplexity = exp(mean_loss)
- **AND** 输出到训练日志：`[Eval] step=NNNN perplexity=X.XX (baseline=5.55)`

#### Scenario: Perplexity 达标判据
- **WHEN** 100K 步训练完成
- **THEN** perplexity 应低于随机基线 5.55
- **AND** 理想目标：perplexity < 4.0（学到字节级统计规律）

### Requirement: Checkpoint 扩展

系统 SHALL 在 v3 checkpoint 中新增 "decode_weights" section，保存 W_decode 矩阵。

#### Scenario: Checkpoint 保存
- **WHEN** save_checkpoint 被调用
- **THEN** 新增 section "decode_weights"，大小为 N_TOTAL_NEURONS × 256 × 4 字节
- **AND** section 数据为 W_decode 矩阵的 device-to-host 拷贝

#### Scenario: Checkpoint 恢复
- **WHEN** load_checkpoint 被调用且存在 "decode_weights" section
- **THEN** 从 section 数据恢复 W_decode 矩阵
- **AND** 若 section 不存在（旧 checkpoint），用零初始化并打印警告

## MODIFIED Requirements

### Requirement: 网络规模

[修改自 add-cortical-layer-hierarchy spec]

网络总神经元数从 55000 扩展为 **60000**（55000 联合皮层 + 5000 运动皮层）。柱结构保持不变（50 柱 × 1000 神经元），运动皮层作为独立区域。

### Requirement: 显存预算

[修改自 bio-mechanisms-design v4]

持久显存预算从 1401 MB 扩展为 **~1462 MB**（+61 MB 解码权重矩阵）。仍在 RTX 3060 的 1.5 GB 预算上限内，余量 ~38 MB。

### Requirement: 训练日志

[修改自现有 main.cpp 训练日志]

训练日志每 1000 步新增输出：
- `perplexity=X.XX`（next-byte 预测困惑度）
- `decode_acc=X.XX%`（最近 1000 步预测准确率）
- `da_release=X.XX`（平均 DA 释放量）

## REMOVED Requirements

### Requirement: 离线查表解码器

**Reason**: 现有 [decoder.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/decoder.cu) 是离线工具，只能从 checkpoint 的 neuron_byte_counts 矩阵反查偏好字节，不是真正的在线解码。引入在线解码后，离线工具仍保留用于调试，但不再是主要评估手段。
**Migration**: [decoder.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/decoder.cu) 和 [show_decode_effect.py](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/show_decode_effect.py) 保留作为离线分析工具，不删除。新评估流程改用在线 perplexity 指标。

---

## 显存预算明细

| 缓冲区 | 数量 | 大小 | 字节 | MB |
|--------|------|------|------|-----|
| d_motor_neurons | 5000 | 56B | 280000 | 0.27 |
| d_motor_spike_flags | 5000 | 1B | 5000 | 0.005 |
| d_decode_weights (60K×256) | 15360000 | 4B | 61440000 | **58.59** |
| d_decode_logits | 256 | 4B | 1024 | 0.001 |
| d_decode_error | 256 | 4B | 1024 | 0.001 |
| d_l5_to_motor_synapses | 250000 | 80B | 20000000 | 19.07 |
| d_l5_to_motor_weights | 250000 | 4B | 1000000 | 0.95 |
| **合计新增** | | | | **~79 MB** |

**修正后总显存**：1401 MB + 79 MB = **1480 MB**（仍在 1.5 GB 预算内，余量 20 MB）

> ⚠️ 余量紧张。若 L5→Motor 突触改为稀疏 CSR（每运动神经元 50 突触 × 5000 = 250K 突触，已按此计算），可控。若需更多余量，可减少运动群数量（50→25）。

---

## 与后续升级的关系

本 spec 是"后续机制升级路线图"的第一步，后续升级（独立 spec）：

| 后续 spec | 依赖本 spec | 显存增量 |
|-----------|------------|---------|
| add-dendritic-compartments-full | 否（可并行） | +30-190 MB |
| add-hippocampal-replay-loop | 是（用 perplexity 作为重放触发信号） | +560 MB |
| add-basal-ganglia-action-selection | 是（用运动皮层作为行动输出层） | +420 MB |
| add-cerebellar-error-correction | 是（用预测误差作为攀登纤维信号） | +600 MB |
| add-neuromodulator-closed-loop | 是（DA 与预测误差已在本 spec 耦合） | +25 MB |
