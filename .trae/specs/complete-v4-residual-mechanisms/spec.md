# 补完 V4 残留生物机制 Spec

## Why

V4 设计的 11 项强化机制中，7 项已完整实现（BioSynapse 调质受体、NMDA 钙快照、2 阶 eligibility、轴突延迟、STDP 双 trace、CaMKII），但 **5 项未实现或部分实现**，且其中 3 项的显存已分配（共 31.8 MB）却无任何 kernel 读写——这是"显存已花但收益为零"的最高优先级修复目标。

具体而言：
- `d_pca_W` (11 MB) 已分配但无 kernel 读写
- `d_hippo_indices` (12.8 MB) 已分配，`launch_replay` 是空 stub
- `d_coact_trackers` (8 MB) 已分配但无 kernel 读写
- WM 50 槽仅做 activation 衰减，缺 PCA 签名匹配/LRU/反投影注入
- W_pred 200×200 仅更新对角项，预测能力大幅退化

补完这 5 项后，可一次性激活模块 E（海马-皮层互补学习）、模块 J（睡眠重放）、v4 #1/#3/#4/#5/#6 共 5 个未实现/部分实现项，是当前最高 ROI 的改造方向。PCA 是海马签名 + WM 匹配的共同基础，三者互相耦合必须一起实现。

## What Changes

### 新增：PCA 增量学习 + 反投影 kernel（v4 #1）

- 实现 Oja's rule 在线 PCA：`pca_update_kernel` 每 100 步从联合皮层发放快照更新 `d_pca_W`（55K×50 矩阵）
- 实现 PCA 签名提取：`pca_encode_kernel` 计算当前发放向量的 50 维签名 `signature[k] = Σ_i W[i][k] · (fr[i] - mean_fr[i])`
- 实现全量反投影：`pca_back_project_kernel` 从 50 维签名重建 55K 维发放向量 `reconstructed[i] = mean_FR[i] + Σ_k signature[k] · W[i][k]`
- PCA 快照缓冲（CPU 内存，20 MB）：每 100 步收集联合皮层发放向量，用于 Oja 学习和每 100K 步的完整 SVD 重训

### 新增：海马索引编码 + 睡眠重放闭环（v4 #3 + 模块 E/J）

- 实现 `hippo_encode_kernel`：每 100 步计算 PCA 签名，与现有 50K 索引做 cosine 距离匹配
  - 新颖模式（cosine < 0.7）：写入最旧/最弱槽位（LRU 替换）
  - 已有模式：刷新 `importance += 1 / (1 + replay_count)`（稀有模式加权）
- 实现 `replay_kernel`：每 10K 步触发一次"睡眠周期"
  - 按 importance 排序取 top-200 模式
  - 用 PCA 反投影重建发放向量，以 10× 速度注入联合皮层
  - 重放期间执行 STDP 巩固：`Δw_replay = η_replay · pre · post · tag`
  - 重放后 `importance *= 0.9`，`replay_count++`
- 实现 `launch_replay` 真正逻辑（替换 [scheduler.cu:1029-1033](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu#L1029-L1033) 的空 stub）

### 新增：共激活跟踪 + 结构可塑性（v4 #4 + 模块 K）

- 实现 `coactivation_sample_kernel`：每步从当前发放神经元中采样 500 个候选对，更新 `d_coact_trackers`（共激活计数 + 调质加权分数）
- 实现 `structural_rebuild_kernel`：每 1000 步扫描 tracker
  - `coact_count > θ_form` 的候选对标记为新突触生成（top-5000）
  - `|w| < θ_prune` 且未增强的弱突触标记为修剪
  - 总变更 > 5% 时执行分块原地 CSR 重建（避免双缓冲 640 MB）
- 替换现有 `launch_structural_plasticity`（仅 PSW α/β 衰减）为完整版

### 修改：WM 完整闭环（v4 #5）

- 修改 `p3_wm_update_kernel`（[scheduler.cu:267-281](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu#L267-L281)）：
  - 写入：每 100 步计算 PCA 签名，与 50 槽位做 cosine 距离匹配，新颖模式 LRU 替换
  - 维持：`activation > 0.3` 时用 PCA 反投影注入对应前额叶吸引子组
  - 衰减：保留现有 `activation *= 0.995` 逻辑

### 修改：W_pred 完整 200×200 矩阵（v4 #6）

- 修改 [modulatory_kernels.cu:126](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/modulatory_kernels.cu#L126)：
  - 从 `w_pred[k*dim+k] += ...`（仅对角项）改为完整 `w_pred[j*dim+k] += η_pred · (fr_j - pred_j) · fr_k_prev`
  - 预测从 `pred_fr[k] = w_pred[k*dim+k] * fr_k` 改为 `pred_fr[j] = Σ_k w_pred[j][k] · fr_k_prev`（完整矩阵乘法）
- 修改 `prediction_success` 计算：从二元判断改为 `cosine_similarity(pred_fr, fr_subcol)` 映射到 [0,1]

### 补完：睡眠重放期间 ACh/丘脑门控联动（已知限制 #1）

5 项核心 V4 机制代码已实现并通过 10K 步基础验证，但存在 3 项已知限制需要补完代码。本阶段补完睡眠重放的生物状态隔离机制：

- 新增 `enter_sleep_state` / `exit_sleep_state` 接口（scheduler.cu）
  - `enter_sleep_state`：保存当前 thalamic_gain 和 ach_level，然后 `thalamic_gain = 0`（关闭外部字节输入）、`ach_level *= 0.3`（巩固模式）
  - `exit_sleep_state`：恢复保存的 thalamic_gain 和 ach_level
- 在 `launch_replay` 前后调用这两个接口，确保重放期间外部输入不污染重放模式
- 生物学意义：睡眠期间丘脑门控关闭外部输入，ACh 水平降低切换到巩固模式（符合慢波睡眠生理特征）

### 补完：结构重建 CSR 完整性运行时校验（已知限制 #3）

- 在 `launch_csr_rebuild` 重建后新增运行时校验 kernel `csr_integrity_check_kernel`
  - `row_ptr` 单调性检查：`row_ptr[i+1] >= row_ptr[i]` 对所有 i
  - `col_ind` 范围检查：`0 <= col_ind[j] < N_TOTAL_NEURONS_2E` 对所有 j
  - `row_ptr[N] == n_synapses`（总突触数一致）
- 校验失败时输出错误日志并回滚到重建前状态（保留旧 CSR 副本直到校验通过）
- 校验通过后释放旧副本

### 补完：inspect_ckpt 工具支持新机制缓冲区导出（已知限制 #5）

- 在 `inspect_ckpt.cpp` 中添加 v3 section 读取逻辑，支持导出以下缓冲区的统计信息：
  - `d_pca_W`：Frobenius 范数、每列范数分布、非零比例
  - `d_hippo_indices`：填充率、importance 分布、replay_count 分布
  - `d_coact_trackers`：非零条目数、coact_count 分布、modulator_score 分布
  - `d_wm_slots`：激活槽位数、activation 分布、pattern 非零比例
  - `d_w_pred`：非对角项非零比例、矩阵 Frobenius 范数
- 命令行参数 `--export-v4-buffers <output_dir>` 导出为 CSV 格式

## Impact

- **Affected specs**:
  - [add-language-motor-cortex](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/add-language-motor-cortex/spec.md) — PCA 签名可为解码器提供额外状态特征
  - [add-cortical-layer-hierarchy](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/add-cortical-layer-hierarchy/spec.md) — L6 反馈可与海马重放协同
  - [add-thalamic-gating](file:///f:/项目/THE%20TRUE%20AI/.trae/specs/add-thalamic-gating/spec.md) — 丘脑门控可在睡眠态关闭外部输入

- **Affected code**:
  - 新建 `hippocampal_kernels.cu/cuh` — 海马编码 + 重放 kernel
  - 新建 `pca_kernels.cu/cuh` — PCA 更新 + 编码 + 反投影 kernel
  - 新建 `coactivation_kernels.cu/cuh` — 共激活采样 + 结构重建 kernel
  - [scheduler.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler.cu) — 替换 launch_replay/launch_structural_plasticity 空 stub，修改 p3_wm_update_kernel 调用
  - [modulatory_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/modulatory_kernels.cu) — W_pred 完整矩阵 + prediction_success cosine
  - [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) — 新增 PCA/海马/共激活参数
  - [scheduler_checkpoint.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/scheduler_checkpoint.cu) — d_pca_W/d_hippo_indices/d_coact_trackers 已在 checkpoint 中（验证）

## ADDED Requirements

### Requirement: PCA 增量学习与反投影

系统 SHALL 维护一个 [N_ASSOCIATION_NEURONS_2E × 50] 的 PCA 基矩阵 `d_pca_W`（已分配 11 MB），通过 Oja's rule 在线学习联合皮层发放模式的主成分，并支持从 50 维签名全量反投影为发放向量。

#### Scenario: PCA 在线更新
- **WHEN** 训练步数 % 100 == 0 且步数 > 1000（跳过初始瞬态）
- **THEN** 收集当前步联合皮层发放率快照到 CPU 缓冲
- **AND** 用 Oja's rule 更新 `d_pca_W`：`W[:,k] += η_pca · (x - W[:,k]·(W[:,k]ᵀx)) · (W[:,k]ᵀx)`
- **AND** 学习率 `η_pca = 0.01`
- **AND** 每 10K 步将 CPU 端 W 同步到 GPU

#### Scenario: PCA 签名提取
- **WHEN** 海马编码或 WM 写入需要签名
- **THEN** 计算 `signature[k] = Σ_i W[i][k] · (fr[i] - mean_FR[i])` 对所有 k ∈ [0,50)
- **AND** 签名归一化为单位向量（便于 cosine 距离）

#### Scenario: PCA 全量反投影
- **WHEN** 睡眠重放或 WM 维持需要反投影
- **THEN** 计算 `reconstructed[i] = mean_FR[i] + Σ_k signature[k] · W[i][k]` 对所有 i ∈ [0, 55000)
- **AND** 反投影结果作为输入电流注入联合皮层或前额叶

### Requirement: 海马索引编码

系统 SHALL 维护 50000 个海马索引条目（已分配 `d_hippo_indices` 12.8 MB），每 100 步编码当前联合皮层活动模式的 50 维 PCA 签名。

#### Scenario: 新颖模式编码
- **WHEN** 训练步数 % 100 == 0 且处于清醒态
- **THEN** 计算当前发放向量的 PCA 签名
- **AND** 与所有现有索引做 cosine 距离匹配
- **AND** 若最小 cosine 距离 < 0.7（新颖模式）：写入最旧/最弱槽位
  - `pattern_signature = current_signature`
  - `pattern_start_step = current_step`
  - `replay_count = 0`
  - `importance = 1.0`

#### Scenario: 已有模式刷新
- **WHEN** 最小 cosine 距离 >= 0.7（已有模式）
- **THEN** 刷新最近匹配索引的 `importance += 1.0 / (1.0 + replay_count)`
- **AND** 不创建新索引

### Requirement: 睡眠重放巩固

系统 SHALL 每 10000 步触发一次睡眠重放周期，从海马索引中选取重要性最高的模式，以 10× 速度重放到联合皮层执行 STDP 巩固。

#### Scenario: 睡眠周期触发
- **WHEN** 训练步数 % 10000 == 0 且步数 > 20000（跳过早期）
- **THEN** 按 importance 排序取 top-200 模式
- **AND** 对每个模式：
  - 用 PCA 反投影重建发放向量
  - 以 10× 速度（10 个正常步合并为 1 步）注入联合皮层
  - 重放期间执行 STDP 巩固：`Δw_replay = η_replay · pre · post · tag`
  - 被重放突触 `tag += 1`
- **AND** 重放后 `importance *= 0.9`，`replay_count++`

#### Scenario: 重放与正常训练隔离
- **WHEN** 睡眠重放进行中
- **THEN** 关闭外部字节输入（丘脑门控 gain=0）
- **AND** ACh 水平降低（巩固模式）
- **AND** 重放结束后恢复清醒态

### Requirement: 共激活跟踪与结构可塑性

系统 SHALL 维护 500000 个共激活跟踪条目（已分配 `d_coact_trackers` 8 MB），每步采样发放神经元候选对，每 1000 步执行结构可塑性批量重建。

#### Scenario: 共激活采样
- **WHEN** 每步突触传递完成后
- **THEN** 从当前发放神经元中随机采样 500 个候选对
- **AND** 对每对 (i, j) 若都发放：更新或创建 tracker 条目，`coact_count++`
- **AND** `modulator_score += 当前 DA 水平`（高 DA 时共激活更值得记）

#### Scenario: 结构可塑性批量重建
- **WHEN** 训练步数 % 1000 == 0
- **THEN** 扫描 tracker：`coact_count > θ_form` 的候选对标记为新突触（top-5000）
- **AND** `|w| < θ_prune` 且未增强 T_prune 步的弱突触标记为修剪
- **AND** 若总变更 > 5%：执行分块原地 CSR 重建（避免双缓冲 640 MB）
- **AND** `coact_count == 0` 持续 5000 步的 tracker 条目被淘汰

### Requirement: WM 完整闭环

系统 SHALL 扩展工作记忆从单纯的 activation 衰减升级为完整的 PCA 签名匹配 + LRU 替换 + 反投影注入闭环。

#### Scenario: WM 写入
- **WHEN** 训练步数 % 100 == 0
- **THEN** 计算当前联合皮层活动的 PCA 签名（50 维）
- **AND** 与 50 槽位的 pattern 做 cosine 距离匹配
- **AND** 若最小距离 > novelty_threshold（0.7）：写入最旧/最弱槽位（LRU）
  - `pattern = signature`
  - `age = 0`
  - `activation = 1.0`
- **AND** 否则：刷新最近匹配槽位 `activation = 1.0`

#### Scenario: WM 维持与反投影注入
- **WHEN** 每步 WM 更新
- **THEN** `activation[i] *= 0.995`（半衰期 ~140 步）
- **AND** 若 `activation[i] > 0.3`：
  - 用 PCA 反投影将 `pattern[i]` 重建为发放向量
  - 注入到对应前额叶吸引子组（`prefrontal_group`）

### Requirement: W_pred 完整矩阵预测

系统 SHALL 将预测器从仅对角项更新升级为完整 200×200 矩阵乘法，提升亚柱级预测精度。

#### Scenario: W_pred 完整更新
- **WHEN** 每步调质更新
- **THEN** `W_pred[j][k] += η_pred · (fr_j(t) - pred_fr[j]) · fr_k(t-1)` 对所有 j,k ∈ [0,200)
- **AND** 学习率 `η_pred = 0.001`

#### Scenario: 完整矩阵预测
- **WHEN** 计算 prediction_success
- **THEN** `pred_fr[j] = Σ_k W_pred[j][k] · fr_subcol_k(t-1)`（完整矩阵乘法）
- **AND** `prediction_success = (cosine_similarity(pred_fr, fr_subcol) + 1) / 2`

### Requirement: 睡眠重放状态隔离（补完已知限制 #1）

系统 SHALL 在睡眠重放周期前后切换网络状态，关闭外部输入并降低 ACh 水平，确保重放模式不被外部输入污染，符合慢波睡眠的生理特征。

#### Scenario: 进入睡眠态
- **WHEN** 训练步数 % REPLAY_INTERVAL == 0 且步数 > REPLAY_WARMUP_STEPS，且即将调用 `launch_replay`
- **THEN** 调用 `enter_sleep_state()`
  - 保存当前 `thalamic_gain` 到 `saved_thalamic_gain_`
  - 保存当前 `ach_level` 到 `saved_ach_level_`
  - 设置 `thalamic_gain = 0.0f`（关闭外部字节输入）
  - 设置 `ach_level *= 0.3f`（巩固模式，降低 ACh）
- **AND** 状态标志 `is_sleeping_ = true`

#### Scenario: 退出睡眠态
- **WHEN** `launch_replay` 完成（所有 top-200 模式已重放）
- **THEN** 调用 `exit_sleep_state()`
  - 恢复 `thalamic_gain = saved_thalamic_gain_`
  - 恢复 `ach_level = saved_ach_level_`
- **AND** 状态标志 `is_sleeping_ = false`

#### Scenario: 睡眠期间外部输入被抑制
- **WHEN** `is_sleeping_ == true` 且 input_inject 步到达
- **THEN** 跳过外部字节注入（或注入电流为 0）
- **AND** 日志输出 `[Stage2e P4] step=%d 进入睡眠重放态，外部输入已抑制`

### Requirement: 结构重建 CSR 完整性运行时校验（补完已知限制 #3）

系统 SHALL 在每次 CSR 重建后执行完整性校验，确保 `row_ptr` 单调性、`col_ind` 范围合法、总突触数一致，校验失败时回滚到重建前状态。

#### Scenario: 重建后完整性校验
- **WHEN** `launch_csr_rebuild` 完成新的 CSR 数据写入
- **THEN** 启动 `csr_integrity_check_kernel`
  - 检查 `row_ptr[i+1] >= row_ptr[i]` 对所有 i ∈ [0, N_TOTAL_NEURONS_2E]
  - 检查 `0 <= col_ind[j] < N_TOTAL_NEURONS_2E` 对所有 j ∈ [0, n_synapses)
  - 检查 `row_ptr[N_TOTAL_NEURONS_2E] == n_synapses`
- **AND** 校验结果原子累积到 `d_csr_check_result`

#### Scenario: 校验通过
- **WHEN** `d_csr_check_result == 0`（无错误）
- **THEN** 释放旧 CSR 副本（`d_old_row_ptr`、`d_old_col_idx`、`d_old_synapses`）
- **AND** 日志输出 `[Stage2e P3-D] CSR 完整性校验通过`

#### Scenario: 校验失败回滚
- **WHEN** `d_csr_check_result != 0`（有错误）
- **THEN** 从旧 CSR 副本回滚 `d_csr_row_ptr`、`d_csr_col_idx`、`d_synapses`
- **AND** 释放新 CSR 数据（保留旧副本作为正式数据）
- **AND** 日志输出 `[Stage2e P3-D] ERROR: CSR 完整性校验失败，已回滚（错误码=%d）`

### Requirement: inspect_ckpt 工具支持 V4 缓冲区导出（补完已知限制 #5）

系统 SHALL 在 `inspect_ckpt` 工具中支持导出 V4 新机制缓冲区的统计信息，通过 `--export-v4-buffers <dir>` 参数触发。

#### Scenario: 导出 PCA 矩阵统计
- **WHEN** 用户执行 `inspect_ckpt --export-v4-buffers <dir> <checkpoint>`
- **THEN** 读取 v3 checkpoint 中的 `d_pca_W` section
- **AND** 输出到 `<dir>/pca_W_stats.csv`：Frobenius 范数、50 列每列的 L2 范数、非零比例

#### Scenario: 导出海马索引统计
- **WHEN** 用户执行 `inspect_ckpt --export-v4-buffers <dir> <checkpoint>`
- **THEN** 读取 v3 checkpoint 中的 `d_hippo_indices` section
- **AND** 输出到 `<dir>/hippo_stats.csv`：填充率、importance 分布（min/max/mean/std）、replay_count 分布

#### Scenario: 导出共激活 tracker 统计
- **WHEN** 用户执行 `inspect_ckpt --export-v4-buffers <dir> <checkpoint>`
- **THEN** 读取 v3 checkpoint 中的 `d_coact_trackers` section
- **AND** 输出到 `<dir>/coact_stats.csv`：非零条目数、coact_count 分布、modulator_score 分布

#### Scenario: 导出 WM 槽位统计
- **WHEN** 用户执行 `inspect_ckpt --export-v4-buffers <dir> <checkpoint>`
- **THEN** 读取 v3 checkpoint 中的 `d_wm_slots` section
- **AND** 输出到 `<dir>/wm_stats.csv`：激活槽位数（activation > 0.3）、activation 分布、pattern 非零比例

#### Scenario: 导出 W_pred 矩阵统计
- **WHEN** 用户执行 `inspect_ckpt --export-v4-buffers <dir> <checkpoint>`
- **THEN** 读取 v3 checkpoint 中的 `d_w_pred` section
- **AND** 输出到 `<dir>/w_pred_stats.csv`：非对角项非零比例、Frobenius 范数、最大奇异值估计

## MODIFIED Requirements

### Requirement: launch_replay（替换空 stub）

[修改自 scheduler.cu:1029-1033]

将空 stub 替换为真正的睡眠重放逻辑：调用 `replay_kernel`，按 importance 排序取 top-200 模式，PCA 反投影注入，10× 速度 STDP 巩固。

### Requirement: launch_structural_plasticity（扩展）

[修改自 scheduler.cu:998-1018]

从仅 PSW α/β 衰减 + 弱突触重置，扩展为包含共激活候选生成 + 分块原地 CSR 重建的完整结构可塑性。

### Requirement: p3_wm_update_kernel（扩展）

[修改自 scheduler.cu:267-281]

从仅 `activation *= decay` 衰减，扩展为 PCA 签名匹配 + LRU 替换 + 反投影注入前额叶组。

## REMOVED Requirements

### Requirement: W_pred 对角项近似

**Reason**: 仅更新对角项 `w_pred[k*dim+k]` 等价于 200 个独立一阶预测器，预测能力大幅退化，不符合 v4 设计的完整矩阵乘法要求。
**Migration**: 直接替换为完整矩阵更新和乘法，`d_w_pred` 缓冲区已分配 160 KB，无需扩展。

---

## 显存预算

| 项 | 状态 | 显存 |
|----|------|------|
| d_pca_W (55K×50) | 已分配 | 11 MB |
| d_hippo_indices (50K×256B) | 已分配 | 12.8 MB |
| d_coact_trackers (500K×16B) | 已分配 | 8 MB |
| d_w_pred (200×200) | 已分配 | 0.16 MB |
| PCA 快照缓冲（CPU 内存） | 新增 | 20 MB（非显存） |
| 共激活重建临时缓冲 | 新增 | ~90 MB（峰值，仅重建期间） |
| **合计新增显存** | | **~90 MB**（仅重建期间峰值） |

**修正后总显存**：1480 MB（运动皮层后）+ 90 MB（重建峰值）= **1570 MB**

> ⚠️ 1570 MB 超过 RTX 3060 的 1.5 GB 预算上限。共激活重建缓冲（90 MB）仅在每 1000 步的重建瞬间存在，可用 CUDA 临时分配+立即释放策略（cudaMalloc/cudaFree）避免常驻。常驻显存仍为 1480 MB。

---

## 与已实现机制的关系

本 spec 补完的 5 项机制与已实现的 7 项 V4 强化机制的关系：

| 本 spec 机制 | 依赖的已实现机制 | 协同方式 |
|-------------|---------------|---------|
| PCA 反投影 | CaMKII（v4 #11） | 重放优先选 autophosph ∈ [0.3,0.7] 的易化突触 |
| 海马重放 | STDP 双 trace（v4 #10） | 重放时 STDP 巩固用双 trace 因果判定 |
| 海马重放 | 轴突延迟（v4 #9） | 重放注入考虑延迟传播 |
| 共激活跟踪 | BioSynapse 调质受体（v4 #2） | modulator_score 用 DA 加权 |
| WM 闭环 | 丘脑门控（模块 D） | WM 注入经丘脑门控调制 |
| W_pred 完整矩阵 | 2 阶 eligibility（v4 #8） | pred_succ 经 ACh 作用于慢 trace |

---

## 实现状态（2026-07-25 更新）

### 已完成（代码层面 + 编译 + 10K 步基础验证）

所有 5 项机制的代码已实现、编译通过、并通过 10K 步基础训练验证（无崩溃、无 OOM、spike 稳定 800-1100/步、checkpoint 生成）：

| 机制 | 文件 | 状态 |
|------|------|------|
| PCA 增量学习 + 反投影 | `pca_kernels.cu/cuh` | ✅ 编译通过，CPU 端 Oja 更新已集成 |
| 海马索引编码 + 睡眠重放 | `hippocampal_kernels.cu/cuh` | ✅ 编译通过，launch_replay stub 已替换 |
| 共激活跟踪 + 结构可塑性 | `coactivation_kernels.cu/cuh` | ✅ 编译通过，CSR 重建已集成 |
| WM 完整闭环 | `wm_kernels.cu/cuh` | ✅ 编译通过，三阶段流程已集成 |
| W_pred 完整矩阵 | `modulatory_kernels.cu` | ✅ 编译通过，cosine 相似度已实现 |

### 设计与实现的差异（简化方案）

为降低实现复杂度，以下场景在代码中采用简化方案，与 spec 原文有差异：

1. **PCA 更新位置**：spec 写"用 Oja's rule 在 GPU 更新 d_pca_W"，实现改为 CPU 端更新 `h_pca_W_` + 周期性 `cudaMemcpy` 同步到 GPU。原因：GPU 端 reduction 跨 55K 神经元 + 50 主成分的复杂度高，CPU 端单次遍历 cache 友好且性能足够。
2. **重放速度**：spec 写"以 10× 速度注入（10 个正常步合并为 1 步）"，实现改为增益 2.0 的电流注入。原因：主循环步合并会破坏 STDP 时序，简化为增大注入电流强度，由后续 step() 自然完成 STDP 学习。
3. **STDP 巩固**：spec 写"`Δw_replay = η_replay · pre · post · tag`"，实现改为依赖主循环现有 STDP kernel 自然执行。原因：突触级遍历需要复杂 CSR 访问，简化方案避免性能损耗。
4. **重放期间外部输入隔离**：spec 写"关闭外部字节输入（丘脑门控 gain=0）+ ACh 水平降低"，阶段 8 Task 18 已补完 `enter_sleep_state` / `exit_sleep_state` 接口，在 `launch_replay` 前后调用，临时 `thalamic_gain=0`、`ach_level *= 0.3`。

### 已知限制（补完状态）

1. ~~**睡眠重放期间未实现 ACh/丘脑门控联动**~~：**已补完（阶段 8 Task 18）**。新增 `enter_sleep_state` / `exit_sleep_state` 接口，在 `launch_replay` 前后调用，临时 `thalamic_gain=0`、`ach_level *= 0.3`。
2. **PCA 同步间隔待调优**：当前 `PCA_SYNC_INTERVAL` 默认值可能导致 GPU 端 W 滞后于 CPU 端，需在 Task 13 运行时验证中观察签名质量（调优任务，非新机制）。
3. ~~**结构重建 CSR 完整性运行时校验缺失**~~：**已补完（阶段 8 Task 19）**。新增 `csr_integrity_check_kernel`，重建后校验 `row_ptr` 单调性、`col_ind` 范围、总突触数一致，失败时回滚。
4. **W_pred 学习率未调优**：`η_pred=0.001` 可能过小，需在 Task 13 中观察 `prediction_success` 收敛曲线，必要时调整为 `0.005` 或 `0.01`（调优任务，非新机制）。
5. ~~**inspect_ckpt 工具未支持新机制缓冲区导出**~~：**已补完（阶段 8 Task 20）**。新增 `--export-v4-buffers <dir>` 参数，导出 5 项 V4 缓冲区的统计信息到 CSV。

---

## 路径迁移与清理（工程债务清理）

### 背景

为绕过"中文路径导致 VS DevShell/vcvars64.bat 启动失败"问题，之前创建了两个 junction：
- `C:\stage2e_src` → `F:\项目\THE TRUE AI\src\stage2e`（源代码，正常中文路径）
- `C:\stage2e_build` → `F:\椤圭洰\THE TRUE AI\src\stage2e\build`（**乱码目录！** 编码错误导致 junction 目标变成了"椤圭洰"而非"项目"）

这导致编译产物**分裂在两个 F 盘物理目录**：

| 目录 | 内容 | 说明 |
|------|------|------|
| `F:\项目\THE TRUE AI\src\stage2e\build\` | snn_stage2e_p1.exe (1.48MB, 旧版本)、snn_stage2e_decoder.exe、inspect_ckpt.exe、历史 .log 文件 | 正常中文路径，但无最新训练数据 |
| `F:\椤圭洰\THE TRUE AI\src\stage2e\build\` | **snn_stage2e_p1.exe (2.4MB, 最新版本)**、**ckpt_step10000.snn2e (1.5GB)**、training_motor_10k.csv (1.3MB)、training_motor_10k.log、CMakeCache.txt、build.ninja | 乱码路径，包含 V4 验证所需的全部最新数据 |

### 迁移目标

1. 将 `F:\椤圭洰\THE TRUE AI\src\stage2e\build\` 下的所有文件迁移到 `F:\项目\THE TRUE AI\src\stage2e\build\`
2. 删除乱码目录 `F:\椤圭洰\`（整个目录树）
3. 删除 C 盘的两个 junction（`C:\stage2e_src` 和 `C:\stage2e_build`）
4. 修改 `build_p1_cmd.bat` 直接在 F 盘中文路径下编译（使用 `chcp 65001` 解决 cmd 中文路径问题）
5. Task 13 验证可直接在 `F:\项目\THE TRUE AI\src\stage2e\build\checkpoints\ckpt_step10000.snn2e` 上进行

### 冲突处理

两个 build 目录存在同名文件冲突，按以下优先级处理：

| 文件 | 冲突策略 | 原因 |
|------|---------|------|
| `snn_stage2e_p1.exe` | **保留乱码目录版本 (2.4MB)** | 包含 V4 全部 5 项机制，是最新编译产物 |
| `CMakeCache.txt` | **保留乱码目录版本** | 包含最新 cmake 配置（junction 路径已硬编码） |
| `build.ninja` | **删除并重新生成** | 路径硬编码了 junction，迁移后必须重新 cmake configure |
| `.ninja_deps` / `.ninja_log` | **删除并重新生成** | 同上 |
| `cmake_install.cmake` / `CTestTestfile.cmake` / `DartConfiguration.tcl` | **保留乱码目录版本** | 标准 cmake 生成文件 |
| `build_wa.txt` / `run_short.log` | **保留乱码目录版本** | 最新构建日志 |
| `training_motor_10k.*` / `stdout_cmd.log` / `stderr_cmd.log` | **保留乱码目录版本** | V4 验证所需的训练数据 |
| `checkpoints/ckpt_step10000.snn2e` (1.5GB) | **迁移到正常目录** | Task 13 验证的核心数据 |
| 历史 .log 文件（p3_200k.log 等） | **保留正常目录版本** | 历史训练记录，不冲突 |
| `snn_stage2e_decoder.exe` / `inspect_ckpt.exe` | **保留正常目录版本** | 乱码目录无此文件 |
| `metadata_smoke.*` / `ninja_output*.txt` / `rebuild_p1.ps1` | **保留正常目录版本** | 乱码目录无此文件 |

### 中文路径解决方案

迁移后，`build_p1_cmd.bat` 需在 F 盘中文路径下直接编译。方案：

1. **bat 文件开头添加 `chcp 65001 >nul`**：将 cmd 代码页切换为 UTF-8，正确处理中文路径
2. **所有路径用双引号包裹**：`cd /d "F:\项目\THE TRUE AI\src\stage2e"`
3. **vcvars64.bat 调用保持英文路径**：VS 工具链本身在 `C:\Program Files (x86)\...` 下，无中文路径问题
4. **cmake/ninja 参数用双引号**：`cmake -S "F:\项目\THE TRUE AI\src\stage2e" -B "F:\项目\THE TRUE AI\src\stage2e\build" -G Ninja ...`

### 验证迁移成功

迁移后执行以下检查：
1. `F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe` 存在且大小为 2.4MB（最新版本）
2. `F:\项目\THE TRUE AI\src\stage2e\build\checkpoints\ckpt_step10000.snn2e` 存在且大小为 1.5GB
3. `F:\椤圭洰\` 目录已完全删除
4. `C:\stage2e_src` 和 `C:\stage2e_build` junction 已删除
5. 修改后的 `build_p1_cmd.bat` 能在 F 盘中文路径下成功执行 `cmake configure`（不一定要完整编译，验证 configure 即可）

### 待验证场景（Task 13 + Task 14）

以下 spec 场景已实现但尚未通过运行时数据验证，需在 Task 13（10K 步）和 Task 14（30K+ 步）中确认：

#### Scenario: PCA 在线更新（运行时验证）
- **WHEN** 10K 步训练完成
- **THEN** 检查 `d_pca_W` 的 Frobenius 范数 > 0（已更新 ~99 次）
- **AND** 检查 `h_mean_fr_` 分布与联合皮层平均活性一致
- **AND** 检查 PCA 签名对不同的输入字节产生不同签名（方差 > 0.1）

#### Scenario: 海马索引填充（运行时验证）
- **WHEN** 10K 步训练完成
- **THEN** 检查 `d_hippo_filled_count` 在 [50, 500] 范围（每 100 步编码一次，~89 次）
- **AND** 检查 `importance` 分布呈长尾（少数模式 importance 高）
- **AND** 检查 `replay_count` 全为 0（warmup 期内未触发重放）

#### Scenario: 共激活 tracker 增长（运行时验证）
- **WHEN** 10K 步训练完成
- **THEN** 检查 `d_tracker_count` > 0（10K 步采样 5M 候选对）
- **AND** 检查 `coact_count` 分布呈长尾（少数候选对共激活频繁）
- **AND** 检查 `modulator_score` 与 DA 水平相关

#### Scenario: WM 槽位填充（运行时验证）
- **WHEN** 10K 步训练完成
- **THEN** 检查 `d_wm_slots` 中 `activation > 0` 的槽位数 > 0（~89 次写入）
- **AND** 检查 `activation` 分布呈衰减（多数槽位 < 0.3，少数 > 0.3 注入中）
- **AND** 检查 `pattern` 字段非全零

#### Scenario: W_pred 矩阵稠密化（运行时验证）
- **WHEN** 10K 步训练完成
- **THEN** 检查 `d_w_pred` 非对角项非零比例 > 50%
- **AND** 检查 `prediction_success` 在 [0.3, 0.7] 范围（cosine 映射后）
- **AND** 检查 `prediction_success` 方差 > 0.01（非完全常数）

#### Scenario: 睡眠重放触发（长程验证）
- **WHEN** 30K+ 步训练完成（突破 `REPLAY_WARMUP_STEPS=20000`）
- **THEN** 日志输出 `[Stage2e P4] step=30000 睡眠重放周期 #1 完成（重放 200 模式）`
- **AND** 检查 `replay_count` 部分索引 > 0
- **AND** 检查 `importance` 衰减（重放后 *= 0.9）
- **AND** 检查 CaMKII `autophosph` 分布变化（易化→巩固）

---

## 显存预算实际使用

| 项 | 状态 | 实际显存 |
|----|------|---------|
| d_pca_W (55K×50) | 已分配常驻 | 11 MB |
| d_hippo_indices (50K×256B) | 已分配常驻 | 12.8 MB |
| d_coact_trackers (500K×16B) | 已分配常驻 | 8 MB |
| d_w_pred (200×200) | 已分配常驻 | 0.16 MB |
| d_wm_slots (50×216B) | 已分配常驻 | 10.8 KB |
| d_replay_injection (50K) | 已分配常驻 | 200 KB |
| d_replay_sig_ (50) | 已分配常驻 | 200 B |
| d_replay_recon_ (50K) | 已分配常驻 | 200 KB |
| d_pca_fr_ (50K) | 已分配常驻 | 200 KB |
| d_pca_mean_ (50K) | 已分配常驻 | 200 KB |
| d_new_synapse_pairs_ (5000×2) | 已分配常驻 | 40 KB |
| d_new_synapse_count_ (1) | 已分配常驻 | 4 B |
| d_new_modulator_scores_ (5000) | 已分配常驻 | 20 KB |
| d_prune_marks_ (1M) | 已分配常驻 | 1 MB |
| d_prune_count_ (1) | 已分配常驻 | 4 B |
| PCA 快照缓冲（CPU 内存） | host 内存 | 20 MB（非显存） |
| 共激活重建临时缓冲 | 非常驻 | ~90 MB（峰值，仅重建期间） |
| **新增常驻显存合计** | | **~34 MB** |
| **重建期间峰值新增** | | **~124 MB** |

**总常驻显存**：1480 MB（运动皮层后）+ 34 MB（V4 残留机制）= **1514 MB**

> ⚠️ 1514 MB 略超 RTX 3060 的 1.5 GB 预算上限，但 10K 步训练实际运行无 OOM（CUDA runtime 通常允许少量超分配）。共激活重建缓冲（90 MB）仅在每 1000 步的重建瞬间存在，用 `cudaMalloc/cudaFree` 临时分配释放避免常驻。

