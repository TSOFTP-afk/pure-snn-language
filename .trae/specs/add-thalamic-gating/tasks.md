# Tasks

## Phase 1: 丘脑门控模块实现

- [x] Task 1: 在 config.h 中新增门控参数
  - [x] SubTask 1.1: 新增门控更新率 `GATE_UPDATE_RATE = 0.01f`（慢速更新，避免活动剧烈波动）
  - [x] SubTask 1.2: 新增门控范围 `GATE_MIN = 0.1f`、`GATE_MAX = 0.9f`（避免完全闭门导致活动归零）
  - [x] SubTask 1.3: 新增活动耦合系数 `GATE_ACTIVITY_COUP = 0.3f`（活动补偿强度）
  - [x] SubTask 1.4: 新增 novelty 耦合系数 `GATE_NOVELTY_COUP = 0.2f`（novelty 增强强度）
  - [x] SubTask 1.5: 新增活动 EMA 衰减率 `GATE_ACTIVITY_EMA_DECAY = 0.99f`（慢速活动估计）
  - [x] SubTask 1.6: 新增 novelty EMA 衰减率 `GATE_NOVELTY_EMA_DECAY = 0.95f`（novelty 慢衰减）
  - [x] SubTask 1.7: 新增初始门控信号 `GATE_INITIAL_SIGNAL = 0.5f`（半开中性初始态）
  - [x] SubTask 1.8: 新增门控更新间隔 `GATE_UPDATE_INTERVAL = 1`（每步更新，但更新率慢）

- [x] Task 2: 创建 thalamic_gate.cuh 头文件
  - [x] SubTask 2.1: 定义 `ThalamicGateState` 结构（16B：gate_signal, activity_ema, novelty_ema, _pad）
  - [x] SubTask 2.2: 声明 `init_thalamic_gate` 函数（初始化所有柱门控状态）
  - [x] SubTask 2.3: 声明 `launch_thalamic_gate_update` 函数（更新门控状态）
  - [x] SubTask 2.4: 声明 `launch_thalamic_gate_stats` 函数（统计门控指标）
  - [x] SubTask 2.5: 包含 config.h、types.h、memory_allocator.cuh

- [x] Task 3: 创建 thalamic_gate.cu 实现
  - [x] SubTask 3.1: 实现 `init_thalamic_gate_kernel`：每柱 gate_signal=GATE_INITIAL_SIGNAL, activity_ema=0, novelty_ema=0
  - [x] SubTask 3.2: 实现 `thalamic_gate_update_kernel`：
    - 输入：d_gate_states, d_column_spikes（每柱 spike count）, current_byte, d_byte_history（字节历史分布）
    - 计算 activity_norm = clamp((activity_ema - current_spikes) / max(activity_ema, 1), -1, 1)
    - 计算 novelty = 当前字节在历史中的罕见度（1 - frequency）
    - 更新 activity_ema = decay * activity_ema + (1-decay) * current_spikes
    - 更新 novelty_ema = decay * novelty_ema + (1-decay) * novelty
    - 计算 gate_target = 0.5 + GATE_ACTIVITY_COUP * activity_norm + GATE_NOVELTY_COUP * novelty_norm
    - clamp gate_target 到 [GATE_MIN, GATE_MAX]
    - gate_signal += GATE_UPDATE_RATE * (gate_target - gate_signal)
  - [x] SubTask 3.3: 实现 `thalamic_gate_stats_kernel`：统计 gate_mean, gate_open_ratio（gate_signal > 0.5 的柱比例）
  - [x] SubTask 3.4: 实现 host launcher 函数，设置 grid/block 配置

- [x] Task 4: 修改 input_encoding.cu 接入门控增益
  - [x] SubTask 4.1: 修改 `input_inject_kernel` 签名，新增 `const float* gate_signal` 参数
  - [x] SubTask 4.2: 修改 kernel 内增益计算：`float effective_gain = gain * gate_signal[col]`
  - [x] SubTask 4.3: 修改 `launch_input_inject` 签名，新增 `const float* d_gate_signal` 参数
  - [x] SubTask 4.4: 更新 input_encoding.cuh 中的函数声明
  - [x] SubTask 4.5: 确保向后兼容：若 d_gate_signal 为 nullptr，使用 gate_signal=1.0（全开，等价原行为）

- [x] Task 5: 修改 scheduler.cu 集成门控
  - [x] SubTask 5.1: 在 BioMechanismScheduler 类中新增 `ThalamicGateState* d_gate_states_` 成员
  - [x] SubTask 5.2: 在 scheduler 构造函数中分配门控缓冲区并调用 init_thalamic_gate
  - [x] SubTask 5.3: 在 step() 流水线中，input_inject 之前插入门控更新：
    - 计算 per-column spike count（从 d_spike_flags 聚合，或复用已有 d_p3_column_spikes）
    - 调用 launch_thalamic_gate_update
  - [x] SubTask 5.4: 修改 launch_input_inject 调用，传入 d_gate_states_ 的 gate_signal 数组
  - [x] SubTask 5.5: 新增门控指标统计（每 N 步调用 launch_thalamic_gate_stats）
  - [x] SubTask 5.6: 在 scheduler 析构函数中释放门控缓冲区

- [x] Task 6: 修改 main.cpp 输出门控指标
  - [x] SubTask 6.1: 在 RUN_PARAM 中输出门控参数（GATE_UPDATE_RATE, GATE_MIN, GATE_MAX 等）
  - [x] SubTask 6.2: 在 FINAL_METRIC 或定期输出中新增 gate_mean, gate_open_ratio 指标
  - [x] SubTask 6.3: 新增判据 `[22] 丘脑门控运行`：gate_mean ∈ [0.2, 0.9] 且 gate_open_ratio > 0.2

- [x] Task 7: 修改 CMakeLists.txt 添加新源文件
  - [x] SubTask 7.1: 在 stage2e 的 CMakeLists.txt 中添加 thalamic_gate.cu

- [x] Task 8: 构建验证
  - [x] SubTask 8.1: 运行 build_p1.ps1，确认无编译错误
  - [x] SubTask 8.2: 确认无编译警告
  - [x] SubTask 8.3: 确认显存占用未显著增加（门控缓冲区仅 800B，可忽略）

- [x] Task 9: 10K 烟雾测试验证
  - [x] SubTask 9.1: 运行 `snn_stage2e_p1.exe --steps 10000`
  - [x] SubTask 9.2: 验证 A（活动不回归）：
    - spike/step ∈ [50, 200]
    - `[7] 发放活动正常` PASS
  - [x] SubTask 9.3: 验证门控运行：
    - `[22] 丘脑门控运行` PASS（gate_mean ∈ [0.2, 0.9]）
    - gate_open_ratio > 0.2（至少 20% 柱门控开启）
  - [x] SubTask 9.4: 验证不破坏已有机制：
    - `[10b] PSW` PASS
    - `[10c] Ca²⁺ 回弹` PASS
    - `[17] P3 稀疏竞争` PASS
  - [x] SubTask 9.5: 记录关键指标：spike/step, gate_mean, gate_open_ratio, col_ratio, mean_fr, PSW mature

- [x] Task 10: 条件回退策略
  - [x] SubTask 10.1: 若 spike/step < 50（A 回归）：提升 GATE_MIN 至 0.3 或 GATE_INITIAL_SIGNAL 至 0.7
  - [x] SubTask 10.2: 若 gate_mean 长期 = GATE_MIN（门控卡死）：降低 GATE_ACTIVITY_COUP，避免活动补偿过强
  - [x] SubTask 10.3: 若 gate_mean 长期 = GATE_MAX（门控饱和）：提升 GATE_UPDATE_RATE 加快响应

# Task Dependencies
- Task 2 依赖 Task 1（参数定义）
- Task 3 依赖 Task 2（头文件）
- Task 4 独立，可与 Task 3 并行
- Task 5 依赖 Task 3, 4
- Task 6 依赖 Task 5
- Task 7 依赖 Task 3（新源文件）
- Task 8 依赖 Task 1-7
- Task 9 依赖 Task 8
- Task 10 条件执行，依赖 Task 9

# Notes
- 门控状态独立存储，不破坏 NeuronStateAdEx 56B 对齐
- 门控缓冲区仅 50×16B=800B，显存影响可忽略
- 门控初始半开（0.5），更新率慢（0.01），确保短期活动稳定
- novelty 计算用字节历史频率，无需复杂预测模型
- DA 耦合预留接口（当前=0），后续可接入 prediction_error
- 保持柱间输入比 GAIN_IN/GAIN_OUT 不变，门控同步缩放
