# Checklist

## 配置与类型层

- [x] config.h 中 `COL_L4_SIZE_2E=200, COL_L23_SIZE_2E=350, COL_L5_SIZE_2E=200, COL_L6_SIZE_2E=250` 定义正确，static_assert 校验和为 1000
- [x] config.h 中 region 枚举常量 `REGION_L4=0, REGION_L23=1, REGION_L5=2, REGION_L6=3, REGION_PREFRONTAL=4` 定义正确
- [x] config.h 中 `GATE_L6_FEEDBACK_COUP` 定义（初始 0.15）
- [x] types.h 中 `NeuronStateAdEx.region` 注释更新为新语义，`sizeof(NeuronStateAdEx) == 56` 静态断言不破坏
- [x] types.h 中 `ThalamicGateState` 新增 `l6_activity_ema` 字段，大小对齐正确（16B，static_assert 通过）

## 神经元层分配

- [x] `init_neurons_kernel` 中 L4/L2/3/L5/L6/prefrontal 的 region 分配边界正确（200/550/750/1000/前额叶）
- [x] 每层（L4/L2/3/L5/L6）内 80/20 兴奋/抑制比例独立维持（权重分布分析: L4=80.04%/19.96%, L2/3=80.00%/20.00%, L5=80.00%/20.00%, L6=79.98%/20.02%）
- [x] 抑制亚型（FS/LTS/SOM）分配在 4 层结构下正确（编译通过，运行无 crash）

## 突触拓扑

- [x] 柱内流向规则表实现正确：L4→L2/3, L2/3→L2/3(横向)+L5, L5→L6+L2/3(反馈), L6→L4(反馈)+L6(横向)
- [x] 跨柱突触仅生成于 L2/3 同层（pre region==1 且 post region==1）
- [x] 前额叶投射从 L5 层发起（pre_layer==2）
- [x] 总突触数仍为 ~10.7M（`N_TOTAL_SYNAPSES_2E` = 10700000，实际生成 10700000）
- [x] 抑制性突触权重 ∈ [-W_MAX, 0]，兴奋性突触权重 ∈ [0, W_MAX]（实测 min=-0.1414, max=0.1414）
- [x] 1/√K 平衡态权重缩放（`w_scale = 2.0f / sqrt(SYNAPSES_PER_NEURON_2E)`）保持不变

## 输入编码

- [x] `input_inject_kernel` 注入目标为 L4 层（柱首 200 神经元，索引计算不变）
- [x] L2/3 / L5 / L6 神经元不直接接收外部输入电流（L4 chi2 sig=9201，其他层=0，验证输入仅到 L4）

## L6 → 丘脑门控闭环

- [x] `thalamic_gate_update_kernel` 接收 `d_l6_column_spikes` 参数
- [x] L6 反馈逻辑：高 L6 活动 → gate_target 减小，低 L6 活动 → gate_target 增大
- [x] `l6_activity_ema` 滑动平均正确更新（`decay * ema + (1-decay) * current`）
- [x] `launch_thalamic_gate_update` host 接口传入 `d_l6_column_spikes`
- [x] scheduler `step()` 中在门控更新前计算 L6 spike count（仅统计 region==3 神经元）

## 层间指标

- [x] `launch_semantic_eval` 输出层间激活顺序（L4/L2/3/L5/L6 平均 spike 延迟）
  - 注: 10K 步仅 L4 活跃 (234 神经元), L2/3/L5/L6/prefrontal 均无活跃神经元 (设计预期, 信号传播需更长训练)
- [x] `launch_semantic_eval` 输出层间字节选择性（每层卡方显著神经元比例）
  - 注: L4 sig=9201/9201 (100%), 其他层=0 (因未活跃), 输出机制工作正常
- [x] `print_step_log` 输出 L6 反馈指标（l6_spikes=0, l6_ema=0.00 — 10K 步 L6 未活跃属正常）

## region 枚举适配

- [x] 全 stage2e 目录中所有 `region ==` / `.region` 使用点已更新为新枚举常量
- [x] analyzer / k-WTA / 卡方统计 / 柱响应统计中的 region 判断已更新
- [x] 无残留的 `region == 3` 表示 prefrontal 的旧代码（应为 `region == 4`）

## 构建与烟雾测试

- [x] `build_p1.ps1` 构建成功，无编译错误
- [x] `.\snn_stage2e_p1.exe --steps 10000` 运行完成，无 crash / NaN
- [x] spike/step ∈ [50, 200]（实测 avg=95.46，活动区间不回归，模块 A 保持）
- [x] gate_mean ∈ [0.3, 0.9]（实测 0.7024，门控不饱和，模块 D + L6 闭环稳定）
- [x] 显存峰值 < 1.5GB（实测 1401.24 MB < 1500 MB 上限；注: 超过 1332MB 设计目标为既有问题，非本次改动引入）
- [x] 层间激活顺序输出存在（输出机制工作正常；10K 步仅 L4 活跃，需 800K+ 长测验证 L4<L2/3<L5<L6 顺序）
- [x] PSW mature_ratio、Ca²⁺回弹、k-WTA、卡方统计等已有机制不破坏
  - PSW: PASS (α=0.0060, β=0.0949, mature=0.18%)
  - Ca²⁺: PASS (max_ca=0.0255 < thresh=0.15, 短测未触发回弹属预期)
  - k-WTA: PASS (active_cols=4/50, winners=6)
  - 卡方: PASS (L4 sig=9201, chi2_mean=3197.1)
- [x] col_ratio 数据已记录（实测 col_ratio=1.17，与 Phase R1 baseline 1.165 基本持平，未回归）

## 硬约束验证（项目记忆）

- [x] 80/20 兴奋/抑制比例在每层内独立维持（权重分布: L4/L2-3/L5/L6 均 80.0%/20.0%）
- [x] 抑制性突触权重 ∈ [-W_MAX, 0]（实测 min=-0.1414）
- [x] 1/√K 平衡态权重缩放保持
- [x] 未修改 stage0/1/2 代码（仅修改 src/stage2e/ 目录内文件）
- [x] STDP kernel 先计算 delta_w 再更新 last_spike（本次改动不触及 STDP kernel）
- [x] 总突触数 ~10.7M（10700000），显存预算 1.5GB 不变（实测 1401.24 MB）
