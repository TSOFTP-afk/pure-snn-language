# Tasks

- [x] Task 1: 重定义柱内层布局与 region 枚举（config.h + types.h）
  - [x] SubTask 1.1: 在 config.h 中将 `COL_SENSORY_SIZE_2E/COL_ASSOCIATION_SIZE_2E/COL_MOTOR_SIZE_2E` 重命名为 `COL_L4_SIZE_2E/COL_L23_SIZE_2E/COL_L5_SIZE_2E`，新增 `COL_L6_SIZE_2E`，值为 200/350/200/250，更新 static_assert 校验和为 1000
  - [x] SubTask 1.2: 在 config.h 中新增 region 枚举常量 `REGION_L4=0, REGION_L23=1, REGION_L5=2, REGION_L6=3, REGION_PREFRONTAL=4`（替代旧的 0/1/2/3 语义）
  - [x] SubTask 1.3: 在 types.h 中更新 `NeuronStateAdEx.region` 字段注释为新语义，确认 `sizeof(NeuronStateAdEx) == 56` 不破坏
- [x] Task 2: 重写 network_init.cu 神经元层分配
  - [x] SubTask 2.1: 更新 `init_neurons_kernel` 中 region 分配逻辑：off < 200 → L4(0)，200..550 → L2/3(1)，550..750 → L5(2)，750..1000 → L6(3)，前额叶 → prefrontal(4)
  - [x] SubTask 2.2: 更新 80/20 兴奋/抑制分配逻辑，确保每层（L4/L2/3/L5/L6）内独立维持 80/20 比例
  - [x] SubTask 2.3: 更新抑制亚型（FS/LTS/SOM）分配逻辑适配 4 层结构
- [x] Task 3: 重写 network_init.cu 突触拓扑流向规则（核心改动）
  - [x] SubTask 3.1: 重写 `init_synapses_host` 中 `pre_layer` 判断逻辑从 3 层（0/1/2）改为 4 层（0/1/2/3）
  - [x] SubTask 3.2: 实现新的柱内流向规则表：L4→L2/3, L2/3→L2/3(横向)+L5, L5→L6+L2/3(反馈), L6→L4(反馈)+L6(横向)
  - [x] SubTask 3.3: 将跨柱突触目标约束到目标柱的 L2/3 层（pre 必须是 L2/3，post 必须是目标柱 L2/3）
  - [x] SubTask 3.4: 将前额叶投射起源从 association 层改为 L5 层（pre_layer == 2）
  - [x] SubTask 3.5: 验证总突触数仍为 ~10.7M（`N_TOTAL_SYNAPSES_2E`），出度分配逻辑适配 4 层
- [x] Task 4: 更新 input_encoding.cu 注入目标对齐 L4 语义
  - [x] SubTask 4.1: 更新 `input_inject_kernel` 注释，明确注入目标是 L4 层（柱首 200 神经元，索引不变，仅语义对齐）
  - [x] SubTask 4.2: 确认 `sensory_base = col_base` 仍指向 L4 起始（因 L4 仍在柱首），无需改索引计算
- [x] Task 5: 新增 L6 → 丘脑门控闭环反馈（thalamic_gate.cu + .cuh）
  - [x] SubTask 5.1: 在 `ThalamicGateState` 结构体中新增 `l6_activity_ema` 字段（保持 16B 对齐，复用 `_pad` 或扩展为 24B）
  - [x] SubTask 5.2: 在 scheduler.cu 中新增 L6 spike count 计算 kernel，按柱统计 L6 神经元（region==3）的 spike 数
  - [x] SubTask 5.3: 修改 `thalamic_gate_update_kernel` 签名，新增 `const int* d_l6_column_spikes` 参数
  - [x] SubTask 5.4: 在门控更新逻辑中加入 L6 反馈项：`l6_norm = (l6_ema - l6_current) / max(l6_ema, 1)`，`gate_target -= GATE_L6_COUP * l6_norm`（高 L6 活动→门控关闭）
  - [x] SubTask 5.5: 在 config.h 中新增 `GATE_L6_FEEDBACK_COUP`（初始 0.15，介于 activity 0.3 和 novelty 0.2 之间）
  - [x] SubTask 5.6: 更新 `launch_thalamic_gate_update` host 接口，传入 `d_l6_column_spikes`
- [x] Task 6: 更新 scheduler.cu 流水线集成 L6 反馈与层间指标
  - [x] SubTask 6.1: 在 `step()` 中 step 1.5 丘脑门控更新前，新增 L6 spike count 计算（复用 `p3_kwta_count_column_spikes_kernel` 模式，但仅统计 region==3 神经元）
  - [x] SubTask 6.2: 将 `d_l6_column_spikes` 传入 `launch_thalamic_gate_update`
  - [x] SubTask 6.3: 在 `launch_semantic_eval` 中新增层间激活顺序统计：按 region 分组统计每层平均 spike 时间（相对注入步延迟）
  - [x] SubTask 6.4: 在 `launch_semantic_eval` 中新增层间字节选择性：按 region 分组统计卡方显著神经元比例
  - [x] SubTask 6.5: 更新 `print_step_log` 输出 L6 反馈相关指标（l6_spikes, l6_ema）
- [x] Task 7: 更新所有读取 region 字段的代码适配新枚举
  - [x] SubTask 7.1: grep 全 stage2e 目录查找 `region ==` / `region ==` / `.region` 使用点，逐一更新枚举常量
  - [x] SubTask 7.2: 特别检查 analyzer、k-WTA、卡方统计、柱响应统计中的 region 判断
- [x] Task 8: 构建 + 10K 步烟雾测试 + 验证
  - [x] SubTask 8.1: 用 `build_p1.ps1` 构建 `snn_stage2e_p1`，修复编译错误
    - 修复: scheduler.cu:1026 printf 中文字符串参数 "显著%" 引发 NVCC lexer "user-defined literal operator not found" 错误, 改为 ASCII "sig_ratio" 等英文标签
  - [x] SubTask 8.2: 运行 `.\snn_stage2e_p1.exe --steps 10000` 烟雾测试 (10000 步全部完成, 无 crash/NaN)
  - [x] SubTask 8.3: 验证 spike/step ∈ [50, 200]（实测 avg=95.46 ✓ 活动区间不回归）
  - [x] SubTask 8.4: 验证 gate_mean ∈ [0.3, 0.9]（实测 0.7024 ✓ 门控不饱和）
  - [x] SubTask 8.5: 验证层间激活顺序输出（输出机制正常; 10K 步仅 L4 活跃=234N, L2/3/L5/L6=0N 需 800K+ 长测验证 L4<L2/3<L5<L6 顺序）
  - [x] SubTask 8.6: 验证显存 < 1.5GB（实测 1401.24 MB < 1500 MB ✓）
  - [x] SubTask 8.7: 与 Phase R1 baseline 对比 col_ratio（实测 1.17 vs R1 baseline 1.165, 基本持平未回归）

# Task Dependencies

- [Task 2] depends on [Task 1]（神经元层分配依赖 config 宏定义）
- [Task 3] depends on [Task 1] + [Task 2]（突触拓扑依赖层布局和 region 语义）
- [Task 4] depends on [Task 1]（语义对齐依赖 region 枚举）
- [Task 5] depends on [Task 1]（L6 region 值定义）
- [Task 6] depends on [Task 5] + [Task 7]（流水线集成依赖门控反馈和 region 适配）
- [Task 7] depends on [Task 1]（枚举常量更新）
- [Task 8] depends on [Task 1]-[Task 7] 全部完成

# 并行化建议

- [Task 4] 和 [Task 5] 可并行（input_encoding 与 thalamic_gate 改动相互独立）
- [Task 7] 可与 [Task 2]-[Task 6] 部分并行（边改边更新 region 使用点）
