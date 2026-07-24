# Tasks

## Phase 1: 跨柱突触权重降低

- [ ] Task 1: 在 config.h 中新增跨柱突触权重范围宏
  - [ ] SubTask 1.1: 在 `config.h` 突触相关参数区（约行 47-57 附近，`SYNAPSES_PER_NEURON_2E` 之后）新增 4 个宏：
    ```cpp
    // 跨柱突触权重范围 (生物学: 柱内强连接, 跨柱弱连接, Braitenberg & Schüz 1998)
    // 调参: 原 randf(0.5,1.0)/randf(-1.0,-0.5) 与柱内同量级, 活动提升后跨柱兴奋传播抹平柱间差异
    // 降至 [0.1,0.3]/[-0.3,-0.1] 约弱 3 倍, 让柱内处理强、跨柱传播弱, 恢复柱间分化
    #define CROSS_COL_W_EXC_MIN   0.1f
    #define CROSS_COL_W_EXC_MAX   0.3f
    #define CROSS_COL_W_INH_MIN  -0.3f
    #define CROSS_COL_W_INH_MAX  -0.1f
    ```
  - [ ] SubTask 1.2: 确认宏定义位置在 `#include <cstdint>` 之后、`N_COLUMNS_2E` 之前或突触预算区附近，保持参数聚合

- [ ] Task 2: 在 network_init.cu 中修改跨柱突触权重初始化
  - [ ] SubTask 2.1: 定位行 392-397（跨柱突触权重赋值段）
  - [ ] SubTask 2.2: 将兴奋性权重从 `randf(0.5f, 1.0f) * w_scale` 改为 `randf(CROSS_COL_W_EXC_MIN, CROSS_COL_W_EXC_MAX) * w_scale`
  - [ ] SubTask 2.3: 将抑制性权重从 `randf(-1.0f, -0.5f) * w_scale` 改为 `randf(CROSS_COL_W_INH_MIN, CROSS_COL_W_INH_MAX) * w_scale`
  - [ ] SubTask 2.4: 更新行 392 注释，说明从"跨柱较弱"改为"跨柱大幅弱化（约 3 倍），抑制跨柱兴奋传播"
  - [ ] SubTask 2.5: 确认柱内突触权重（行 322-326）和前额叶投射权重（行 457-461）未被改动

- [ ] Task 3: 构建验证
  - [ ] SubTask 3.1: 运行 `build_p1.ps1`，确认无编译错误（ninja exit: 0）
  - [ ] SubTask 3.2: 确认无编译警告
  - [ ] SubTask 3.3: 确认新宏 `CROSS_COL_W_*` 被 network_init.cu 正确引用（无未定义宏错误）

- [ ] Task 4: 10K 烟雾测试验证
  - [ ] SubTask 4.1: 运行 `snn_stage2e_p1.exe --steps 10000`
  - [ ] SubTask 4.2: 验证 A（活动不回归）：
    - spike/step ∈ [50, 200]（基线 127.5，预期 ~110-115）
    - `[7] 发放活动正常 (avg > 10)` PASS
    - `[5] spike count 极差 > 100` PASS
  - [ ] SubTask 4.3: 验证 B（柱间分化提升）：
    - col_ratio > 2.0（基线 1.13，目标 > 2.0）
    - `[10d] 平衡态网络验证` PASS（CV > 0.5 且 col_ratio > 1.5）
    - `[21] P3-C 语义聚类评估` 中 col_ratio > 2.0
  - [ ] SubTask 4.4: 记录关键指标：spike/step, col_ratio, CV, mean_fr, PSW mature_ratio, silhouette, 卡方显著神经元数

- [ ] Task 5: 条件回退策略（仅在 A 回归时执行）
  - [ ] SubTask 5.1: 若 spike/step < 50（A 回归）：微调 `POP_CODING_GAIN` +10~20（如 80→90 或 80→100）补偿跨柱驱动下降
  - [ ] SubTask 5.2: 若 col_ratio 仍 < 1.5（B 未改善）：进一步降低跨柱权重至 `[0.05, 0.15]`/`[-0.15, -0.05]`（约 6 倍降低）
  - [ ] SubTask 5.3: 若 silhouette 仍停滞（≈ 0 或更负）：记录数据，本 spec 范围内不解决，建议后续 spec 评估新机制

# Task Dependencies
- Task 2 依赖 Task 1（宏定义）
- Task 3 依赖 Task 1, 2
- Task 4 依赖 Task 3
- Task 5 条件执行，依赖 Task 4 结果

# Notes
- 本 spec 仅修改跨柱突触权重，不引入新机制
- 保留前序 `boost-activity-and-column-ratio` spec 的全部参数调整（POP_CODING_GAIN=80, K=100, GAIN_OUT=0.03, INTERVAL=3, w_scale=2/√K）
- PSW、Ca²⁺ 回弹、k-WTA、STDP 等已实现机制保持不变
- 柱内突触权重和前额叶投射权重保持不变
- 预期跨柱权重降低后，偏好柱响应不再淹没其他柱，col_ratio 从 1.13 提升至 > 2.0
- 预期活动小幅下降（~11%）但仍在 [50, 200] 范围内
