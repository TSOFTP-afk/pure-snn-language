# Tasks

- [x] Task 1: 在 config.h 中新增前馈 STP 参数宏
  - [x] SubTask 1.1: 在 [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 第 163-176 行新增 `STP_U_FEEDFORWARD 0.02f`、`STP_TAU_FAC_FEEDFORWARD 200.0f`、`STP_TAU_REC_FEEDFORWARD 50.0f`、`RECEPTOR_FLAG_FEEDFORWARD 0x10` 四个宏
- [x] Task 2: 在 network_init.cu 中标记前馈连接的 STP 类型
  - [x] SubTask 2.1: `init_syn_fields` 新增 `is_feedforward` 参数（默认 false），设置 `receptor_flags` bit4（0x10）和 `utilization=STP_U_FEEDFORWARD`
  - [x] SubTask 2.2: 主循环和补足循环的 `init_syn_fields` 调用传递 `is_feedforward`
- [x] Task 3: 修改 stdp_stp_kernel 根据标志位选择 STP 参数
  - [x] SubTask 3.1: `stdp_stp_kernel` 根据 `receptor_flags & RECEPTOR_FLAG_FEEDFORWARD` 判断前馈连接，选择易化型 STP 参数
- [x] Task 4: 重新构建并跑 100K 步测试
  - [x] SubTask 4.1: 运行 `build_p1.ps1` 重新构建，确认无编译错误
  - [x] SubTask 4.2: 启动 100K 步训练，在 step 23000 停止（L2/3 仅 1 个激活，L5/L6 仍为 0）
- [x] Task 5: 验证层级联激活
  - [x] SubTask 5.1: 检查 step 10000 的 semantic eval — **L2/3=1（首次激活!），L5/L6 仍为 0**
  - [x] SubTask 5.2: 检查 `layer_delay[0..3]` — L4=0.04（正常），L2/3/L5/L6=-1（未形成级联）
  - [x] SubTask 5.3: 检查网络总活动 — 注入步 spikes=249-287（正常），非注入步 0-53
  - [x] SubTask 5.4: 检查 L4 chi2 显著性 — 100% 显著，chi2_mean 从 3159→6352（健康增长）

## 验证结论

**易化型 STP 修复方向正确且有进展**：L2/3 层从完全静默（0 个激活）→ 首次激活（1 个神经元，chi2=399.60 显著），L4 持续健康（chi2_mean 翻倍）。

**但激活范围太窄无法形成级联**：L2/3 仅 1 个神经元激活，10K-20K 步内没有增长，无法驱动 L5/L6。需要进一步增强前馈信号或调整其他参数。

# Task Dependencies

- Task 2 依赖 Task 1（先定义参数宏，但实际标记位不需要宏，可并行）
- Task 3 依赖 Task 1 和 Task 2（需要参数宏和标志位）
- Task 4 依赖 Task 3（先改代码再构建）
- Task 5 依赖 Task 4（先跑测试再验证）
