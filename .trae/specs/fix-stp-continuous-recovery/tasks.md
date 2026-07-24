# Tasks

- [x] Task 1: 修改 STP_TAU_REC_FEEDFORWARD 参数
  - [x] SubTask 1.1: 修改 [config.h](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/config.h) 第 170 行 `STP_TAU_REC_FEEDFORWARD` 从 50.0f 改为 3.0f
- [x] Task 2: 在 synapse_nmda_kernel 中增加前馈连接每步 resource 恢复
  - [x] SubTask 2.1: 修改 [synapse_kernels.cu](file:///f:/项目/THE%20TRUE%20AI/src/stage2e/synapse_kernels.cu) 的 `synapse_nmda_kernel`（第 80-86 行），在电导衰减之后增加前馈连接的每步 resource 恢复代码
- [x] Task 3: 重新构建并跑 100K 步测试
  - [x] SubTask 3.1: 运行 `build_p1.ps1` 重新构建，确认无编译错误
  - [x] SubTask 3.2: 启动 100K 步训练，**100K 步完整完成**，累计脉冲 14.6M
- [x] Task 4: 验证层级联激活
  - [x] SubTask 4.1: step 10000 semantic eval — **L2/3=16719, L5=10000, L6=10877 全部激活**
  - [x] SubTask 4.2: L2/3 激活神经元数持续增长 — **16719→17251→17329→17338→17341**
  - [x] SubTask 4.3: `layer_delay` — L4/L2/3 有延迟（0.04→1.01），L5/L6 延迟=-1（注入步活跃）
  - [x] SubTask 4.4: 网络总活动 — 注入步 247-432，非注入步 27-104（正常）
  - [x] SubTask 4.5: L4 chi2 显著性 — 100% 显著，chi2_mean 从 3150→31802（10 倍增长）

# Task Dependencies

- Task 2 依赖 Task 1（参数修改可与代码修改并行，但逻辑上先确认参数）
- Task 3 依赖 Task 1 和 Task 2（先改代码再构建）
- Task 4 依赖 Task 3（先跑测试再验证）
