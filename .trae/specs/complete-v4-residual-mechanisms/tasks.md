# Tasks

## 阶段 1：PCA 增量学习与反投影（v4 #1）

- [x] Task 1: 新建 PCA kernel 文件并实现 Oja 学习
  - [x] SubTask 1.1: 新建 `pca_kernels.cuh` 和 `pca_kernels.cu`
  - [x] SubTask 1.2: 实现 `pca_update_kernel`：Oja's rule 在线学习，输入联合皮层发放快照，更新 d_pca_W（55K×50）
  - [x] SubTask 1.3: 实现 `pca_encode_kernel`：从发放向量提取 50 维签名 `signature[k] = Σ_i W[i][k]·(fr[i]-mean)`
  - [x] SubTask 1.4: 实现 `pca_back_project_kernel`：从签名全量反投影 `reconstructed[i] = mean + Σ_k sig[k]·W[i][k]`
  - [x] SubTask 1.5: 实现 host 端 wrapper：`launch_pca_update`、`launch_pca_encode`、`launch_pca_back_project`
  - [x] 验证：单元测试 Oja 更新后 W 列正交性

- [x] Task 2: 集成 PCA 到 scheduler 训练循环
  - [x] SubTask 2.1: 在 config.h 新增 `PCA_UPDATE_INTERVAL=100`、`PCA_LEARNING_RATE=0.01f`、`PCA_WARMUP_STEPS=1000`
  - [x] SubTask 2.2: 在 scheduler.cu 新增 CPU 端 PCA 快照缓冲（20MB host 内存）
  - [x] SubTask 2.3: 每 100 步收集联合皮层发放率到 CPU，调用 `launch_pca_update_cpu`（CPU 端 Oja 更新避免 GPU 内核 reduction 复杂度）
  - [x] SubTask 2.4: 每 `PCA_SYNC_INTERVAL` 步将 CPU 端 W 用 cudaMemcpy 同步到 GPU d_pca_W（同时同步 mean_fr）
  - [x] 验证：编译通过；10K 步训练无崩溃（实际运行累计脉冲 7.5M）

## 阶段 2：海马索引编码与睡眠重放（v4 #3 + 模块 E/J）

- [x] Task 3: 新建海马 kernel 文件并实现编码逻辑
  - [x] SubTask 3.1: 新建 `hippocampal_kernels.cuh` 和 `hippocampal_kernels.cu`
  - [x] SubTask 3.2: 实现 `hippo_encode_kernel`：单 block 256 线程协作，shared memory 加载 signature[50]，分块遍历 50K 索引做 cosine 匹配，tree reduction 找全局最佳
  - [x] SubTask 3.3: 新颖模式（cosine < 0.7）单线程（tid==0）写入 LRU 槽位（避免原子竞争）
  - [x] SubTask 3.4: 已有模式刷新 `importance += 1/(1+replay_count)`
  - [x] SubTask 3.5: host wrapper `launch_hippo_encode`，每 100 步调用
  - [x] 验证：编译通过；运行时填充率待 Task 12 详细验证

- [x] Task 4: 实现睡眠重放 kernel
  - [x] SubTask 4.1: 实现 `hippo_get_top_k_kernel`：单 block K 轮 partial selection sort + selected bitmap，取 importance top-200
  - [x] SubTask 4.2: 对每个 top-K 模式用 `launch_pca_back_project` 重建发放向量
  - [x] SubTask 4.3: 实现 `replay_inject_kernel`：元素级 `d_replay_injection[i] = reconstructed[i] * REPLAY_INJECT_GAIN`（10× 速度简化为增益 2.0）
  - [x] SubTask 4.4: STDP 巩固由主循环后续 step() 自然执行（重放注入电流 → 神经元发放 → STDP kernel 学习），简化方案避免突触级遍历
  - [x] SubTask 4.5: 重放后调用 `launch_hippo_decay`：`importance *= 0.9`，`replay_count++`
  - [x] SubTask 4.6: 重放期间通过 `REPLAY_WARMUP_STEPS=20000` 保护，避免早期噪声污染；外部输入隔离由调用方在主循环中处理（当前版本未实现 ACh/丘脑门控联动，记为已知限制）
  - [x] 验证：编译通过；10K 步内未触发重放（warmup 20000），需 30K+ 步验证

- [x] Task 5: 替换 launch_replay 空 stub
  - [x] SubTask 5.1: scheduler.cu:1174-1202 替换原空 stub
  - [x] SubTask 5.2: 调用 `launch_replay_cycle` 完整流程
  - [x] SubTask 5.3: 在 config.h 新增 `REPLAY_INTERVAL=10000`、`HIPP_REPLAY_BATCH=200`、`REPLAY_INJECT_GAIN=2.0f`、`REPLAY_WARMUP_STEPS=20000`、`REPLAY_LEARNING_RATE=0.01f`
  - [x] 验证：编译通过；stub 已替换

## 阶段 3：共激活跟踪与结构可塑性（v4 #4 + 模块 K）

- [x] Task 6: 新建共激活 kernel 并实现采样
  - [x] SubTask 6.1: 新建 `coactivation_kernels.cuh` 和 `coactivation_kernels.cu`
  - [x] SubTask 6.2: 实现 `coactivation_sample_kernel`：每步采样 500 候选对，更新 d_coact_trackers
  - [x] SubTask 6.3: 共激活计数 `coact_count++`，`modulator_score += DA`
  - [x] SubTask 6.4: 实现 `coactivation_prune_kernel`：`coact_count == 0` 持续 5000 步淘汰
  - [x] SubTask 6.5: host wrapper `launch_coactivation_sample`，集成到 scheduler step()
  - [x] 验证：编译通过；运行时 tracker 增长待 Task 12 详细验证

- [x] Task 7: 实现结构可塑性批量重建
  - [x] SubTask 7.1: 实现 `structural_rebuild_kernel`：每 1000 步扫描 tracker
  - [x] SubTask 7.2: `coact_count > θ_form` 的候选对标记为新突触（top-5000）
  - [x] SubTask 7.3: `|w| < θ_prune` 且 CaMKII 未巩固的弱突触标记为修剪
  - [x] SubTask 7.4: 实现 `launch_csr_rebuild` 分块原地 CSR 重建（临时分配，重建后释放）
  - [x] SubTask 7.5: host wrapper `launch_structural_rebuild`
  - [x] 验证：编译通过；CSR 完整性校验待 Task 12 详细验证

- [x] Task 8: 替换 launch_structural_plasticity
  - [x] SubTask 8.1: scheduler.cu:1102+ 保留 PSW α/β 衰减 + 弱突触重置逻辑
  - [x] SubTask 8.2: 新增调用 `launch_structural_rebuild`（每 `STRUCTURAL_REBUILD_INTERVAL=1000` 步）
  - [x] SubTask 8.3: 在 config.h 新增 `COACT_SAMPLE_SIZE=500`、`COACT_FORM_THRESHOLD=5`、`STRUCTURAL_REBUILD_INTERVAL=1000`、`PRUNE_WEIGHT_THRESHOLD=0.05f`
  - [x] SubTask 8.4: 重建后输出日志 `[Stage2e P3-D] step=%d 结构重建 #%d: new=%d prune=%d total=%d`
  - [x] 验证：编译通过；10K 步内应触发 ~10 次重建

## 阶段 4：WM 完整闭环（v4 #5）

- [x] Task 9: 扩展 WM 实现完整闭环
  - [x] SubTask 9.1: 新建 `wm_kernels.cuh` 和 `wm_kernels.cu`（独立文件，替代 scheduler 内联 kernel）
  - [x] SubTask 9.2: 实现 `wm_write_kernel`：单 block 50 线程，每线程一个槽位，cosine 匹配 + LRU 替换
  - [x] SubTask 9.3: 实现 `wm_maintain_kernel`：单 block 50 线程，`activation *= 0.995` 衰减 + 反投影注入前额叶组
  - [x] SubTask 9.4: 保留原 `p3_wm_update_kernel` 用于 activity_drive 统计兼容
  - [x] SubTask 9.5: scheduler.cu:802+ 集成三阶段流程（写入 / 维持注入 / 兼容统计）
  - [x] SubTask 9.6: 在 config.h 新增 `WM_NOVELTY_THRESHOLD=0.7f`、`WM_INJECT_THRESHOLD=0.3f`、`WM_DECAY=0.995f`、`WM_WRITE_INTERVAL=100`
  - [x] 验证：编译通过；运行时槽位填充待 Task 12 详细验证

## 阶段 5：W_pred 完整矩阵（v4 #6）

- [x] Task 10: 升级 W_pred 从对角项到完整矩阵
  - [x] SubTask 10.1: 实现 `w_pred_predict_kernel`：完整矩阵乘法 `pred_fr[j] = Σ_k W_pred[j][k] · fr_k_prev`
  - [x] SubTask 10.2: 实现 `w_pred_update_full_kernel`：完整 `W_pred[j][k] += η · (fr_j - pred_j) · fr_k_prev`
  - [x] SubTask 10.3: 实现 `cosine_similarity_kernel`：单 block reduction 计算 `cosine_sim(pred, actual)` 映射 [0,1]
  - [x] SubTask 10.4: 修改 `launch_modulatory_update` 调用新 kernel（替换原对角项逻辑）
  - [x] SubTask 10.5: 验证防 0 除处理（冷启动首步 fr_prev=0 → pred_fr=0 → cosine 防御）
  - [x] 验证：编译通过；prediction_success 数值待 Task 12 详细验证

## 阶段 6：编译与验证

- [x] Task 11: 编译并修复错误
  - [x] SubTask 11.1: 用 junction 方式（C:\stage2e_src → f:\项目\THE TRUE AI\src\stage2e）编译 snn_stage2e_p1.exe
  - [x] SubTask 11.2: 通过 build_p1_cmd.bat 在 cmd 上下文执行 vcvars64.bat + cmake + ninja
  - [x] SubTask 11.3: 所有 16 个 .cu/.cpp 文件编译生成 .obj（含 pca/hippocampal/coactivation/wm/modulatory/scheduler）
  - [x] SubTask 11.4: snn_stage2e_p1.exe 成功生成
  - [x] 验证：编译无错误无警告；exe 存在

- [x] Task 12: 10K 步快速验证（已完成）
  - [x] SubTask 12.1: 运行 10K 步训练（training_motor_10k.log）
  - [x] SubTask 12.2: 无崩溃、无 OOM（最终步 9999，累计脉冲 7.5M，平均 756/步）
  - [x] SubTask 12.3: spike 数稳定在 800-1100 范围（合理活性）
  - [x] SubTask 12.4: checkpoint 生成（ckpt_step10000.snn2e）
  - [x] 验证：基础功能通过；子系统详细指标待 Task 13

- [ ] Task 13: 子系统运行时指标详细验证（待执行）
  - [ ] SubTask 13.1: 检查 PCA W 范数非零（10K 步后 W 应已更新 ~99 次，因 warmup=1000）
  - [ ] SubTask 13.2: 检查海马索引填充率（10K 步应编码 ~89 次，填充率 < 50K）
  - [ ] SubTask 13.3: 检查共激活 tracker 非零条目增长（10K 步采样 5M 候选对）
  - [ ] SubTask 13.4: 检查 WM 槽位填充与 activation 分布（10K 步应写入 ~89 次）
  - [ ] SubTask 13.5: 检查 W_pred 非对角项非零（10K 步更新后矩阵应稠密化）
  - [ ] SubTask 13.6: 检查结构重建日志输出（10K 步应触发 ~10 次重建）
  - [ ] SubTask 13.7: 实现 v3 checkpoint 读取工具支持新机制缓冲区（d_pca_W, d_hippo_indices, d_coact_trackers）的导出
  - [ ] 验证：所有 5 项机制有实际活动数据

## 阶段 7：路径迁移与清理（工程债务清理）

- [x] Task 15: 迁移乱码目录 `F:\椤圭洰\` 到正常目录 `F:\项目\`
  - [x] SubTask 15.1: 备份 `F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe`（旧版本 1.48MB）为 `snn_stage2e_p1_old.exe`
  - [x] SubTask 15.2: 备份 `F:\项目\THE TRUE AI\src\stage2e\build\CMakeCache.txt` 为 `CMakeCache.old.txt`
  - [x] SubTask 15.3: 将 `F:\椤圭洰\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe`（2.4MB 新版本）复制覆盖到正常目录
  - [x] SubTask 15.4: 将 `F:\椤圭洰\THE TRUE AI\src\stage2e\build\checkpoints\ckpt_step10000.snn2e`（1.5GB / 1537398188 字节）迁移到 `F:\项目\THE TRUE AI\src\stage2e\build\checkpoints\`（已创建 checkpoints 目录，1.7 秒完成）
  - [x] SubTask 15.5: 将 `F:\椤圭洰\THE TRUE AI\src\stage2e\build\` 下的 `training_motor_10k.csv`、`training_motor_10k.log`、`training_motor_10k_console.log`、`stdout_cmd.log`、`run_short.log`、`build_wa.txt` 迁移到正常目录
  - [x] SubTask 15.6: 将 `F:\椤圭洰\THE TRUE AI\src\stage2e\build\CMakeCache.txt`、`cmake_install.cmake`、`CTestTestfile.cmake`、`DartConfiguration.tcl` 迁移到正常目录（覆盖旧版本）
  - [x] SubTask 15.7: 删除正常目录下的 `build.ninja`、`.ninja_deps`、`.ninja_log`（迁移后需重新生成）
  - [x] SubTask 15.8: 删除乱码目录 `F:\椤圭洰\`（整个目录树，确认无残留）
  - [x] SubTask 15.9: 删除 C 盘 junction `C:\stage2e_src` 和 `C:\stage2e_build`（用 .NET Directory.Delete 不跟随 reparse point，不删除实际文件）
  - [x] 验证：`F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe` 大小为 2409472 字节（新编译）；`checkpoints\ckpt_step10000.snn2e` 存在且大小为 1537398188 字节

- [x] Task 16: 修改 `build_p1_cmd.bat` 直接在 F 盘中文路径下编译
  - [x] SubTask 16.1: 在 bat 文件开头添加 `chcp 65001 >nul` 切换 UTF-8 代码页
  - [x] SubTask 16.2: 将所有 `C:\stage2e_src` 替换为 `"F:\项目\THE TRUE AI\src\stage2e"`
  - [x] SubTask 16.3: 将所有 `C:\stage2e_build` 替换为 `"F:\项目\THE TRUE AI\src\stage2e\build"`
  - [x] SubTask 16.4: 删除 junction 验证逻辑（第 8-13 行）
  - [x] SubTask 16.5: cmake configure 命令改为 `cmake -S "F:\项目\THE TRUE AI\src\stage2e" -B "F:\项目\THE TRUE AI\src\stage2e\build" -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON`
  - [x] SubTask 16.6: 修改 run_train_cmd.bat 同步更新路径引用
  - [x] SubTask 16.7: 文件编码修正为 UTF-8 with BOM + CRLF 行结束符（cmd.exe 正确解析中文路径）
  - [x] 验证：执行修改后的 `build_p1_cmd.bat` cmake configure 阶段 26.6 秒成功完成

- [x] Task 17: 重新编译验证（迁移后首次在中文路径下编译）
  - [x] SubTask 17.1: 删除 `F:\项目\THE TRUE AI\src\stage2e\build\CMakeCache.txt` 和 `CMakeFiles/` 目录（清除 junction 路径残留）
  - [x] SubTask 17.2: 执行 `build_p1_cmd.bat` 完整编译（configure + build）
  - [x] SubTask 17.3: 验证编译生成的 `snn_stage2e_p1.exe` 大小约 2.4MB（2409472 字节，18/18 文件全部编译通过）
  - [x] SubTask 17.4: 执行 100 步快速 smoke test，确认中文路径下训练可正常运行（累计脉冲 40075，avg 400.8/步，显存 1528.69 MB，全部 P1 判据通过）
  - [x] 验证：中文路径下完整编译 + 训练流程闭环

- [ ] Task 14: 长程训练验证（30K+ 步，触发睡眠重放）
  - [ ] SubTask 14.1: 运行 30K+ 步训练（突破 REPLAY_WARMUP_STEPS=20000）
  - [ ] SubTask 14.2: 验证睡眠重放周期触发（日志输出 `[Stage2e P4] step=XXX 睡眠重放周期 #N 完成`）
  - [ ] SubTask 14.3: 验证重放后 importance 衰减（replay_count 增长）
  - [ ] SubTask 14.4: 验证 CaMKII autophosph 分布变化（易化→巩固）
  - [ ] SubTask 14.5: 验证重放期间外部输入已抑制（依赖 Task 18 完成）
  - [ ] 验证：睡眠重放机制功能完整

## 阶段 8：V4 已知限制补完（代码层面补完）

- [ ] Task 18: 实现睡眠重放期间 ACh/丘脑门控联动（补完已知限制 #1）
  - [ ] SubTask 18.1: 在 `scheduler.cuh` 新增私有成员 `saved_thalamic_gain_`、`saved_ach_level_`、`is_sleeping_`
  - [ ] SubTask 18.2: 在 `scheduler.cu` 实现 `enter_sleep_state()`：保存当前 thalamic_gain 和 ach_level，设置 `thalamic_gain=0`、`ach_level *= 0.3`，`is_sleeping_=true`
  - [ ] SubTask 18.3: 在 `scheduler.cu` 实现 `exit_sleep_state()`：恢复保存的 thalamic_gain 和 ach_level，`is_sleeping_=false`
  - [ ] SubTask 18.4: 在 `launch_replay` 调用前调用 `enter_sleep_state()`，调用后调用 `exit_sleep_state()`
  - [ ] SubTask 18.5: 在 input_inject 步检查 `is_sleeping_`，若为 true 则跳过外部字节注入
  - [ ] SubTask 18.6: 在 config.h 新增 `SLEEP_ACH_FACTOR=0.3f`（睡眠态 ACh 衰减系数）
  - [ ] SubTask 18.7: 添加日志输出 `[Stage2e P4] step=%d 进入睡眠重放态，外部输入已抑制` 和 `[Stage2e P4] step=%d 退出睡眠重放态，恢复外部输入`
  - [ ] 验证：编译通过；30K+ 步训练时日志输出睡眠态切换（依赖 Task 14）

- [ ] Task 19: 实现结构重建 CSR 完整性运行时校验（补完已知限制 #3）
  - [ ] SubTask 19.1: 在 `coactivation_kernels.cuh` 声明 `csr_integrity_check_kernel` 和 `launch_csr_integrity_check`
  - [ ] SubTask 19.2: 在 `coactivation_kernels.cu` 实现 `csr_integrity_check_kernel`：检查 row_ptr 单调性、col_ind 范围、row_ptr[N]==n_synapses，错误码原子累积到 d_csr_check_result
  - [ ] SubTask 19.3: 在 `launch_csr_rebuild` 中重建前保存旧 CSR 副本（d_old_row_ptr、d_old_col_idx、d_old_synapses）
  - [ ] SubTask 19.4: 重建后调用 `launch_csr_integrity_check`
  - [ ] SubTask 19.5: 校验通过（d_csr_check_result==0）：释放旧副本，输出 `[Stage2e P3-D] CSR 完整性校验通过`
  - [ ] SubTask 19.6: 校验失败（d_csr_check_result!=0）：从旧副本回滚，输出 `[Stage2e P3-D] ERROR: CSR 完整性校验失败，已回滚（错误码=%d）`
  - [ ] SubTask 19.7: 在 config.h 新增 `CSR_INTEGRITY_CHECK_ENABLED=true`（可配置开关）
  - [ ] 验证：编译通过；10K 步训练时日志输出 ~10 次校验通过

- [ ] Task 20: 实现 inspect_ckpt 工具支持 V4 缓冲区导出（补完已知限制 #5）
  - [ ] SubTask 20.1: 在 `inspect_ckpt.cpp` 新增命令行参数解析 `--export-v4-buffers <output_dir>`
  - [ ] SubTask 20.2: 使用 `CkptV3Reader::find_section()` 查找 `d_pca_W`、`d_hippo_indices`、`d_coact_trackers`、`d_wm_slots`、`d_w_pred` section
  - [ ] SubTask 20.3: 实现 PCA 矩阵统计导出：Frobenius 范数、50 列每列 L2 范数、非零比例 → `<dir>/pca_W_stats.csv`
  - [ ] SubTask 20.4: 实现海马索引统计导出：填充率、importance 分布、replay_count 分布 → `<dir>/hippo_stats.csv`
  - [ ] SubTask 20.5: 实现共激活 tracker 统计导出：非零条目数、coact_count 分布、modulator_score 分布 → `<dir>/coact_stats.csv`
  - [ ] SubTask 20.6: 实现 WM 槽位统计导出：激活槽位数、activation 分布、pattern 非零比例 → `<dir>/wm_stats.csv`
  - [ ] SubTask 20.7: 实现 W_pred 矩阵统计导出：非对角项非零比例、Frobenius 范数 → `<dir>/w_pred_stats.csv`
  - [ ] SubTask 20.8: 更新 CMakeLists.txt 确保 inspect_ckpt 编译包含新逻辑
  - [ ] 验证：编译通过；在 `ckpt_step10000.snn2e` 上执行 `inspect_ckpt --export-v4-buffers v4_export/` 成功生成 5 个 CSV 文件

---

# Task Dependencies

- Task 2 依赖 Task 1（需要 PCA kernel）
- Task 3 依赖 Task 1（海马编码需要 PCA 签名）
- Task 4 依赖 Task 1 + Task 3（重放需要 PCA 反投影 + 海马索引）
- Task 5 依赖 Task 4（替换 stub 需要重放 kernel）
- Task 7 依赖 Task 6（重建需要共激活采样）
- Task 8 依赖 Task 7（替换需要重建 kernel）
- Task 9 依赖 Task 1（WM 需要 PCA 签名 + 反投影）
- Task 11 依赖 Task 1-10 全部完成
- Task 12 依赖 Task 11
- Task 13 依赖 Task 12 + **Task 15**（需要 10K 步 checkpoint 迁移到正常路径后才能用 inspect_ckpt 读取）
- Task 14 依赖 Task 13 + Task 17 + **Task 18**（子系统验证通过 + 中文路径编译闭环 + 睡眠态隔离实现后做长程训练）
- **Task 15 依赖 Task 12**（10K 步训练已生成数据后才能迁移）
- **Task 16 依赖 Task 15**（迁移完成后才能修改 bat 路径）
- **Task 17 依赖 Task 16**（bat 修改后才能重新编译验证）
- **Task 18 依赖 Task 5**（sleep state 接口需在 launch_replay 替换后才能集成）
- **Task 19 依赖 Task 8**（CSR 校验需在 structural_rebuild 集成后才能添加）
- **Task 20 依赖 Task 17**（需在中文路径编译闭环后才能修改 inspect_ckpt 并编译）
- **Task 13.7 升级为 Task 20**（原 SubTask 13.7 的 inspect_ckpt 导出功能独立为 Task 20）

## 可并行任务

- Task 6（共激活采样）与 Task 1-5（PCA+海马）无依赖，可并行
- Task 10（W_pred 矩阵）与 Task 1-9 无依赖，可并行
- Task 9（WM 闭环）依赖 Task 1 但不依赖 Task 3-8，可与 Task 3-8 并行
- Task 13 各 SubTask 之间相互独立，可并行检查（但需 Task 15 先完成路径迁移）
- Task 15（路径迁移）与 Task 13（验证）有数据依赖但无代码依赖，可部分并行（迁移完 checkpoint 后即可开始 13.1-13.6）
- **Task 18、19、20 之间无依赖，可并行实现**（sleep state 在 scheduler.cu，CSR 校验在 coactivation_kernels.cu，inspect_ckpt 在独立工具文件）

## 已知限制补完状态

1. ~~**睡眠重放期间未实现 ACh/丘脑门控联动**~~：**Task 18 补完**（新增 enter_sleep_state / exit_sleep_state 接口）
2. **PCA 同步间隔待调优**：调优任务，非新机制，需 Task 13 运行时数据决定
3. ~~**结构重建 CSR 完整性运行时校验缺失**~~：**Task 19 补完**（新增 csr_integrity_check_kernel + 回滚机制）
4. **W_pred 学习率未调优**：调优任务，非新机制，需 Task 13 运行时数据决定
5. ~~**inspect_ckpt 工具未支持新机制缓冲区导出**~~：**Task 20 补完**（新增 --export-v4-buffers 参数）
