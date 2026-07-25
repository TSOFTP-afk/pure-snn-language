# Checklist

## 阶段 1：PCA 增量学习与反投影

- [x] 新建 pca_kernels.cu/cuh 文件
- [x] pca_update_kernel 实现 Oja's rule 在线学习
- [x] pca_encode_kernel 提取 50 维 PCA 签名
- [x] pca_back_project_kernel 全量反投影重建发放向量
- [x] host wrapper 函数实现（launch_pca_update/encode/back_project）
- [x] config.h 新增 PCA_UPDATE_INTERVAL=100、PCA_LEARNING_RATE=0.01f、PCA_WARMUP_STEPS=1000
- [x] scheduler.cu 集成 PCA 更新（launch_pca_update_cpu，每 100 步）
- [x] CPU 端 PCA 快照缓冲（h_pca_W_ 11MB、h_mean_fr_ 200KB、h_fr_snapshot_ 200KB、h_spike_buf_ 50KB）
- [x] 每 PCA_SYNC_INTERVAL 步 W 同步到 GPU（cudaMemcpy H2D）
- [x] compute_pca_signature 辅助函数（供海马编码和 WM 写入调用）
- [x] 编译通过（pca_kernels.cu.obj 生成）
- [ ] 训练日志输出 PCA 更新步数和 W 范数（待 Task 13 添加监控 printf）

## 阶段 2：海马索引编码与睡眠重放

- [x] 新建 hippocampal_kernels.cu/cuh 文件
- [x] hippo_encode_kernel 计算 PCA 签名并 cosine 匹配 50K 索引
- [x] 单 block 256 线程协作 + shared memory 加载 signature[50]
- [x] tree reduction 找全局最佳匹配（sim, idx）
- [x] 新颖模式（cosine < 0.7）单线程（tid==0）LRU 写入（避免原子竞争）
- [x] 已有模式 importance += 1/(1+replay_count) 刷新
- [x] 每 100 步调用 hippo_encode（由 scheduler 集成）
- [x] hippo_get_top_k_kernel：单 block K 轮 partial selection sort + selected bitmap
- [x] 取 importance top-200 模式
- [x] launch_replay_cycle：对每个 top-K 模式 PCA 反投影重建发放向量
- [x] replay_inject_kernel：元素级注入 `d_replay_injection[i] = reconstructed[i] * REPLAY_INJECT_GAIN`
- [x] 10× 速度简化为增益 2.0（避免主循环步合并复杂度）
- [x] STDP 巩固由主循环后续 step() 自然执行（简化方案）
- [x] hippo_decay_importance_kernel：重放后 importance *= 0.9，replay_count++
- [x] launch_replay 空 stub 替换为真正逻辑（scheduler.cu:1174-1202）
- [x] config.h 新增 REPLAY_INTERVAL=10000、HIPP_REPLAY_BATCH=200、REPLAY_INJECT_GAIN=2.0f、REPLAY_WARMUP_STEPS=20000
- [x] 编译通过（hippocampal_kernels.cu.obj 生成）
- [ ] 海马索引条目逐渐填充（待 Task 13 运行时验证）
- [ ] importance 分布合理（待 Task 13 运行时验证）
- [ ] 30K+ 步后睡眠重放触发（待 Task 14 长程训练）
- [ ] 重放期间 ACh/丘脑门控联动（已知限制，待后续 spec）

## 阶段 3：共激活跟踪与结构可塑性

- [x] 新建 coactivation_kernels.cu/cuh 文件
- [x] coactivation_sample_kernel 每步采样 500 候选对
- [x] coact_count++ 和 modulator_score += DA
- [x] coactivation_prune_kernel：coact_count == 0 持续 5000 步淘汰
- [x] structural_rebuild_kernel 每 1000 步扫描 tracker
- [x] coact_count > θ_form 候选对标记为新突触（top-5000）
- [x] |w| < θ_prune 且 CaMKII 未巩固的弱突触标记为修剪
- [x] launch_csr_rebuild 分块原地 CSR 重建
- [x] 临时缓冲分配后释放（非常驻）
- [x] launch_structural_plasticity 保留 PSW 衰减 + 新增重建（scheduler.cu:1102+）
- [x] config.h 新增 COACT_SAMPLE_SIZE=500、COACT_FORM_THRESHOLD=5、STRUCTURAL_REBUILD_INTERVAL=1000、PRUNE_WEIGHT_THRESHOLD=0.05f
- [x] 重建后日志输出 `[Stage2e P3-D] step=%d 结构重建 #%d: new=%d prune=%d total=%d`
- [x] 编译通过（coactivation_kernels.cu.obj 生成）
- [ ] 重建后 CSR 完整性校验通过（待 Task 13 运行时验证）
- [ ] 突触总数变化在 ±5% 内（待 Task 13 运行时验证）

## 阶段 4：WM 完整闭环

- [x] 新建 wm_kernels.cu/cuh 文件（独立文件）
- [x] wm_write_kernel：单 block 50 线程，PCA 签名 cosine 匹配 50 槽位
- [x] 新颖模式 LRU 替换（cosine < 0.7 写入最旧槽位）
- [x] 已有模式 activation 刷新为 1.0
- [x] wm_maintain_kernel：单 block 50 线程，activation *= 0.995 衰减
- [x] activation > 0.3 时 PCA 反投影注入前额叶组
- [x] 保留原 p3_wm_update_kernel 用于 activity_drive 统计兼容
- [x] scheduler.cu:802+ 集成三阶段流程（写入 / 维持注入 / 兼容统计）
- [x] config.h 新增 WM_NOVELTY_THRESHOLD=0.7f、WM_INJECT_THRESHOLD=0.3f、WM_DECAY=0.995f、WM_WRITE_INTERVAL=100
- [x] 编译通过（wm_kernels.cu.obj 生成）
- [ ] WM 槽位逐渐填充（待 Task 13 运行时验证）
- [ ] 前额叶组在 activation > 0.3 时有注入电流（待 Task 13 运行时验证）

## 阶段 5：W_pred 完整矩阵

- [x] w_pred_predict_kernel：完整矩阵乘法 `pred_fr[j] = Σ_k W_pred[j][k] · fr_k_prev`
- [x] w_pred_update_full_kernel：完整矩阵更新 `W_pred[j][k] += η · (fr_j - pred_j) · fr_k_prev`
- [x] cosine_similarity_kernel：单 block reduction 计算 cosine(pred, actual) 映射 [0,1]
- [x] launch_modulatory_update 调用新 kernel（替换原对角项逻辑）
- [x] 防 0 除处理（冷启动首步 fr_prev=0 → pred_fr=0 → cosine 防御 eps=1e-8）
- [x] 编译通过（modulatory_kernels.cu.obj 生成）
- [ ] prediction_success 数值更平滑（待 Task 13 运行时验证）
- [ ] DA 信号质量提升（待 Task 13 运行时验证）
- [ ] W_pred 非对角项非零（待 Task 13 运行时验证）

## 阶段 6：编译与验证

- [x] junction 方式编译（C:\stage2e_src → f:\项目\THE TRUE AI\src\stage2e）
- [x] build_p1_cmd.bat 在 cmd 上下文执行 vcvars64.bat + cmake + ninja
- [x] 所有 16 个 .cu/.cpp 文件编译生成 .obj
- [x] snn_stage2e_p1.exe 成功生成
- [x] 编译无错误无警告
- [x] 10K 步训练无崩溃、无 OOM（最终步 9999，累计脉冲 7.5M）
- [x] spike 数稳定在 800-1100 范围（合理活性）
- [x] checkpoint 生成（ckpt_step10000.snn2e）
- [ ] 10K 步后 PCA W 范数非零（待 Task 13）
- [ ] 10K 步后海马索引非零条目增长（待 Task 13）
- [ ] 10K 步后共激活 tracker 非零条目增长（待 Task 13）
- [ ] 10K 步后 WM 槽位有填充（待 Task 13）
- [ ] 10K 步后 W_pred 非对角项非零（待 Task 13）
- [ ] 10K 步后结构重建日志输出（待 Task 13）
- [ ] 30K+ 步后睡眠重放触发（待 Task 14）

## 显存预算验证

- [x] 常驻显存 ≤ 1500 MB（共激活重建缓冲非常驻）
- [x] d_pca_W (55K×50) = 11 MB 已分配
- [x] d_hippo_indices (50K×256B) = 12.8 MB 已分配
- [x] d_coact_trackers (500K×16B) = 8 MB 已分配
- [x] d_w_pred (200×200) = 0.16 MB 已分配
- [x] d_wm_slots (50×216B) = 10.8 KB 已分配
- [x] d_replay_injection / d_replay_sig_ / d_replay_recon_ 临时缓冲已分配
- [x] 10K 步训练无 CUDA OOM 错误
- [ ] 重建期间峰值显存 ≤ 1600 MB（待 Task 13 监控）
- [ ] 重建后临时缓冲已释放（待 Task 13 监控）

## 生物合理性验证

- [x] PCA 主成分捕获联合皮层活动主要方差（设计符合 Oja's rule）
- [x] 海马索引签名区分不同输入模式（设计符合 cosine 匹配 + LRU）
- [x] 睡眠重放优先选 importance 高的模式（设计符合 top-K 选取）
- [x] 共激活生成的新突触连接相关神经元（设计符合 coact_count 阈值）
- [x] WM 维持的槽位对应近期输入模式（设计符合 PCA 签名匹配）
- [x] W_pred 预测基于完整亚柱发放历史（设计符合 200×200 矩阵乘法）
- [ ] PCA 主成分实际捕获方差比例（待 Task 13 运行时验证）
- [ ] 海马索引签名实际区分度（待 Task 13 运行时验证）
- [ ] 睡眠重放后 CaMKII autophosph 分布变化（易化→巩固）（待 Task 14）
- [ ] 共激活生成的新突触实际连接相关神经元（待 Task 13 运行时验证）
- [ ] WM 维持的槽位实际对应近期输入模式（待 Task 13 运行时验证）
- [ ] W_pred 预测的发放与实际发放 cosine > 0.3（待 Task 13 运行时验证）

## 集成验证

- [x] scheduler.cu step() 调用顺序：input_inject → neuron_update → synapse_propagate → coactivation_sample → wm_update → pca_update → hippo_encode → structural_plasticity → replay
- [x] 所有新机制在 warmup 期内不触发（PCA_WARMUP_STEPS=1000、REPLAY_WARMUP_STEPS=20000）
- [x] checkpoint v3 格式已包含新机制缓冲区（d_pca_W, d_hippo_indices, d_coact_trackers, d_wm_slots, d_w_pred）
- [ ] inspect_ckpt 工具支持导出新机制缓冲区（待 Task 13.7）
- [ ] decoder 工具可利用 PCA 签名作为额外状态特征（待后续 spec）

## 阶段 7：路径迁移与清理

### Task 15: 迁移乱码目录
- [x] 备份正常目录下的旧版本 snn_stage2e_p1.exe（1.48MB）为 snn_stage2e_p1_old.exe
- [x] 备份正常目录下的旧 CMakeCache.txt 为 CMakeCache.old.txt
- [x] 迁移 snn_stage2e_p1.exe（2.4MB 新版本）从乱码目录到正常目录
- [x] 创建 F:\项目\THE TRUE AI\src\stage2e\build\checkpoints\ 目录
- [x] 迁移 ckpt_step10000.snn2e（1.5GB / 1537398188 字节）从乱码目录到正常目录
- [x] 迁移 training_motor_10k.csv（1.3MB）从乱码目录到正常目录
- [x] 迁移 training_motor_10k.log / training_motor_10k_console.log 从乱码目录到正常目录
- [x] 迁移 stdout_cmd.log / run_short.log / build_wa.txt 从乱码目录到正常目录
- [x] 迁移 CMakeCache.txt / cmake_install.cmake / CTestTestfile.cmake / DartConfiguration.tcl（覆盖旧版本）
- [x] 删除正常目录下的 build.ninja / .ninja_deps / .ninja_log（需重新生成）
- [x] 删除乱码目录 F:\椤圭洰\（整个目录树）
- [x] 删除 C:\stage2e_src junction（用 rmdir，不删除实际文件）
- [x] 删除 C:\stage2e_build junction（用 rmdir，不删除实际文件）
- [x] 验证 F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe 大小为 2.4MB（2409472 字节，新编译）
- [x] 验证 F:\项目\THE TRUE AI\src\stage2e\build\checkpoints\ckpt_step10000.snn2e 大小为 1.5GB（1537398188 字节）
- [x] 验证 F:\椤圭洰\ 已完全删除
- [x] 验证 C:\stage2e_src 和 C:\stage2e_build 已删除

### Task 16: 修改 build_p1_cmd.bat
- [x] bat 文件开头添加 chcp 65001 >nul
- [x] 所有 C:\stage2e_src 替换为 "F:\项目\THE TRUE AI\src\stage2e"
- [x] 所有 C:\stage2e_build 替换为 "F:\项目\THE TRUE AI\src\stage2e\build"
- [x] 删除 junction 验证逻辑（原第 8-13 行）
- [x] cmake configure 命令使用中文路径
- [x] run_train_cmd.bat 同步更新路径引用
- [x] 验证修改后的 bat 能成功执行 cmake configure（26.6s 配置完成）
- [x] 文件编码为 UTF-8 with BOM + CRLF 行结束符（cmd.exe 正确解析中文路径）

### Task 17: 重新编译验证
- [x] 删除 F:\项目\THE TRUE AI\src\stage2e\build\CMakeCache.txt
- [x] 删除 F:\项目\THE TRUE AI\src\stage2e\build\CMakeFiles\ 目录
- [x] 执行 build_p1_cmd.bat 完整编译
- [x] 编译无错误（中文路径下首次编译，18/18 文件全部通过）
- [x] snn_stage2e_p1.exe 生成，大小约 2.4MB（2409472 字节）
- [x] 100 步 smoke test 通过（中文路径下训练可运行，累计脉冲 40075，avg 400.8/步，显存 1528.69 MB）

## 阶段 8：V4 已知限制补完

### Task 18: 睡眠重放期间 ACh/丘脑门控联动
- [ ] scheduler.cuh 新增私有成员 saved_thalamic_gain_、saved_ach_level_、is_sleeping_
- [ ] enter_sleep_state() 实现：保存 thalamic_gain 和 ach_level，设置 thalamic_gain=0、ach_level *= 0.3
- [ ] exit_sleep_state() 实现：恢复保存的 thalamic_gain 和 ach_level
- [ ] launch_replay 前后调用 enter/exit_sleep_state
- [ ] input_inject 步检查 is_sleeping_，true 时跳过外部字节注入
- [ ] config.h 新增 SLEEP_ACH_FACTOR=0.3f
- [ ] 日志输出睡眠态切换（进入/退出）
- [ ] 编译通过
- [ ] 30K+ 步训练时日志输出睡眠态切换（依赖 Task 14 长程训练）

### Task 19: 结构重建 CSR 完整性运行时校验
- [ ] coactivation_kernels.cuh 声明 csr_integrity_check_kernel 和 launch_csr_integrity_check
- [ ] csr_integrity_check_kernel 实现：row_ptr 单调性 + col_ind 范围 + row_ptr[N]==n_synapses
- [ ] launch_csr_rebuild 重建前保存旧 CSR 副本
- [ ] 重建后调用 launch_csr_integrity_check
- [ ] 校验通过：释放旧副本，输出通过日志
- [ ] 校验失败：从旧副本回滚，输出错误日志（含错误码）
- [ ] config.h 新增 CSR_INTEGRITY_CHECK_ENABLED=true
- [ ] 编译通过
- [ ] 10K 步训练时日志输出 ~10 次校验通过

### Task 20: inspect_ckpt 工具支持 V4 缓冲区导出
- [ ] inspect_ckpt.cpp 新增 --export-v4-buffers <output_dir> 参数解析
- [ ] 使用 CkptV3Reader::find_section() 查找 5 个 V4 section
- [ ] PCA 矩阵统计导出（pca_W_stats.csv）：Frobenius 范数、50 列 L2 范数、非零比例
- [ ] 海马索引统计导出（hippo_stats.csv）：填充率、importance 分布、replay_count 分布
- [ ] 共激活 tracker 统计导出（coact_stats.csv）：非零条目数、coact_count 分布、modulator_score 分布
- [ ] WM 槽位统计导出（wm_stats.csv）：激活槽位数、activation 分布、pattern 非零比例
- [ ] W_pred 矩阵统计导出（w_pred_stats.csv）：非对角项非零比例、Frobenius 范数
- [ ] CMakeLists.txt 更新确保 inspect_ckpt 编译包含新逻辑
- [ ] 编译通过
- [ ] 在 ckpt_step10000.snn2e 上执行成功生成 5 个 CSV 文件
