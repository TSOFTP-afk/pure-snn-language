# DGX Spark 训练实施方案

本方案针对 `src/stage2e` 的 55K 神经元 / 10.7M 突触基线。第一目标是获得可恢复、可比较的 3M 步实验，不在首轮同时扩大网络规模。

## 1. 当前基线

- 单 CUDA 设备，默认 device 0。
- 约 1.37 GiB 持久 GPU/统一内存状态。
- 完整 checkpoint 约 1.37 GiB；默认保留最近 3 个，约 4.1 GiB。
- 真实输入为 UTF-8 字节流，每 3 个 simulation steps 注入一个字节。
- `--steps` 是绝对停止步数。恢复 800K checkpoint 并传 `--steps 3000000`，会运行 `[800000, 3000000)`。

## 2. 不可跳过的门禁

### G0：本机源码门禁

```powershell
python -m unittest discover -s src/stage2e/tests -p "test_*.py" -v
powershell -ExecutionPolicy Bypass -File src/stage2e/tests/host_syntax_check.ps1
git diff --check
```

### G1：Spark 编译门禁

通过现有 SSH 别名登录，不开放新端口，不停止现有服务：

```bash
export PATH=/usr/local/cuda/bin:$PATH
cd ~/projects/pure-snn-language
src/stage2e/build_p1.sh clean
```

必须确认：

- CMake 检测 `aarch64` host 和 CUDA 13.x；
- `sm_120` 编译、链接成功；
- `stage2e_run_config_test` 通过；
- 没有使用 `<direct.h>`、MSVC `/utf-8` 或硬编码 Windows 路径。

### G2：10K smoke

10K 不写大 checkpoint：

```bash
src/stage2e/build/snn_stage2e_p1 \
  --steps 10000 \
  --text data/lccc_sample_1mb.txt \
  --seed 42 \
  --checkpoint-interval 0
```

门禁：退出码 0、无 CUDA error/NaN/OOM、delay drop rate 在既有基线范围、核心指标能输出。
默认退出码表示运行完整性，不要求 PSW 等长时间尺度科学指标在 10K 内成熟；需要把全部科学判据作为硬门禁时显式增加 `--strict-criteria`。

### G3：checkpoint 等价性

使用短间隔专门验证恢复，不用于正式指标：

1. 连续运行到 20K；
2. 运行到 10K，保存 checkpoint，再恢复到 20K；
3. 比较 20K 的全部 `FINAL_METRIC` 和 checkpoint CRC。

任何动力学指标不一致都阻断长训。GPU 浮点非确定性存在时，必须预先定义容差，而不是事后挑选容差。

### G4：100K 历史回归

- 使用固定语料、seed 42 和相同消融模式。
- 对照仓库现有 100K 报告。
- 报告硬件、驱动、CUDA、git commit、完整参数和语料指纹。

### G5：800K 关键期

- 验证从至少两个 checkpoint 成功恢复。
- 验证 checkpoint 完整性：

```bash
python src/stage2e/tools/inspect_checkpoint.py \
  src/stage2e/checkpoints/ckpt_step800000.snn2e --verify
```

- 只有在恢复测试、磁盘余量和运行稳定性均通过后进入 3M。

### G6：3M 正式训练

```bash
src/stage2e/run_train.sh 3000000 bg
tail -f src/stage2e/training_3000000.log
```

停止时发送普通 `SIGTERM`，程序会在完成当前 step 后保存 checkpoint：

```bash
kill "$(cat src/stage2e/training.pid)"
```

不要使用 `kill -9`，除非进程已无法响应；`SIGKILL` 无法执行安全保存。

## 3. 科学对照

正式结论至少需要以下实验共享同一语料切分和评估代码：

1. Stage 2e 完整机制，至少 3 个 seed；
2. E0 纯 STDP 消融；
3. 随机打乱语料顺序；
4. 字节—柱偏好映射置换；
5. 移除手工柱偏好；
6. held-out 语料上的 next-byte 解码指标。

柱间 JS、卡方显著神经元数和输入预览不能单独作为“语义涌现”证据。解码器完成前，结论限定为网络动力学、选择性和表征分化。

## 4. Spark 运维边界

- 沿用 `cc` 现有 SSH/Tailscale 和主机指纹策略，不新增公网暴露。
- 正式测速前记录 `free -h`、`df -h`、`nvidia-smi`、当前 GPU 进程和系统负载。
- 首轮基线只需约 1.4 GiB 状态内存，但统一内存和 GPU 计算仍会与模型/ComfyUI 任务竞争；正式指标应错峰运行。
- checkpoint、日志和语料不写入模型服务目录。
- 不自动启用开机自启；稳定性门禁通过后再考虑 user-level service。

## 5. 扩展网络的顺序

完成基线 3M 后再做规模化：

1. 把柱数、每柱神经元和出度从编译期宏迁移到运行配置；
2. 使用溢出检查的 64 位计数和索引预算；
3. 删除固定 1.5 GiB 预算，改为按设备可用统一内存计算上限；
4. 先做 2×、4×、10×，每档重复 G2–G5；
5. 单机基线稳定后，再讨论多 Spark/NCCL 分区。

不要把“128 GB 可用”直接等价为“网络可扩大 100×”：突触计算量、原子冲突、checkpoint I/O 和索引范围会先于容量成为瓶颈。
