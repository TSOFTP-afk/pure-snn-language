# THE TRUE AI — 纯脉冲神经网络（SNN）语言习得实验

> 一个用纯 CUDA C++ 从零实现的脉冲神经网络项目，探索**低参数成本下 SNN 能否涌现自发学习能力与语义理解**。
> 全流程零框架依赖，所有神经元模型、突触动力学、STDP、BPTT、分析工具链均从零自研。
>
> **项目正在 Stage 2e 阶段**：在 5.5×10⁴ 神经元 / 1.07×10⁷ 突触 / 50 柱规模上叠加多层级生物机制
> （AdEx / NMDA / STP / PSW / Ca²⁺回弹 / CaMKII / 丘脑门控 / 前额叶-工作记忆雏形），
> 验证"更高生物保真度能否跨越从突触结构到语义结构的鸿沟"。
> 阶段 0-2c 已证伪"纯 STDP 能产生语义级结构"的假设；Stage 2e 是对该假设的第二次试探。

许可：**CC BY 4.0**（仅署名）— 见 [LICENSE](./LICENSE)。

***

## 目录

- [项目动机](#项目动机)
- [核心结论（TL;DR）](#核心结论tldr)
- [架构与机制](#架构与机制)
- [阶段划分](#阶段划分)
- [目录结构](#目录结构)
- [依赖与编译](#依赖与编译)
- [复现指南](#复现指南)
- [实验报告索引](#实验报告索引)
- [关键数据](#关键数据)
- [诚实评估](#诚实评估)
- [致谢](#致谢)

***

## 项目动机

主流大语言模型（GPT、Claude 等）大多采用transformer架构，并且需要数千亿参数和海量数据，普通人既无设备也无资源复现。本项目从相反方向切入：**舍弃反向传播和全局梯度，采用纯脉冲神经网络（SNN）和局部学习规则（STDP），探索在低参数成本（仅 10,000 神经元、1,000,000 突触）下，能否涌现出自发学习能力乃至语义理解。**

实验产出了部分成功——纯 STDP 确实让突触权重形成了双峰分布，网络自发实现了稀疏化和兴奋/抑制平衡。但碍于创作者现实中的个人能力与设备限制（单卡 RTX 3060，仅能承载 10⁴ 神经元规模），所有实验研究只能进行到这里。

**现在她还不是一个完成的果实，而是一个需要被继续的探索方向。** 希望未来有人能在更大规模、更丰富机制的条件下延续这个方向——验证 SNN 在合理规模下能否跨越从"突触结构"到"语义结构"的鸿沟。

创作者没有希望项目能超越transformer架构但这是一次**对纯局部学习规则能力边界的科学探查**——并且它远未结束。

***

## 核心结论（TL;DR）

### Stage 0-2c（10⁴ 神经元 / 10⁶ 突触 / 1M 步纯 STDP）

#### ✓ 纯 STDP 能学到的

| 能力                         | 数据                               |
| -------------------------- | -------------------------------- |
| 突触权重的双峰化（LTP 强化 + LTD 修剪）  | 饱和突触 24.6%→46.4%，零权重 14.6%→36.2% |
| 网络稀疏化（homeostatic 控制活跃度）   | 活跃神经元 100%→61%                   |
| 兴奋/抑制平衡（生物合理的 50/50）       | 从 68/32 修正到 50/50                |
| 基本信号传播（sensory→motor 路径打通） | spikes/step 3,309→2,024          |

#### ✗ 纯 STDP 学不到的

| 能力      | 数据                                        |
| ------- | ----------------------------------------- |
| 字节级语义映射 | max\_chi² = 58.7（临界值 123.2，0/6,073 神经元显著） |
| 功能柱自发分化 | 簇退化（4 大簇 + 微簇噪声）                          |
| 字节选择性响应 | 0/6,073 神经元通过卡方检验                         |
| 幂律发放分布  | α = 0.14（目标 \[1.5, 3.0]），R² = 0.02        |

### Stage 2e（5.5×10⁴ 神经元 / 1.07×10⁷ 突触 / 50 柱 + 多机制）

#### ✓ Stage 2e P1 已达成

| 能力 | 数据 |
| --- | --- |
| 活动区间校准（模块 A） | spike/step 24.8 → 95.48（进入目标 [50,200]） |
| 丘脑-皮层门控运行（模块 D） | gate_mean=0.7024，gate_open_ratio=1.0，活动补偿与 novelty 增强生效 |
| 现有机制不破坏 | PSW（mature=2.5%）、Ca²⁺回弹、k-WTA、卡方（9202 显著）、语义准备度 0.85 全部 PASS |
| 显存预算 | 1401 MB / 1.5 GB（余量 ~100 MB） |

#### ⏳ Stage 2e P1 待解（项目级阶段目标 A+B 未完全达成）

| 问题 | 当前数据 | 根因 |
| --- | --- | --- |
| 柱间分化（B）恶化 | col_ratio 1.33 → 1.17 | 丘脑门控同步缩放输入增益，加剧跨柱信号平均化 |
| 跨柱权重过强 | 跨柱/柱内权重比 ≈ 1.0 | 不符合生物皮层"局部强、远程弱"小世界拓扑 |

#### 📋 Stage 2e 下一步（按 [人脑差距模块评估](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md) 路线图）

1. **Phase R1**：跨柱权重三档消融（Baseline [0.5,1.0] / Weak [0.1,0.3] / VeryWeak [0.05,0.15]），失败后停止搜索转向假设审查
2. **Phase R2**：引入真实皮层层级 L4 / L2-3 / L5-6
3. **Phase R3**：完善丘脑-皮层门控（已部分完成）
4. **Phase R4**：建立海马-皮层重放闭环
5. **Phase R5**：建立 next-byte / next-token 输出解码

**这是真实科学结论**：纯局部学习规则在合理规模下**能学到突触级、网络级结构**，但**学不到语义级结构**。
Stage 2e 通过叠加多层级生物机制正试探"机制保真度提升能否跨越这一鸿沟"。
详见 [实验报告](#实验报告索引)。

***

## 架构与机制

### 神经元模型：LIF（Leaky Integrate-and-Fire）

```
V[t+1] = β·V[t] + α·σ(S[t]) - V_reset·Spikes[t] + I_ext[t]
S[t+1] = σ(S[t])·(1 - Spikes[t]) + (1-σ(S[t]))·decay
Spikes[t] = 1  if V[t] ≥ θ  (and refractory period elapsed)
         = 0  otherwise
```

参数：β=0.95, θ=1.0, V\_reset=0, refractory=2 步

### 突触模型：CSR 稀疏格式

每个突触 32 字节，包含：

- `weight` (float, 4B) — 当前权重
- `pre_idx` / `post_idx` (int, 8B) — 前后突触神经元索引
- `last_pre_spike` / `last_post_spike` (long, 16B) — 上次发放时间戳（STDP 用）
- `eligibility` (float, 4B) — 资格迹（dopamine 调制用）

兴奋性突触 clamp 到 \[0, W\_MAX]，抑制性突触 clamp 到 \[-W\_MAX, 0]（避免权重归零的关键约束）。

### 学习规则

#### 1. STDP（脉冲时序依赖可塑性）

```
Δw = A+·exp(-(t_post - t_pre)/τ+)   if t_post > t_pre   (LTP)
   = -A-·exp(-(t_pre - t_post)/τ-)  if t_pre > t_post    (LTD)
```

参数：A± = 0.05, τ± = 20 步

**关键实现细节**：必须**先计算 Δw 再更新 last\_pre\_spike / last\_post\_spike**，否则 LTP 永远不触发（曾因此让所有突触单调归零）。

#### 2. Eligibility Trace + Dopamine 调制

```
eligibility[t] = eligibility[t-1]·λ + STDP_event
Δw[t] = dopamine[t]·eligibility[t]
```

dopamine 默认 1.0（不调制）。奖励机制降级为 B3 消融实验选项，不在主开发路径。

#### 3. Homeostatic（稳态可塑性）

每个神经元的发放阈值 θ 动态调节：

```
fire_rate[neuron] = sliding average of spikes
threshold_offset[neuron] += LR·(fire_rate - target_fr)
effective_threshold = θ + threshold_offset
```

不同脑区目标发放率：sensory/association = 5 Hz, motor = 30 Hz。

### 网络拓扑

#### Stage 0：三层均匀随机稀疏

- 10,000 神经元 = 2,000 sensory + 6,000 association + 2,000 motor
- 1,000,000 突触，每个神经元 100 个（80% 兴奋 + 20% 抑制）
- 均匀随机连接

**问题**：信号被中心极限定理稀释，motor 神经元对所有输入响应相同。

#### Stage 2A：柱内三层流水线（生物合理的皮层柱）

- 10 柱 × 1000 神经元/柱
- 每柱内 = 200 sensory + 600 association + 200 motor（三层流水线）
- intra-column 突触（同柱内）：p = 0.1，每个神经元 80 个
- inter-column 突触（跨柱）：p = 0.005，每个神经元 20 个
- 信号在柱内传播**无稀释**（vs stage0 的两次 20% 稀释）

```
柱 c 神经元布局:
  [c*1000,        c*1000 + 200)    = 柱 c 的 sensory 层
  [c*1000 + 200,  c*1000 + 800)    = 柱 c 的 association 层
  [c*1000 + 800,  c*1000 + 1000)   = 柱 c 的 motor 层
```

### 输入编码：8-bit UTF-8 字节流

UTF-8 字节流（256 维）→ one-hot 注入到柱 0 的 sensory 层前 256 个神经元。

支持中文（每个汉字 3 字节 UTF-8）。曾尝试 7-bit ASCII 编码，但排除了所有非英文文本，已废弃。

### Stage 2e 多层级生物机制（v4 设计）

> 设计文档：[docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md](./docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md)

在 RTX 3060 6GB 硬件约束下，将显存预算推至 1.5GB，规模升级到 **5.5×10⁴ 神经元 / 1.07×10⁷ 突触 / 50 柱**，按时间尺度分层引入生物机制：

| 时间尺度 | 机制 | 实现状态 |
| --- | --- | --- |
| ~1ms | AdEx 神经元（替代 LIF）、NMDA 受体、STP 短期可塑性、轴突延迟 | ✅ Phase 1 |
| ~10ms | STDP 双 trace、PSW 概率突触权重（贝叶斯 STDP）、Ca²⁺ 回弹 LTD、CaMKII 分子巩固、k-WTA 柱内竞争 | ✅ Phase 1 |
| ~100ms | 神经调质受体（DA/ACh/NE/5HT，突触级）、3 种抑制亚型（FS/LTS/SOM） | ✅ Phase 1 |
| ~1s | 丘脑-皮层门控（活动补偿 + novelty 增强） | ✅ 模块 D |
| ~10s | 结构可塑性、发育阶段调度（5 阶段 3M 步） | ⏳ 设计完成待实施 |
| ~1min | 海马重放、睡眠巩固 | ⏳ 雏形 |
| 长时 | 前额叶-工作记忆（5K 神经元 / 50 槽）、PCA 反投影 | ⏳ 雏形 |

#### Stage 2e 关键防饱和机制

PSW（Probabilistic Synaptic Weights）以 Beta(α,β) 分布维护权重，从数学结构上消除权重饱和：LTP 累积 α（"成功"证据），LTD 累积 β（"失败"证据），w_eff = W_MAX·α/(α+β)。
配合 Ca²⁺ 回弹 LTD（高频 Ca²⁺ 超载触发主动削弱）与 CaMKII 自磷酸化巩固，构成三重防饱和路径。

#### Stage 2e 柱拓扑

- 50 柱 × 1000 神经元（柱内：200 sensory + 600 association + 200 motor）
- 80% 兴奋 / 20% 抑制（FS / LTS / SOM 三亚型）
- 柱内突触：p=0.1，权重 |w| ∈ [0.057, 0.143]（1/√K 平衡态缩放）
- 跨柱突触：p=0.005，权重 |w| ∈ [0.014, 0.042]（**Phase R1 消融中**）
- 独立前额叶：5000 神经元（50 组 × 100），承担工作记忆

### 训练数据：LCCC-base

- 清华+三星 2020 年发布的中文对话语料
- 829 MB，2,016 万对话轮次
- 下载地址：<https://github.com/thu-coai/CDial-GPT>

***

## 阶段划分

| 阶段              | 目标                                     |  状态 | 结果                                                |
| --------------- | -------------------------------------- | :-: | ------------------------------------------------- |
| **Stage 0**     | 三层 SNN 管线 MVP（STDP + 奖励 + homeostatic） |  ✓  | 跑通 30.83 MB 显存的完整训练管线，meanW 0.09→0.318            |
| **Stage 1a**    | BPTT + 代理梯度（从零实现）                      |  ✓  | 梯度检查 20/20 通过，max rel\_err 1.5e-6                 |
| **Stage 1b**    | 字符自编码器                                 |  ✓  | 32/32 round-trip fidelity 100%，real\_loss 2.0→0.0 |
| **Stage 2a**    | 柱拓扑 + 数据流（10⁴ 神经元 + LCCC）              |  ✓  | 10k 步 smoke test 通过，柱拓扑正确生成                       |
| **Stage 2b**    | 1M 步无监督 STDP 训练                        |  ✓  | meanW 0.09→0.318，相变发生                             |
| **Stage 2c**    | 结构分析（PCA / K-means / 卡方 / 幂律）          |  ✓  | 4 个判定标准 0/4 达成，纯 STDP 学不到字节映射                     |
| **Stage 2d-v1** | P0+P1（弱化 homeostatic + one-hot 编码）     |  ✓  | 短训练有效，长训练回退                                       |
| **Stage 2d-v2** | + P2（k-WTA 柱间竞争）                       |  ✓  | 稀疏化成功，但卡方仍 0 显著（手工机制贡献假阳性）                        |
| **Stage 2A**    | 柱内三层流水线（方案 A 结构重构）                     |  ✓  | max\_chi² +41%，但仍是手工机制贡献                          |
| **Pure-SNN**    | 移除所有手工机制，跑"纯 SNN"实验                    |  ✓  | **真实能力边界**：STDP 学到突触结构，学不到语义结构                    |
| **Stage 2e-P1** | 快时间尺度生物机制（AdEx/NMDA/STP/PSW/Ca²⁺/CaMKII/双trace） |  ✓  | 10K 烟雾测试通过，显存 1401 MB / 1.5 GB，机制全部 PASS |
| **Stage 2e-A**  | 活动区间校准（模块 A） |  ✓  | spike/step 24.8 → 95.48（进入 [50,200]） |
| **Stage 2e-D**  | 丘脑-皮层门控（模块 D） |  ✓  | gate_mean=0.7024，活动补偿与 novelty 增强生效，已有机制不破坏 |
| **Stage 2e-B**  | 柱内强/跨柱弱拓扑（模块 B） | ⏳ | Phase R1 三档消融进行中（Weak 已实施待测，失败转 VeryWeak） |
| **Stage 2e-R2~R5** | 真实皮层层级 / 海马重放 / 输出解码 | ❌ | 未启动，见 [人脑差距模块评估](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md) |

***

## 目录结构

```
THE TRUE AI/
├── LICENSE                           # CC BY 4.0
├── README.md                         # 本文件
├── .gitignore
│
├── src/
│   ├── include/                      # 公共头文件
│   │   ├── config.h                  # 全局配置（神经元/突触数、学习率等）
│   │   ├── types.h                   # BrainRegion 枚举、get_region()
│   │   ├── neuron.cuh                # LIF 神经元参数
│   │   ├── synapse.cuh               # Synapse 结构 + sync_weights()
│   │   ├── stdp.cuh                  # STDP 参数
│   │   ├── network.h                 # SNNNetwork 类（含 stage2 getter）
│   │   ├── network.cuh               # 网络内部接口
│   │   ├── io.cuh                    # NetworkStats + compute_stats()
│   │   └── trainer.h                 # stage0 trainer 接口
│   │
│   ├── cuda/                         # CUDA kernels（stage0 核心）
│   │   ├── neuron_kernel.cu          # LIF 更新 + homeostatic
│   │   ├── stdp_kernel.cu            # STDP + eligibility trace
│   │   ├── synapse_kernel.cu         # 突触传播 + 权重同步
│   │   ├── io_kernel.cu              # 统计计算（meanFR / meanW）
│   │   └── network_init.cu           # stage0 均匀随机拓扑（已被 stage2 替换）
│   │
│   ├── host/                         # Host 包装层（stage0）
│   │   ├── network.cpp               # SNNNetwork 类实现
│   │   ├── io.cpp                    # 权重保存/加载
│   │   ├── monitor.cpp               # 训练监控
│   │   ├── trainer.cpp               # stage0 训练器（stage2 不链接）
│   │   └── main.cpp                  # stage0 入口（stage2 不链接）
│   │
│   ├── stage1/                       # Stage 1: BPTT 实验（独立代码库）
│   │   ├── CMakeLists.txt
│   │   ├── bptt_kernels.cu/.cuh      # BPTT 前向/反向 kernel
│   │   ├── bptt_demo.cu              # 梯度检查 demo
│   │   ├── text_codec.cu/.cuh        # 5-bit 字符编码
│   │   └── main.cpp                  # 字符自编码器
│   │
│   ├── stage2/                       # Stage 2: 联合皮层发育
│   │   ├── CMakeLists.txt            # 构建 snn_stage2 + snn_stage2_analyze
│   │   ├── config.h                  # stage2 配置（8-bit 编码、柱参数）
│   │   ├── columnar_topology.cu/.cuh # 柱拓扑生成器
│   │   ├── text_codec_ext.cu/.cuh    # 8-bit UTF-8 字节编码
│   │   ├── text_stream.cu/.cuh       # LCCC 语料流式读取
│   │   ├── unsupervised_trainer.cu/.cuh # 训练循环 + checkpoint I/O
│   │   ├── competition.cu/.cuh       # k-WTA 柱间竞争（手工机制）
│   │   ├── analyzer.cu/.cuh          # PCA / K-means / 卡方 / 幂律 / silhouette
│   │   ├── analyze_main.cpp          # 分析入口（--ckpt / --random / --csv）
│   │   ├── main.cpp                  # 训练入口（--steps / --text / --ckpt）
│   │   └── preprocess_lccc.py        # LCCC JSON→纯文本预处理脚本
│   │
│   ├── stage2e/                      # Stage 2e: 多层级生物机制增强 v4
│   │   ├── CMakeLists.txt            # 构建 snn_stage2e_p1
│   │   ├── config.h                  # v4 全参数（55K 神经元 / 10.7M 突触 / 50 柱）
│   │   ├── types.h                   # AdEx / BioSynapse / 抑制亚型等扩展类型
│   │   ├── memory_allocator.cu/.cuh  # 1.33GB GPU 缓冲池
│   │   ├── network_init.cu/.cuh      # 50 柱拓扑 + 1/√K 平衡态缩放 + PSW 初始化
│   │   ├── neuron_kernels.cu/.cuh    # AdEx + 适应性 + 阈值动态
│   │   ├── synapse_kernels.cu/.cuh   # NMDA/AMPA/GABA + STP + 延迟队列 + STDP 双trace
│   │   ├── input_encoding.cu/.cuh    # 群体编码 + 柱字节偏好 + 门控增益
│   │   ├── thalamic_gate.cu/.cuh     # 丘脑-皮层门控（模块 D）
│   │   ├── modulatory_kernels.cu/.cuh # DA/ACh/NE/5HT + PSW + Ca²⁺回弹 + CaMKII
│   │   ├── scheduler.cu/.cuh         # v4 多时间尺度流水线调度
│   │   ├── main.cpp                  # P1 烟雾测试入口（--steps / --csv / --e0）
│   │   ├── analyze_burst.py          # 簇状发放分析
│   │   ├── analyze_profile.py        # 显存/性能 profile 分析
│   │   ├── build_p1.ps1              # P1 构建脚本
│   │   └── p1_profile.csv            # P1 烟雾测试 profile 数据
│   │
│   └── scripts/
│       ├── build.ps1                 # 构建脚本
│       └── run.ps1                   # 运行脚本
│
├── data/                             # 实验数据（部分，大文件见下载说明）
│   ├── lccc_sample_1mb.txt           # LCCC 1MB 样本（用于分析）
│   ├── stage2_pure_1M_log.txt        # 纯 SNN 训练日志
│   ├── stage2_pure_1M_analyze_log.txt
│   ├── stage2_pure_1M_clusters.csv   # PCA 坐标（10,000 点）
│   ├── stage2_b1_random_*.txt/.csv   # B1 随机基线
│   ├── stage2A_1M_*.txt/.csv         # 方案 A 结果
│   ├── stage2d_v2_1M_*.txt/.csv      # 2d-v2 结果
│   └── (检查点 *.bin 不归档，见下载说明)
│
├── 全量神经元模拟对话智能-工程落地方案.md  # 项目最初规划
├── 全量神经元模拟对话智能-总纲.md            # 新方向奠基性总纲（结构同构假设）
├── 阶段2-联合皮层发育-实施规划.md          # Stage 2 实施规划 v3
├── stage2b-训练结果分析报告.md             # 2b 训练结果分析
├── stage2c-结构分析报告.md                 # 2c 结构分析报告
├── 人类脑差距评估.md                        # 与人脑的规模/机制/功能差距评估
├── docs/superpowers/specs/                # 设计与 spec 文档
│   ├── 2026-07-19-bio-mechanisms-design.md       # Stage 2e 多层级生物机制设计 v4
│   └── 2026-07-24-human-brain-gap-module-assessment.md  # 人脑差距模块评估 + 路线图
└── .trae/specs/                           # 已实施 spec（add-thalamic-gating / boost-activity-and-column-ratio / boost-column-differentiation / fix-architectural-issues）
```

***

## 依赖与编译

### 硬件要求

- NVIDIA GPU，compute capability ≥ 8.6（RTX 30/40/A系列）
- 显存 ≥ 6 GB（用于 10k 神经元 / 1M 突触）
- 内存 ≥ 16 GB（用于 829MB LCCC 语料加载）

### 软件依赖

- **CUDA Toolkit 13.x**（13.3 测试通过）
- **Visual Studio 2022 Build Tools**（MSVC v143, x64）
- **CMake ≥ 3.18**
- **Ninja**（推荐，比 MSBuild 快 3-5×）

> ⚠️ **中文路径注意**：CUDA 13.3 在中文路径下需要用 x64 cl.exe（不能用默认的 x86）。
> 启动 VS DevShell 时必须加 `-HostArch amd64 -Arch amd64` 参数，否则 cudafe++ 会崩溃。

### 编译步骤

```powershell
# 1. 启动 x64 VS DevShell
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation `
    -DevCmdArguments "-arch=x64 -host_arch=x64"

# 2. 构建 stage2
cd "f:\项目\THE TRUE AI\src\stage2"
mkdir build; cd build
cmake -G Ninja ..
ninja snn_stage2 snn_stage2_analyze

# 3. 验证
.\snn_stage2.exe --help
.\snn_stage2_analyze.exe --help
```

### 构建 Stage 2e（多机制平台）

```powershell
cd "f:\项目\THE TRUE AI\src\stage2e"
.\build_p1.ps1     # 自动配置 x64 VS DevShell + cmake -G Ninja + ninja snn_stage2e_p1
# 或手动:
mkdir build; cd build
cmake -G Ninja ..
ninja snn_stage2e_p1

# P1 烟雾测试（10K 步）
.\snn_stage2e_p1.exe --steps 10000
# 评估模式（输出 spike 序列 CSV）
.\snn_stage2e_p1.exe --steps 10000 --csv p1_profile.csv
# E0 消融基线（纯 STDP，关闭三因素+CaMKII+调质）
.\snn_stage2e_p1.exe --steps 10000 --e0
```

***

## 复现指南

### 1. 准备 LCCC 语料

```powershell
# 从 https://github.com/thu-coai/CDial-GPT 下载 LCCC-base-split.ZIP
# 解压后用预处理脚本转纯文本：
python src\stage2\preprocess_lccc.py `
    --input  path\to\LCCC-base-split\train.json `
    --output data\lccc_base.txt

# 生成 1MB 样本用于分析
$bytes = [System.IO.File]::ReadAllBytes("data\lccc_base.txt")[0..1048575]
[System.IO.File]::WriteAllBytes("data\lccc_sample_1mb.txt", $bytes)
```

### 2. 训练 1M 步

```powershell
.\snn_stage2.exe `
    --steps 1000000 `
    --ckpt  data\stage2_pure_1M_ckpt.bin `
    --text  data\lccc_base.txt `
    --quiet
# 预计耗时：~37 分钟（RTX 4060）
```

### 3. 结构分析

```powershell
.\snn_stage2_analyze.exe `
    --ckpt  data\stage2_pure_1M_ckpt.bin `
    --text  data\lccc_sample_1mb.txt `
    --steps 10000 `
    --label "Pure-SNN-1M" `
    --csv   data\stage2_pure_1M_clusters.csv
```

### 4. 随机基线对比

```powershell
.\snn_stage2_analyze.exe `
    --random `
    --text  data\lccc_sample_1mb.txt `
    --steps 10000 `
    --label "B1-Random" `
    --csv   data\stage2_b1_random_clusters.csv
```

### 5. 命令行参数

`snn_stage2.exe`（训练）：

- `--steps N` — 训练步数（默认 10000）
- `--text PATH` — 输入语料路径
- `--ckpt PATH` — checkpoint 保存/加载路径
- `--resume` — 从 checkpoint 恢复训练
- `--quiet` — 减少日志输出
- `--seed N` — 随机种子
- `--help` — 显示帮助

`snn_stage2_analyze.exe`（分析）：

- `--ckpt PATH` — 加载训练后网络
- `--random` — 生成随机初始化网络（B1 基线）
- `--text PATH` — 测试语料
- `--steps N` — 分析步数（默认 10000）
- `--label NAME` — 实验标签
- `--csv PATH` — 输出 PCA + 聚类坐标 CSV

***

## 实验报告索引

| 文档                                               | 内容                                      |
| ------------------------------------------------ | --------------------------------------- |
| [全量神经元模拟对话智能-总纲.md](./全量神经元模拟对话智能-总纲.md)         | **新方向奠基性总纲**：结构同构假设、SNN 全量模拟、对话主目标、阶段路线图与停损点 |
| [全量神经元模拟对话智能-工程落地方案.md](./全量神经元模拟对话智能-工程落地方案.md) | 项目最初规划（包含 stage0/1/2/3 的整体设计）           |
| [阶段2-联合皮层发育-实施规划.md](./阶段2-联合皮层发育-实施规划.md)       | Stage 2 详细实施规划 v3（含 2a/2b/2c/2d 子阶段）    |
| [stage2b-训练结果分析报告.md](./stage2b-训练结果分析报告.md)     | 1M 步训练结果分析（指标定义、演化轨迹、5 大关键发现）           |
| [stage2c-结构分析报告.md](./stage2c-结构分析报告.md)         | 结构分析报告（PCA / K-means / 卡方 / 幂律，4 个判定标准） |
| [人类脑差距评估.md](./人类脑差距评估.md)                       | 与人脑的规模 / 机制 / 功能 / 物理实现差距评估（10-15 个数量级） |
| [docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md](./docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md) | **Stage 2e 多层级生物机制设计 v4**（AdEx/NMDA/PSW/CaMKII/丘脑门控/前额叶等） |
| [docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md) | **人脑差距模块评估**：14 个模块评级 + Phase R1-R5 路线图 + 防止无限调参硬规则 |
| [.trae/specs/add-thalamic-gating/](./.trae/specs/add-thalamic-gating/) | 模块 D 丘脑-皮层门控 spec（已实施 ✓） |
| [.trae/specs/boost-activity-and-column-ratio/](./.trae/specs/boost-activity-and-column-ratio/) | 模块 A 活动区间校准 spec（已实施 ✓） |
| [.trae/specs/boost-column-differentiation/](./.trae/specs/boost-column-differentiation/) | 模块 B 跨柱权重消融 spec（Phase R1 进行中 ⏳） |
| [.trae/specs/fix-architectural-issues/](./.trae/specs/fix-architectural-issues/) | 1/√K 权重缩放 + 柱特异性输入 spec（已实施 ✓） |

***

## 关键数据

### Stage 0-2c 训练资源消耗

| 维度     | 值                         |
| ------ | ------------------------- |
| 神经元数   | 10,000                    |
| 突触数    | 1,000,000                 |
| GPU 显存 | 38 MB（网络）+ 6 GB（kernels）  |
| 内存     | \~1 GB（LCCC 加载）           |
| 训练速度   | \~471 steps/sec（RTX 4060） |
| 1M 步耗时 | 35-37 分钟                  |

### Stage 2e 训练资源消耗

| 维度     | 值                         |
| ------ | ------------------------- |
| 神经元数   | 55,000（50K 联合皮层 + 5K 前额叶） |
| 突触数    | 10,700,000                |
| 皮层柱数   | 50                        |
| GPU 显存 | 1401 MB / 1.5 GB 预算（RTX 3060 6GB） |
| 抑制亚型   | 3 种（FS / LTS / SOM，80/20 比例） |
| 神经调质   | 4 种（DA / ACh / NE / 5HT，突触级受体） |
| 训练目标步数 | 3M（5 个发育阶段：胚胎/突触发生/临界期/修剪/成熟） |

### Pure-SNN vs B1-Random 对比

| 指标          | B1-Random | Pure-SNN-1M | 学习效果             |
| ----------- | --------: | ----------: | ---------------- |
| 权重均值        |     0.195 |       0.299 | **+53%**         |
| 零权重突触       |     14.6% |       36.2% | **+22%**（LTD 修剪） |
| 饱和突触        |     24.6% |       46.4% | **+22%**（LTP 强化） |
| 活跃神经元       |      100% |       61.1% | **-39%**（稀疏化）    |
| spikes/step |     3,309 |       2,024 | **-39%**         |
| 兴奋/抑制比      |     68/32 |       50/50 | **平衡化**          |
| max\_chi²   |      57.8 |        58.7 | **+1.5%**（几乎无变化） |
| 卡方显著神经元     |   0/9,933 |     0/6,073 | **未学到字节映射**      |

### 各阶段 max\_chi² 演化

| 阶段                   | max\_chi² | 说明            |
| -------------------- | --------: | ------------- |
| B1-Random（随机基线）      |      57.8 | 无学习           |
| Pure-SNN-1M（纯 STDP）  |      58.7 | STDP 真实能力     |
| 2c（默认参数）             |      57.6 | -             |
| 2d-v2（P0+P1+P2 手工机制） |      59.5 | 手工机制微弱贡献      |
| 2A（柱内三层+硬编码映射+k-WTA） |      83.9 | 手工机制强贡献（+30%） |

临界值（df=88, p<0.01）= 123.2，**所有阶段均未达成**。

***

## 诚实评估

### Stage 0-2c 证明了什么？

1. ✓ **纯 STDP 能学到突触级结构**（双峰化、稀疏化、兴奋/抑制平衡）
2. ✓ **柱内三层流水线架构是正确的**（信号路径打通，spikes/step 合理）
3. ✓ **规模 10⁴ 神经元 / 10⁶ 突触 / 1M 步训练的纯局部学习有能力边界**
4. ✓ **手工机制（硬编码映射 + 外部 k-WTA）能"画"出映射，但不是网络涌现的能力**

### Stage 2e（截至 2026-07-24）证明了什么？

1. ✓ **5.5×10⁴ 神经元 / 1.07×10⁷ 突触规模在 6GB 显存上工程可行**（1.4GB / 1.5GB 预算）
2. ✓ **多层级生物机制可独立叠加并通过烟雾测试**（AdEx / NMDA / STP / PSW / Ca²⁺ / CaMKII / 双trace）
3. ✓ **丘脑-皮层门控作为状态依赖输入控制工程可行**（gate_mean=0.70，活动补偿与 novelty 增强生效）
4. ✓ **PSW + Ca²⁺回弹 + CaMKII 三重防饱和路径在 10K 步内未出现权重饱和**
5. ⚠ **但项目级阶段目标 A+B 仍未完全达成**：A 解决（活动稀疏），B 反而恶化（柱分化 1.33→1.17）

### 这个项目没有证明什么？

1. ✗ SNN 能学到字节级语言映射（仍未达成，max_chi² 远低于临界值）
2. ✗ 柱拓扑能自发形成功能分化（col_ratio 反而恶化）
3. ✗ 纯 STDP / 即使叠加多机制仍未产生语义表征
4. ✗ 丘脑门控本身不解决柱分化问题（其状态依赖输入机制与柱分化是正交问题）

### 距离真实生物脑的差距

> 详细评估见：[人类脑差距评估.md](./人类脑差距评估.md) 与 [人脑差距模块评估](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md)

| 维度         |                      本项目 |                         人脑 |   差距（数量级） |
| ---------- | -----------------------: | -------------------------: | --------: |
| **规模差距**   |                   <br /> |                     <br /> |    <br /> |
| 神经元数       |                  5.5×10⁴ |                       10¹¹ |   **10⁶** |
| 突触数        |                  1.07×10⁷ |                       10¹⁴ |   **10⁷** |
| 单神经元突触     |                      200 |               7,000-10,000 |       10² |
| 皮层柱数       |                       50 |                      \~10⁶ |       10⁴ |
| **机制差距**   |                   <br /> |                     <br /> |    <br /> |
| 神经元模型      |      AdEx (适应性+阈值动态) | Hodgkin-Huxley (10-20 种通道) |         — |
| 突触模型       | AMPA+NMDA+GABA_A/B+STP | AMPA+NMDA+GABA+... (多受体亚型) |         — |
| 学习机制覆盖     | ~15% (STDP+PSW+Ca²⁺+CaMKII+调质+门控) | 100% (数十种机制) |         — |
| 脑区系统       | 联合皮层+前额叶雏形 | 数百脑区+丘脑+海马+基底节+小脑 |         — |
| **物理差距**   |                   <br /> |                     <br /> |    <br /> |
| 能效 (神经元/瓦) |                   ~1100 |                      4×10⁹ | **10⁶** |
| 功耗         |              \~50W (GPU) |                      \~20W |         — |

**直观类比**：

- Stage 0-2c SNN ≈ **蚂蚁神经节**
- Stage 2e SNN ≈ **蜗牛神经节到果蝇神经节之间**（规模上了半个数量级，机制覆盖从 <5% 提升到 ~15%）
- 距离果蝇全脑：~1 个数量级
- 距离小鼠脑：~3 个数量级
- 距离人脑：**6-7 个数量级**（机制差距缩小但仍巨大）

**综合差距**：规模 6-7 个数量级 + 机制 2-3 个数量级 + 功能不可比 = **至少 8-13 个数量级**。

本项目做不到真实的虚拟人脑，而是在探索"在受限规模下 SNN + 多机制能否跨越从突触结构到语义结构的鸿沟"。每一个"不能学到"的结论都是科学贡献——量化了纯局部学习规则的真实能力边界。

### 下一步路线图（按 [人脑差距模块评估](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md)）

短期（Stage 2e 剩余）：

1. **Phase R1**：完成跨柱权重三档消融（Weak / VeryWeak），失败后停止搜索转向假设审查
2. **Phase R2**：引入真实皮层层级 L4 / L2-3 / L5-6（最高优先级，比继续扩大规模更重要）
3. **Phase R3**：完善丘脑-皮层门控（已部分完成，需加入 DA 预测误差接口）
4. **Phase R4**：建立海马-皮层重放闭环
5. **Phase R5**：建立 next-byte / next-token 输出解码（从内部表征走向可观察输出）

中期（Stage 3+，需云租用算力）：

6. **规模扩展到 100K-1M 神经元**
7. **延长训练到 10M-100M 步**
8. **建立简单对话闭环**

长期（Stage 5-6，需机构合作）：

9. **验证结构同构假设的 scaling law**
10. **接近人类判据检验**

### 防止无限调参的硬规则

> 详见 [人脑差距模块评估 §17](./docs/superpowers/specs/2026-07-24-human-brain-gap-module-assessment.md#17-防止无限调参的硬规则)

1. 每个改动必须对应一个生物机制假设，不能只写"提升 col_ratio/silhouette"
2. 每个参数最多三档消融：baseline、biologically plausible、extreme boundary
3. 失败后停止该方向：三档均无效时，不继续搜索参数，转向假设审查
4. 必须设置负对照：随机字节、打乱标签、打乱柱偏好、held-out corpus
5. 中间指标不能当最终目标：spike/step、col_ratio、silhouette 只是结构指标，不等于语义
6. 语义结论必须依赖泛化行为：held-out 序列、next-token/byte 预测、简单问答或检索行为
7. 活动区间已达标后冻结：除非后续结构改动导致 A 回归，否则不再调输入增益

***

## 致谢

- **LCCC-base 语料**：清华大学 + 三星，2020 年发布，<https://github.com/thu-coai/CDial-GPT>
- **CUDA Toolkit**：NVIDIA
- **Visual Studio Build Tools**：Microsoft

***

## 许可

本项目采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议（仅署名）。

你可以自由地：

- **共享** — 复制、分发本作品
- **改编** — 修改、转换本作品

只要遵守：

- **署名** — 注明原作者，并提供许可证链接

详见 [LICENSE](./LICENSE)。
