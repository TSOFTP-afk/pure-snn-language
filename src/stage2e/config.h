#ifndef SNN_STAGE2E_CONFIG_H
#define SNN_STAGE2E_CONFIG_H

// =============================================================================
// Stage 2e: 多层级生物机制增强方案 v4
// =============================================================================
// 对应设计文档: docs/superpowers/specs/2026-07-19-bio-mechanisms-design.md
//
// v4 关键参数:
//   - 神经元: 55,000 (50K 联合皮层 + 5K 前额叶)
//   - 突触: 10.7M (含前额叶自反馈 + 跨柱连接)
//   - 柱数: 50 (柱内 L4/L2-3/L5/L6 四层皮层结构, Phase R2 模块 C)
//   - 抑制亚型: 3 种 (FS/LTS/SOM)
//   - 显存预算: 1332MB / 1.5GB (余量 168MB)
//   - 训练步数: 3M (5 个发育阶段)
//
// 硬约束 (项目记忆):
//   - 不修改 stage0/1/2 代码
//   - 80/20 兴奋/抑制神经元比例 (每层内独立维持)
//   - STDP kernel 先计算 delta_w 再更新 last_spike
//   - 抑制性突触 [-W_MAX, 0] 区间
// =============================================================================

#include <cstdint>

// -----------------------------------------------------------------------------
// 网络规模 (v4)
// -----------------------------------------------------------------------------
#define N_COLUMNS_2E               50
#define NEURONS_PER_COLUMN_2E      1000    // 联合皮层每柱 1000 神经经元
#define N_PREFRONTAL_NEURONS       5000    // 独立前额叶 (50 组 × 100)
#define N_ASSOCIATION_NEURONS_2E   (N_COLUMNS_2E * NEURONS_PER_COLUMN_2E)  // 50,000
#define N_TOTAL_NEURONS_2E         (N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS)  // 55,000

// 柱内四层皮层结构 (每柱 1000 = 200 L4 + 350 L2/3 + 200 L5 + 250 L6)
// 对应 Phase R2 模块 C: 真实新皮层层级结构 (L4/L2-3/L5/L6)
#define COL_L4_SIZE_2E            200
#define COL_L23_SIZE_2E           350
#define COL_L5_SIZE_2E            200
#define COL_L6_SIZE_2E            250
static_assert(COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E + COL_L6_SIZE_2E
              == NEURONS_PER_COLUMN_2E, "column layer sizes must sum to 1000");

#define N_L4_TOTAL_2E             (N_COLUMNS_2E * COL_L4_SIZE_2E)      // 10,000
#define N_L23_TOTAL_2E            (N_COLUMNS_2E * COL_L23_SIZE_2E)    // 17,500
#define N_L5_TOTAL_2E             (N_COLUMNS_2E * COL_L5_SIZE_2E)    // 10,000
#define N_L6_TOTAL_2E             (N_COLUMNS_2E * COL_L6_SIZE_2E)     // 12,500

// region 枚举常量 (NeuronStateAdEx.region 字段语义, uint8_t)
// 0=L4 (丘脑输入层), 1=L2/3 (整合+跨柱), 2=L5 (输出层), 3=L6 (丘脑反馈层), 4=前额叶
#define REGION_L4                 0
#define REGION_L23                1
#define REGION_L5                 2
#define REGION_L6                 3
#define REGION_PREFRONTAL          4

// 突触预算 (v4: 10.7M, 80/20 兴奋/抑制)
#define SYNAPSES_PER_NEURON_2E     200
#define N_TOTAL_SYNAPSES_2E        10700000
#define EXCITATORY_RATIO_2E        0.8f

// 跨柱突触权重范围 (生物学: 柱内强连接, 跨柱弱连接, Braitenberg & Schüz 1998)
// R1 消融实验 VeryWeak 档: Weak 档 [0.1,0.3] 无效(col_ratio 1.165→1.165), 测试极弱边界
// 降至 [0.05,0.15]/[-0.15,-0.05] 约弱 6 倍, 探索跨柱传播是否是 col_ratio 停滞的根因
// 停止规则: 若 VeryWeak 仍无效, 停止搜索权重, 转向假设审查
#define CROSS_COL_W_EXC_MIN   0.05f
#define CROSS_COL_W_EXC_MAX   0.15f
#define CROSS_COL_W_INH_MIN  -0.15f
#define CROSS_COL_W_INH_MAX  -0.05f

// 前馈层级突触权重范围 (L4→L2/3, L2/3→L5, L5→L6)
// 修复: 原柱内统一 [0.4,1.0]×w_scale 不足以驱动下游层发放
//   (稳态 V≈0.36 < 阈值 1.0, "鸡生蛋"困境: L2/3 需发放才能 LTP, 需 LTP 才能发放)
// 提升至 [2.5,3.5]×w_scale≈[0.354,0.495], 让稳态 V≈1.52 > 阈值, 打破困境
// 仅前馈连接使用此范围, 反馈/横向/跨柱连接保持原范围
#define FEEDFORWARD_W_EXC_MIN   2.5f
#define FEEDFORWARD_W_EXC_MAX   3.5f
#define FEEDFORWARD_W_INH_MIN  -3.5f
#define FEEDFORWARD_W_INH_MAX  -2.5f

// -----------------------------------------------------------------------------
// 平衡态网络权重缩放 (van Vreeswijk & Sompolinsky 1996)
// 平衡态条件: w ∝ 1/√K, 激活兴奋/抑制动态平衡 → 活动去相关
// 实际缩放因子在 network_init.cu 中用 1/sqrt(SYNAPSES_PER_NEURON_2E) 动态计算
// -----------------------------------------------------------------------------
#define BALANCED_NETWORK_K_AVG       195       // 平均入度 (≈ SYNAPSES_PER_NEURON_2E)
#define BALANCED_WEIGHT_SCALE_DIVISOR 4        // 有效入度修正因子 (稀疏激活下实际参与突触约 K/4, 缩放因子 = sqrt(DIVISOR)/sqrt(K) = 2/sqrt(K))

// 突触延迟 (v4 强化 I: 轴突延迟)
#define DELAY_STEPS_MAX            20      // 延迟环形队列槽位数
#define DELAY_INTRA_MIN            1       // 柱内延迟 1-3 步
#define DELAY_INTRA_MAX            3
#define DELAY_INTER_MIN            5       // 跨柱延迟 5-10 步
#define DELAY_INTER_MAX            10
#define DELAY_LONG_MIN             15      // 长程延迟 15-20 步
#define DELAY_LONG_MAX            20
#define DELAY_RING_SLOT_CAPACITY   750000  // 每槽位最大活跃突触数

// -----------------------------------------------------------------------------
// AdEx 神经元参数 (v3 强化, Brette & Gerstner 2005)
// -----------------------------------------------------------------------------
#define ADEX_C                    281.0f    // pF
#define ADEX_GL                   30.0f     // nS
#define ADEX_EL                   -70.6f    // mV
#define ADEX_VT                   -50.4f    // mV
#define ADEX_DELTA_T              2.0f      // mV
#define ADEX_VR                   -70.6f    // mV
#define ADEX_VPEAK                20.0f     // mV
#define ADEX_TAUM                 (ADEX_C / ADEX_GL)  // ~9.37 ms
#define ADEX_TAUM_INV             (1.0f / ADEX_TAUM)

// 电压单位重映射: V_bio = -70 + 50 * V_norm (V_norm ∈ [-0.5, 1.5])
// V_norm = 0.0  → V_bio = -70 mV (静息)
// V_norm = 1.0  → V_bio = -20 mV (接近阈值)
// V_norm = 1.5  → V_bio = 5 mV (峰值)
#define V_BIO_OFFSET              -70.0f
#define V_BIO_SCALE               50.0f

// NMDA 电压依赖 (Jahr & Stevens 1990)
#define NMDA_MG_BLOCK_THRESHOLD   -0.4f    // V_norm, 对应 ~-50 mV
#define NMDA_CA_TAU                50.0f   // ms, 钙衰减时间常数
// 前馈连接树突区室化 (模拟基底树突 Ca²⁺ 动力学)
// 生物学原理: 基底树突富含 calbindin 缓冲蛋白, Ca²⁺ 快速清除, 不易触发回弹 LTD
// 修复: 前馈 Ca²⁺ 上限 0.12 < CA_REBOUND_THRESHOLD 0.15, 回弹 LTD 永不触发
#define NMDA_CA_TAU_FEEDFORWARD   10.0f   // 基底树突快速衰减 (原 50.0f 的 1/5)
#define CA_MAX_FEEDFORWARD        0.12f   // 基底树突 Ca²⁺ 上限 (低于回弹 LTD 阈值)
#define NMDA_CONDUCTANCE_MAX      0.05f
#define NMDA_MG_CONCENTRATION     1.0f     // [Mg²⁺] mM (生理浓度)
#define NMDA_TAU                  150.0f   // ms, NMDA 电导衰减
#define NMDA_G_MAX                0.5f     // NMDA 峰值电导 (nS)

// AMPA 受体 (快速兴奋性)
#define AMPA_TAU                  5.0f     // ms
#define AMPA_G_MAX                1.0f     // AMPA 峰值电导

// GABA_A / GABA_B (抑制性)
#define GABA_A_TAU                10.0f    // ms
#define GABA_A_G_MAX              1.0f
#define GABA_B_TAU                150.0f   // ms
#define GABA_B_G_MAX              0.3f

// -----------------------------------------------------------------------------
// AdEx 适应性参数 (Brette & Gerstner 2005, §2.1)
// -----------------------------------------------------------------------------
#define ADEX_A_ADAPT              4.0f     // nS, 适应耦合电导
#define ADEX_B_RESET              0.0805f  // nA, 脉冲后适应跳变 (簇状发放关键)
#define ADEX_TAU_W                144.0f   // ms, 适应时间常数
#define ADEX_TAU_W_INV            (1.0f / ADEX_TAU_W)
#define ADEX_V_THRESH_NORM        1.0f     // V_norm 阈值
#define ADEX_V_RESET_NORM         0.0f     // V_norm 重置值
#define ADEX_REFRACTORY_STEPS     2        // 不应期步数

// 阈值动态 (钠通道失活)
#define ADEX_THETA_ADAPT_RATE     0.001f   // 每脉冲阈值提升率
#define ADEX_THETA_DECAY          0.999f   // 阈值衰减
#define ADEX_THETA_MAX            0.3f     // 最大阈值偏移 (V_norm 单位)

// -----------------------------------------------------------------------------
// 短期可塑性 STP (§2.2, Tsodyks-Markram 1998)
// -----------------------------------------------------------------------------
#define STP_U_SE                  0.2f     // 兴奋性基线利用率 U
#define STP_U_SI                  0.05f    // 抑制性基线利用率 U
#define STP_TAU_FAC               200.0f   // ms, 易化时间常数
#define STP_TAU_REC               500.0f   // ms, 恢复时间常数
#define STP_TAU_FAC_INV           (1.0f / STP_TAU_FAC)
#define STP_TAU_REC_INV           (1.0f / STP_TAU_REC)

// 前馈连接易化型 STP 参数 (L4→L2/3, L2/3→L5, L5→L6)
// 修复: 原统一抑郁型 STP 在高频输入下 resource 稳态≈0.005, 有效电流削弱~200倍
//   导致前馈权重提升 3.5 倍仍无法驱动 L2/3 发放 (3步累积 V≈0.02 << 阈值 1.0)
// 改为易化型: 低 U + 快速恢复, resource 稳态≈0.19, 有效信号增强~32倍
// 生物学依据: 皮层前馈连接多为易化型突触 (Markram 1998, Thomson & Bannister 2003)
#define STP_U_FEEDFORWARD         0.02f    // 前馈连接基线利用率 U (低, 易化型特征)
#define STP_TAU_FAC_FEEDFORWARD   200.0f   // ms, 前馈易化时间常数 (长于 τ_rec 实现易化)
#define STP_TAU_REC_FEEDFORWARD   3.0f     // ms, 前馈恢复时间常数 (配合每步恢复, 让 resource 稳态≈0.13)

// 前馈连接标志位 (receptor_flags bit4, 避免与抑制性 GABA_A bit2 冲突)
// 兴奋性突触 receptor_flags=0x03 (AMPA+NMDA), bit2-7 未使用
// 抑制性突触 receptor_flags=0x0C (GABA_A+GABA_B), bit2 已占用
// 用 bit4 (0x10) 作为前馈标志, 仅对兴奋性前馈连接设置
#define RECEPTOR_FLAG_FEEDFORWARD 0x10

// -----------------------------------------------------------------------------
// 群体编码输入 (§2.3)
// -----------------------------------------------------------------------------
#define POP_CODING_K_PER_COLUMN   100      // 每柱激活神经元数 (100 × 50 = 5000, 调参: 50→100 增强群体编码信号)
// P1 修正: AdEx 归一化阈值 V_norm=1.0, τ_m=9.37, 单步 dV=I/τ_m
//   要让 V 从 0 → 1.0 单步发放, 需 I ≥ 9.37
//   设计文档原值 1.5 严重不足 (dV=0.16), 调整为 15.0 (dV=1.6, 28% 余量)
// 架构修复: 1/√K 权重缩放后突触驱动衰减约 14 倍
//   直接输入不经过突触, 但下游层 (assoc/motor) 经突触接收, 需更强的感官驱动补偿
//   偏好柱: 30×2.0=60 (vs 原 7.0, ~8.5x), 非偏好柱: 30×0.3=9 (vs 原 7.0, ~1.3x)
#define POP_CODING_GAIN           80.0f    // 输入增益 (调参: 30→80, 配合缩放放宽提升活动至 [50,200] spikes/step)
// P1 修正: 10 步间隔导致 90% 步静默, 改为 5 步 (50% 注入密度)
// 配合修复后的权重 [0.8, 1.5], 延迟队列可在非注入步维持活动
#define INPUT_INJECT_INTERVAL     3        // 每 3 步注入一个新字节 (调参: 5→3 提升注入密度至 33%)
#define INPUT_TEXT_CORPUS_LEN     256      // 0..255 字节遍历 (P1 用伪字节流)

// 柱特异性字节偏好 (打破柱间输入对称性, 为 STDP 提供对称破缺源)
// 每柱偏好 256/N_COLUMNS 个字节, 偏好柱增益 ×2.0, 非偏好柱增益 ×0.3
#define COLUMN_BYTE_PREF_RANGE       (256 / N_COLUMNS_2E)  // 每柱偏好字节数 (= 5)
#define COLUMN_BYTE_PREF_GAIN_IN     2.0f      // 偏好柱增益倍数 (Task 6.1 试 3.0 无效, 恢复 2.0; 根因是跨柱突触传播, 需新 spec)
#define COLUMN_BYTE_PREF_GAIN_OUT    0.03f     // 非偏好柱增益倍数 (调参: 0.1→0.03, 输入比 20:1→66:1)

// -----------------------------------------------------------------------------
// 丘脑-皮层门控 (Thalamic-Cortical Gating, §1.1 注意力门控)
// -----------------------------------------------------------------------------
// 生物学基础: 丘脑根据内部状态(唤醒/注意/预测误差)动态控制输入进入皮层
//   - 活动过低 → 门控开大(补偿), 活动过高 → 门控关小(保护)
//   - novelty 高 → 门控开大(注意新输入), novelty 低 → 回归中性
//   - 门控信号 gate_signal ∈ [0,1] 调制输入增益, 替代固定 gain_in/gain_out
// 门控状态独立存储(每柱 16B), 不嵌入 NeuronStateAdEx
#define GATE_UPDATE_RATE           0.01f   // 门控更新率(慢速, 避免活动剧烈波动)
#define GATE_MIN                   0.1f    // 门控下限(避免完全闭门导致活动归零)
#define GATE_MAX                   0.9f    // 门控上限(避免完全开门失去调制)
#define GATE_ACTIVITY_COUP         0.3f    // 活动补偿耦合系数
#define GATE_NOVELTY_COUP          0.2f    // novelty 增强耦合系数
#define GATE_L6_FEEDBACK_COUP      0.15f   // L6 反馈耦合系数 (Phase R2 模块 C: 皮层-丘脑闭环)
#define GATE_ACTIVITY_EMA_DECAY    0.99f   // 活动 EMA 衰减率(慢速估计)
#define GATE_NOVELTY_EMA_DECAY     0.95f   // novelty EMA 衰减率
#define GATE_INITIAL_SIGNAL        0.5f    // 初始门控信号(半开, 中性)
#define GATE_UPDATE_INTERVAL       1       // 每步更新(但更新率慢)

// -----------------------------------------------------------------------------
// 突触级调质受体密度 (v3 强化 B)
// -----------------------------------------------------------------------------
#define DA_RECEPTOR_INIT_EXC      0.5f     // 兴奋性突触偏向 D1
#define DA_RECEPTOR_INIT_INH     -0.3f     // 抑制性突触偏向 D2
#define ACH_RECEPTOR_INIT         0.3f
#define NE_RECEPTOR_INIT          0.3f
#define HT5_RECEPTOR_INIT         0.3f

// -----------------------------------------------------------------------------
// STDP 双 trace (v4 强化 J: Bi & Poo 2001)
// -----------------------------------------------------------------------------
#define STDP_X_PRE_TAU            20.0f    // ms
#define STDP_X_POST_TAU           20.0f    // ms
#define STDP_A_PLUS_2E            0.03f
#define STDP_A_MINUS_2E           0.03f
#define STDP_W_MAX_2E             1.5f

// -----------------------------------------------------------------------------
// PSW 概率突触权重 (Probabilistic Synaptic Weights, 贝叶斯 STDP)
// -----------------------------------------------------------------------------
// 核心思想: 每个突触维护 Beta(α,β) 分布, w_eff = W_MAX · α/(α+β)
//   - LTP 事件 → α 累积 ("成功"证据)
//   - LTD 事件 → β 累积 ("失败"证据)
//   - α+β = 证据强度 → 自适应学习率衰减 (元可塑性自然涌现)
//   - w_eff 物理上 ∈ (0, W_MAX), 不可能饱和
// 取代旧的硬 clamp 机制, 从数学结构上消除权重饱和
// -----------------------------------------------------------------------------
// 初始证据强度 α+β=0.1 (弱先验, 让 STDP 增量相对显著)
// network_init 中按 |w|/W_MAX 比例分配 α/β, 但 α+β 固定 = 0.1
#define PSW_ALPHA_INIT            0.05f    // 初始 α (无信息先验)
#define PSW_BETA_INIT             0.05f    // 初始 β
#define PSW_EVIDENCE_INIT_TOTAL   0.1f     // 初始 α+β 总证据强度
#define PSW_ALPHA_MIN             1e-4f    // α 下限 (防退化)
#define PSW_BETA_MIN              1e-4f    // β 下限
// LTP/LTD 证据增量系数 (替代旧的 eta·delta_w·clamp)
// 实测: eta=1.0 + A_PLUS=0.03 → 单次 LTP evidence ≈ 0.00045 (含 M_ij~0.5)
//   10K 步内大多数突触仅 0-1 次 STDP 事件, ev=0.00045 远不足以从 0.1 增长到 mature=0.2
// 提升到 20.0: 单次 evidence ≈ 0.009, 约 11 次事件即可成熟 (高活跃突触 10K 内可达)
// PSW 元可塑性自动防止过快学习: α+β 增大 → 有效学习率 ∝ 1/(α+β)² 自然衰减
// 800K 长测: α+β 可达 ~9, 学习率降低 ~8100x, 实现稳定化 (防饱和的核心机制)
#define PSW_ETA_ALPHA             20.0f
#define PSW_ETA_BETA              20.0f
// 前馈连接专用 PSW 学习率 (减慢权重饱和, 防止 L5/L6 chi2 停滞)
// 诊断: 原 ETA=20.0 时, 前馈权重在 10K 步内 α 累积到 15 (饱和到 W_MAX=1.5)
//   饱和后 dw/dα ≈ 0.00033, STDP 无法改变有效权重, L5/L6 spike 模式固化
// 降为 2.0 (1/10): 10K 步内 α 增量 ≈ 1.5, 权重 ≈ 1.45 (接近但不饱和)
#define PSW_ETA_ALPHA_FEEDFORWARD 2.0f
#define PSW_ETA_BETA_FEEDFORWARD  2.0f
// 成熟度阈值: (α+β) > PSW_MATURITY_THRESH 视为已学习稳定突触
// 设为 0.2 = 初始证据 0.1 的 2 倍, STDP 累积 0.1 证据即视为"开始学习"
#define PSW_MATURITY_THRESH       0.2f

// -----------------------------------------------------------------------------
// Ca²⁺ 回弹 LTD (Calcium-Rebound LTD, 生物学防饱和核心机制)
// -----------------------------------------------------------------------------
// 生物机制 (Yang et al. 1999, Cho et al. 2001):
//   突触后 Ca²⁺ 浓度存在双向阈值:
//     - 低 Ca²⁺ (θ_low < ca < θ_high): 触发 LTP (α 累积)
//     - 高 Ca²⁺ (ca > θ_high): 触发回弹 LTD (β 累积)  ← 防过强
//   这是 BCM 理论的分子基础, 高频刺激导致 Ca²⁺ 超载 → 主动削弱突触
//
// PSW 实现:
//   在 STDP kernel 中, 当 s.ca_concentration > CA_REBOUND_THRESHOLD 时
//   额外累积 β (LTD 证据), 强度 ∝ (ca - threshold)
//   这让 PSW 获得了生物学的"高频保护"机制, 补全了元可塑性之外的防饱和路径
//
// 参数标定 (基于实测 ca_mean=0.042, ca∈[0,1]):
//   架构修复: 实测 max_ca≈0.35, 原 θ=0.5 永不触发; 降至 0.15 让回弹 LTD 可激活
//   gain = 0.5:  回弹 LTD 强度, (ca-θ)·gain 作为额外 β 增量
#define CA_REBOUND_THRESHOLD      0.15f    // 回弹 LTD 触发阈值 (架构修复: 0.5→0.15 适配低活动)
#define CA_REBOUND_LTD_GAIN       0.5f     // 回弹 LTD 强度系数

// 2 阶 eligibility trace (v3 强化 H)
#define STDP_E1_TAU               20.0f    // 快 trace (ms)
#define STDP_E2_TAU               200.0f   // 慢 trace (ms)

// CaMKII 分子巩固 (v4 强化 K)
#define CAMKII_K1                 0.8f     // Ca2+ 驱动激活率
#define CAMKII_K2                 0.1f     // PP1 去磷酸化率
#define CAMKII_K3                 0.5f     // 自磷酸化率
#define CAMKII_K4                 0.05f    // 慢去磷酸化率
#define CAMKII_AUTOPHOS_FACIL     0.3f     // 易化态阈值
#define CAMKII_AUTOPHOS_CONSOL    0.7f     // 巩固态阈值

// -----------------------------------------------------------------------------
// 3 种抑制性亚型 (v2 修复 4: FS/LTS/SOM, 80/20 硬约束)
// -----------------------------------------------------------------------------
// 抑制性神经元总数 = 16% × 50K = 8,000 (保留 4% 余量给跨柱抑制)
//   FS:  4,000 (50%)
//   LTS: 2,400 (30%)
//   SOM: 1,600 (20%, 其中 60% SST + 40% VIP)
#define INHIBITORY_NEURON_RATIO   0.16f
#define INHIB_FS_RATIO            0.50f
#define INHIB_LTS_RATIO           0.30f
#define INHIB_SOM_RATIO           0.20f

enum class InhibitorySubtype : uint8_t {
    NONE = 0,   // 兴奋性
    FS   = 1,   // Fast-spiking (PV+)
    LTS  = 2,   // Low-threshold spiking (SST+)
    SOM  = 3,   // Somatostatin (VIP+ / SST+)
};

// -----------------------------------------------------------------------------
// 调质系统 (v3/v4)
// -----------------------------------------------------------------------------
#define N_NEUROMODULATORS_2E      4        // DA / ACh / NE / 5HT
#define DA_TAU                    100.0f   // ms, 多巴胺衰减
#define ACH_TAU                   200.0f
#define NE_TAU                    150.0f
#define HT5_TAU                   300.0f

// DA 价值函数 (v2 修复 3)
#define W_VALUE_DIM               200      // 亚柱级 (v3 强化 F: 50→200)
#define W_PRED_DIM                200      // 线性预测器维度
#define TD_GAMMA                  0.9f
#define ETA_VALUE                 0.001f
#define ETA_PRED                  0.001f
#define NOVELTY_EMA_BETA          0.99f
#define DA_DOWNGRADE_THRESHOLD    5.0f     // |δ| 持续超此值触发降级

// -----------------------------------------------------------------------------
// 海马体索引 (v3 强化 C)
// -----------------------------------------------------------------------------
#define HIPP_INDEX_SIZE           50000
#define PATTERN_DIM               50       // v3: 10→50 维 PCA 签名
#define HIPP_REPLAY_BATCH         200      // v3: 50→200
#define HIPP_NOVELTY_THRESHOLD    0.5f
#define HIPP_REPLAY_DECAY         0.9f

// -----------------------------------------------------------------------------
// 工作记忆 + 前额叶 (v3 强化 E)
// -----------------------------------------------------------------------------
#define WM_SLOTS                  50       // v3: 10→50
#define WM_PATTERN_DIM            PATTERN_DIM
#define WM_DECAY_FACTOR           0.995f
#define WM_ACTIVATION_THRESHOLD   0.3f

// 前额叶分组
#define PREFRONTAL_GROUPS         50
#define NEURONS_PER_PF_GROUP      100      // 50 × 100 = 5000
static_assert(PREFRONTAL_GROUPS * NEURONS_PER_PF_GROUP == N_PREFRONTAL_NEURONS,
              "prefrontal groups × neurons must equal N_PREFRONTAL_NEURONS");

// -----------------------------------------------------------------------------
// 共激活跟踪 (v3 强化 D)
// -----------------------------------------------------------------------------
#define COACT_TRACKER_SIZE        500000
#define COACT_K_SAMPLE            500      // 每步采样 500 对
#define COACT_FORM_THRESHOLD      5        // 形成突触的共激活阈值
#define COACT_EVICT_STEPS         5000
#define N_FORM_PER_CYCLE          5000

// -----------------------------------------------------------------------------
// PCA 反投影 (v3 强化 A: 全量 GPU 矩阵)
// -----------------------------------------------------------------------------
#define PCA_SNAPSHOT_BUFFER       100
#define PCA_UPDATE_INTERVAL       100      // 每 100 步增量更新
#define PCA_RETRAIN_INTERVAL      100000   // 每 100K 步全量重训
#define PCA_ANCHOR_REFRESH        10000    // 每 10K 步刷新锚点
#define PCA_LEARNING_RATE         0.01f

// -----------------------------------------------------------------------------
// NMDA 钙浓度快照 (v3 强化 G)
// -----------------------------------------------------------------------------
#define CA_SNAPSHOT_INTERVAL      10       // 每 10 步归档
#define CA_HISTORY_LEN            10
#define CA_HISTORY_MAX_ACTIVE     100000   // 稀疏归档, 仅 |ca| > θ

// -----------------------------------------------------------------------------
// 发育时间线 (v2 修复 5: 1M → 3M)
// -----------------------------------------------------------------------------
#define DEV_PHASE_EMBRYO_END      5000     // 0 - 5K (加速涌现验证, 原 30K)
#define DEV_PHASE_SYNAPTO_END     200000   // 30K - 200K
#define DEV_PHASE_CRITICAL_END    800000   // 200K - 800K
#define DEV_PHASE_PRUNE_END       1500000  // 800K - 1.5M
#define DEV_PHASE_MATURE_END      3000000  // 1.5M - 3M
#define DEV_TOTAL_STEPS           3000000

// 硬检查点 (v2 修复 6)
#define CHECKPOINT_PHASE2         200000   // 卡方 > 1% (500 神经元)
#define CHECKPOINT_PHASE3         800000   // silhouette > 0.15 + KL > 0.3
#define CHECKPOINT_PHASE4         1500000  // α∈[2,5], R²>0.85

// -----------------------------------------------------------------------------
// 训练循环
// -----------------------------------------------------------------------------
#define SMOKE_TEST_STEPS_2E       10000    // P0: 10K 步烟雾测试
#define LOG_INTERVAL_2E           1000
#define CHECKPOINT_INTERVAL_2E    50000

// 显存预算
#define VRAM_BUDGET_BYTES         (1500LL * 1024 * 1024)   // 1.5 GB
#define VRAM_PEAK_TARGET_BYTES    (1332LL * 1024 * 1024)   // v4 目标峰值

// -----------------------------------------------------------------------------
// CUDA 工具宏
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK_2E(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); } } while(0)
#define CUDA_CHECK_LAST_2E() do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Kernel Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); } } while(0)

#define THREADS_PER_BLOCK_2E      256

#endif // SNN_STAGE2E_CONFIG_H
