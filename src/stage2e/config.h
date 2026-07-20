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
//   - 柱数: 50 (柱内 sensory/association/motor 三层)
//   - 抑制亚型: 3 种 (FS/LTS/SOM)
//   - 显存预算: 1332MB / 1.5GB (余量 168MB)
//   - 训练步数: 3M (5 个发育阶段)
//
// 硬约束 (项目记忆):
//   - 不修改 stage0/1/2 代码
//   - 80/20 兴奋/抑制神经元比例
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

// 柱内三层 (每柱 1000 = 200 sensory + 600 association + 200 motor)
#define COL_SENSORY_SIZE_2E        200
#define COL_ASSOCIATION_SIZE_2E    600
#define COL_MOTOR_SIZE_2E          200
static_assert(COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E + COL_MOTOR_SIZE_2E
              == NEURONS_PER_COLUMN_2E, "column layer sizes must sum to 1000");

#define N_SENSORY_TOTAL_2E         (N_COLUMNS_2E * COL_SENSORY_SIZE_2E)      // 10,000
#define N_ASSOC_TOTAL_2E           (N_COLUMNS_2E * COL_ASSOCIATION_SIZE_2E) // 30,000
#define N_MOTOR_TOTAL_2E           (N_COLUMNS_2E * COL_MOTOR_SIZE_2E)       // 10,000

// 突触预算 (v4: 10.7M, 80/20 兴奋/抑制)
#define SYNAPSES_PER_NEURON_2E     200
#define N_TOTAL_SYNAPSES_2E        10700000
#define EXCITATORY_RATIO_2E        0.8f

// 突触延迟 (v4 强化 I: 轴突延迟)
#define DELAY_STEPS_MAX            20      // 延迟环形队列槽位数
#define DELAY_INTRA_MIN            1       // 柱内延迟 1-3 步
#define DELAY_INTRA_MAX            3
#define DELAY_INTER_MIN            5       // 跨柱延迟 5-10 步
#define DELAY_INTER_MAX            10
#define DELAY_LONG_MIN             15      // 长程延迟 15-20 步
#define DELAY_LONG_MAX            20

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

// -----------------------------------------------------------------------------
// 群体编码输入 (§2.3)
// -----------------------------------------------------------------------------
#define POP_CODING_K_PER_COLUMN   50       // 每柱激活神经元数 (50 × 50 = 2500)
// P1 修正: AdEx 归一化阈值 V_norm=1.0, τ_m=9.37, 单步 dV=I/τ_m
//   要让 V 从 0 → 1.0 单步发放, 需 I ≥ 9.37
//   设计文档原值 1.5 严重不足 (dV=0.16), 调整为 15.0 (dV=1.6, 28% 余量)
#define POP_CODING_GAIN           15.0f    // 输入增益 (P1 修正: 1.5→15.0)
#define INPUT_INJECT_INTERVAL     10       // 每 10 步注入一个新字节 (避免饱和)
#define INPUT_TEXT_CORPUS_LEN     256      // 0..255 字节遍历 (P1 用伪字节流)

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
#define STDP_W_MAX_2E             1.0f

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
#define DEV_PHASE_EMBRYO_END      30000    // 0 - 30K
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
