#ifndef SNN_STAGE2_CONFIG_H
#define SNN_STAGE2_CONFIG_H

// =============================================================================
// Stage 2 configuration: columnar topology + unsupervised training
// =============================================================================
// This header augments stage0's config.h (which defines N_TOTAL_NEURONS,
// N_TOTAL_SYNAPSES, etc.). Stage2 reuses those values for memory capacity
// and adds columnar-topology-specific parameters.
//
// Hard constraint (project memory): stage2 must NOT modify stage0's config.h.
// All stage2-only constants live here.
// =============================================================================

#include "../include/config.h"  // pulls in N_TOTAL_NEURONS, N_TOTAL_SYNAPSES, ...

// -----------------------------------------------------------------------------
// Columnar topology (方案 A: 真正的皮层柱)
// -----------------------------------------------------------------------------
// 10 columns x 1000 neurons = 10000 neurons (= N_TOTAL_NEURONS).
//
// 方案 A 核心：每柱内含 sensory + association + motor 三层流水线
//   柱 c 的 sensory 层:     [c*1000,       c*1000 + 200)   = 200 神经元
//   柱 c 的 association 层: [c*1000 + 200, c*1000 + 800)   = 600 神经元
//   柱 c 的 motor 层:       [c*1000 + 800, c*1000 + 1000) = 200 神经元
//
// 全局统计保持与 stage0 一致:
//   总 sensory     = 10 * 200 = 2000 = N_SENSORY_NEURONS
//   总 association = 10 * 600 = 6000 = N_ASSOCIATION_NEURONS
//   总 motor       = 10 * 200 = 2000 = N_MOTOR_NEURONS
//
// 信号路径（柱内三层，无跨柱稀释）:
//   字节 b → 柱 (b % 10) 的 sensory 层 (one-hot)
//     → 柱内 association 层 (intra-column, 80 突触)
//     → 柱内 motor 层 (intra-column, 80 突触)
//     → k-WTA 柱间竞争
//
// Each neuron grows 100 synapses total: 80 intra-column + 20 inter-column.
//   - effective p_intra = 80 / 999  ~= 0.080
//   - effective p_inter = 20 / 8991 ~= 0.0022
// -----------------------------------------------------------------------------
#define N_COLUMNS                   10
#define NEURONS_PER_COLUMN          1000
#define INTRA_SYNAPSES_PER_NEURON   80
#define INTER_SYNAPSES_PER_NEURON   20

// 方案 A: 柱内三层尺寸（必须与 types.h 的 STAGE2_COL_*_SIZE 一致）
#define COL_SENSORY_SIZE            200
#define COL_ASSOCIATION_SIZE        600
#define COL_MOTOR_SIZE              200
static_assert(COL_SENSORY_SIZE + COL_ASSOCIATION_SIZE + COL_MOTOR_SIZE
              == NEURONS_PER_COLUMN,
              "stage2: COL layer sizes must sum to NEURONS_PER_COLUMN");
static_assert(COL_SENSORY_SIZE * N_COLUMNS == N_SENSORY_NEURONS,
              "stage2: COL_SENSORY_SIZE * N_COLUMNS must equal N_SENSORY_NEURONS");
static_assert(COL_MOTOR_SIZE * N_COLUMNS == N_MOTOR_NEURONS,
              "stage2: COL_MOTOR_SIZE * N_COLUMNS must equal N_MOTOR_NEURONS");

// 在 stage2 编译路径下，把这些常量也暴露给 types.h
// 用 #ifndef 保护，避免与 types.h 的默认值冲突（值相同，但消除重定义警告）
#ifndef STAGE2_COL_SENSORY_SIZE
  #define STAGE2_COL_SENSORY_SIZE     COL_SENSORY_SIZE
#endif
#ifndef STAGE2_COL_ASSOCIATION_SIZE
  #define STAGE2_COL_ASSOCIATION_SIZE COL_ASSOCIATION_SIZE
#endif
#ifndef STAGE2_NEURONS_PER_COLUMN
  #define STAGE2_NEURONS_PER_COLUMN   NEURONS_PER_COLUMN
#endif

// SYNAPSES_PER_NEURON (100) is inherited from stage0's config.h and must
// equal INTRA + INTER.
static_assert(INTRA_SYNAPSES_PER_NEURON + INTER_SYNAPSES_PER_NEURON
              == SYNAPSES_PER_NEURON,
              "stage2: INTRA + INTER must equal SYNAPSES_PER_NEURON");
static_assert(N_COLUMNS * NEURONS_PER_COLUMN == N_TOTAL_NEURONS,
              "stage2: N_COLUMNS * NEURONS_PER_COLUMN must equal N_TOTAL_NEURONS");

// -----------------------------------------------------------------------------
// 方案 A: 字节→柱的确定性映射
// -----------------------------------------------------------------------------
// 字节 b 始终驱动柱 (b % N_COLUMNS)，确保 STDP 能学到稳定映射。
// 256 字节均匀分布到 10 柱，每柱负责 25-26 个字节。
// 例: 'A'=0x41=65 → 柱 5; 'B'=0x42=66 → 柱 6; 0xE4 (中文头) → 柱 4
// -----------------------------------------------------------------------------
static inline int column_for_byte(int b) {
    return b % N_COLUMNS;
}

// 字节 b 在柱内的 one-hot 位置 (0..24/25)
// 例: 'A'=65, 柱 5, 位置 65/10=6
// 例: 0xE4=228, 柱 8, 位置 228/10=22
static inline int byte_position_in_column(int b) {
    return b / N_COLUMNS;
}

// Initial weight ranges (must obey hard constraints from project memory):
//   - excitatory pre -> [0, W_MAX]
//   - inhibitory pre -> [-W_MAX, 0]
// We initialize with small random magnitudes so the network is quiescent
// at episode 0 and learns via STDP.
#define STAGE2_INIT_WEIGHT_MAX      0.3f

// Synaptic delay (in time steps). Stage0 used 1.0; we keep the same value.
#define STAGE2_SYNAPSE_DELAY        1.0f

// -----------------------------------------------------------------------------
// Text codec (8-bit / 256 byte -- supports UTF-8 multibyte streams)
// -----------------------------------------------------------------------------
// Stage 2d (P1): one-hot encoding instead of binary encoding.
//
// 2c analysis found that binary encoding (8 neurons, bit patterns) dilutes
// byte identity: 'A'=0x41 and 'B'=0x42 differ by only 1 bit, so the SNN
// could not distinguish them. Chi-square test showed 0/7857 neurons with
// significant byte selectivity.
//
// One-hot encoding assigns a dedicated sensory neuron to each of the 256
// possible byte values. Each byte activates exactly one neuron with full
// gain, giving maximum discriminability. 256 << 2000 sensory capacity.
//
// Sensory layout:
//   byte b -> neuron b fires with amplitude SENSORY_INPUT_GAIN
//   neurons (b+1)..(N_SENSORY_NEURONS-1) stay at 0
// -----------------------------------------------------------------------------
#define TEXT_CODEC_BITS             8
#define TEXT_CODEC_ALPHABET_SIZE    256      // all byte values 0x00..0xFF
#define SENSORY_INPUT_NEURONS       256      // 2d: one-hot (was 8)
#define SENSORY_INPUT_GAIN          5.0f     // 2d: stronger drive (was 2.0)
#define ENCODING_MODE_ONEHOT        1        // 2d: 1=one-hot, 0=binary (legacy)

// -----------------------------------------------------------------------------
// Unsupervised training loop
// -----------------------------------------------------------------------------
// Total length: 1,000,000 time steps (~17 minutes at 1ms/step simulated time).
// Stage 2b will run this end-to-end; stage 2a (current) only runs 10,000 steps
// as a smoke test.
// -----------------------------------------------------------------------------
#define STAGE2_TOTAL_TRAIN_STEPS    1000000
#define STAGE2_SMOKE_TEST_STEPS     10000    // for stage 2a verification
#define STAGE2_LOG_INTERVAL         1000
#define STAGE2_CHECKPOINT_INTERVAL  50000
#define STAGE2_RESET_INTERVAL       10000    // periodic network reset to avoid
                                             // runaway activity / saturation

// -----------------------------------------------------------------------------
// Reward / dopamine (disabled in main path; reserved for B3 ablation)
// -----------------------------------------------------------------------------
// Per project memory: reward mechanism is excluded from main development path
// and designated as a B3 ablation study option. Default dopamine level stays
// at 1.0 (no modulation).
// -----------------------------------------------------------------------------
#define STAGE2_DEFAULT_DOPAMINE     1.0f

// -----------------------------------------------------------------------------
// Stage 2d (P2): k-Winners-Take-All columnar competition
// -----------------------------------------------------------------------------
// After each network.step(), keep only the top-k columns by spike count;
// suppress all spikes in other columns. Forces different inputs to drive
// different column subsets, giving STDP a clear signal to reinforce
// byte-column mappings.
//
// k=2 means 20% of columns active per step (2 of 10).
// -----------------------------------------------------------------------------
#define STAGE2_KWTA_K               2

// -----------------------------------------------------------------------------
// Checkpoint format
// -----------------------------------------------------------------------------
// Stage2 saves the FULL d_synapses_ array (32 MB, including STDP state:
// last_pre_spike, last_post_spike, eligibility).
// Stage0's save_weights() only saved d_weights_ (4 MB) and lost STDP state --
// unsuitable for resuming long training runs. Stage2 implements its own
// checkpoint in unsupervised_trainer.cu.
// -----------------------------------------------------------------------------
#define STAGE2_CHECKPOINT_PATH      "stage2_checkpoint.bin"

// Magic number + version for forward-compatible checkpoint format
#define STAGE2_CHECKPOINT_MAGIC     0x534E4E32u  // "SNN2"
#define STAGE2_CHECKPOINT_VERSION   1u

#endif // SNN_STAGE2_CONFIG_H
