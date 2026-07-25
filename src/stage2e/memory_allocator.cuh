#ifndef SNN_STAGE2E_MEMORY_ALLOCATOR_CUH
#define SNN_STAGE2E_MEMORY_ALLOCATOR_CUH

// =============================================================================
// Stage 2e 显存分配器 (P0)
// =============================================================================
// 原始设计目标为 1332MB；运行预算由 --memory-budget-mb 配置。
//
// P0 职责:
//   1. 一次性分配所有持久 GPU 缓冲 (1242MB)
//   2. 提供临时缓冲池 (CSR 重建等, 90MB)
//   3. 实时监控显存使用, 超过 1.4GB 报警
//   4. 析构时统一释放
//
// 注意: P0 阶段不实际初始化数据, 只分配内存
// =============================================================================

#include "config.h"
#include "types.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace stage2e {

// -----------------------------------------------------------------------------
// 持久 GPU 缓冲 (所有指针, 单一所有权)
// -----------------------------------------------------------------------------
struct PersistentBuffers {
    // 神经元 (55K × 56B = 3.08 MB)
    NeuronStateAdEx*   d_neurons;                // 55,000
    bool*              d_spike_flags;            // 55,000 × 1B

    // 突触 (10.7M × 80B = 856 MB)
    BioSynapse*        d_synapses;               // 10,700,000
    int*               d_csr_row_ptr;            // 55,001 × 4B
    int*               d_csr_col_idx;            // 10,700,000 × 4B
    float*             d_weights_cache;          // 10,700,000 × 4B (d_synapses.weight 的镜像, 用于快速统计)
    float*             d_eligibility;            // 10,700,000 × 4B (1 阶, 独立数组便于原子操作)
    float*             d_eligibility_slow;       // 10,700,000 × 4B (v3 强化 H: 2 阶)

    // PSW 概率突触权重 (10.7M × 4B × 2 = 85.6 MB)
    // w_eff = W_MAX · α/(α+β), 取代硬 clamp, 物理上消除饱和
    float*             d_synapse_alpha;          // 10,700,000 × 4B  LTP 证据累积
    float*             d_synapse_beta;           // 10,700,000 × 4B  LTD 证据累积

    // PCA 反投影 (v3 强化 A: 55K × 50 × 4B = 11 MB)
    float*             d_pca_W;                  // 55,000 × 50

    // NMDA 钙浓度快照 (v3 强化 G: 10.7M × 4B = 42.8 MB)
    float*             d_ca_snapshot;            // 当前步钙浓度快照
    float*             d_ca_history_sparse;      // ~100K × 10 = 4 MB (稀疏归档)

    // v4 强化 I: 突触传导延迟 (40.7 MB)
    uint8_t*           d_synapse_delay;          // 10.7M × 1B = 10.7 MB
    // 延迟环形队列: 20 槽 × (突触索引 + 电流 + 计数)
    int*               d_delay_ring_indices;     // 20 × 500K × 4B = 40 MB (估计每步活跃 500K)
    float*             d_delay_ring_current;     // 20 × 500K × 4B = 40 MB
    // 注: 上面的 ring buffer 估算偏大, 实际可能更小

    // v4 强化 J: STDP x_pre trace (10.7M × 4B = 42.8 MB)
    // x_post trace 已在 BioSynapse 中
    float*             d_stdp_x_pre_trace;
    // Lazy STDP trace materialization epoch. The large-memory Spark target
    // trades one int per synapse for avoiding a full expensive trace update
    // on every time step.
    int*               d_stdp_trace_epoch;

    // v4 强化 K: CaMKII activity (10.7M × 4B = 42.8 MB)
    // autophosph 已在 BioSynapse 中
    float*             d_camkii_activity;

    // 输入电流缓冲
    float*             d_input_current;          // 55,000 × 4B
    float*             d_nmda_current;           // 55,000 × 4B
    float*             d_inhibitory_current;     // 55,000 × 4B

    // 调质浓度 (4 种 × 55K × 4B = 0.88 MB)
    float*             d_da_concentration;       // 55,000
    float*             d_ach_concentration;      // 55,000
    float*             d_ne_concentration;       // 55,000
    float*             d_ht5_concentration;      // 55,000

    // 海马体索引 (v3 强化 C: 50K × 256B = 12.8 MB)
    HippoIndex*        d_hippo_indices;          // 50,000

    // 共激活跟踪器 (v3 强化 D: 500K × 16B = 8 MB)
    CoactTracker*      d_coact_trackers;         // 500,000

    // 工作记忆槽位 (v3 强化 E: 50 × 216B)
    WMSlot*            d_wm_slots;               // 50

    // DA 价值函数相关 (亚柱级 200 维, CPU 端为主, GPU 镜像)
    float*             d_subcolumn_fr;           // 200 × 4B (当前亚柱发放直方图)
    float*             d_baseline_fr;            // 200 × 4B (EMA 基线)
    float*             d_w_pred;                 // 200 × 200 × 4B = 160KB (CPU-GPU 镜像)
    float*             d_w_value;                // 200 × 4B
    float*             d_pred_fr;                // 200 × 4B (预测器输出)

    // 字节直方图 (NE 用)
    int*               d_byte_histogram;         // 256 × 4B

    // 卡方检验: 每个神经元对每个字节的发放计数 (55K × 256 × 4B = 56 MB)
    int*               d_neuron_byte_counts;     // N_TOTAL_NEURONS_2E × 256

    // 重放注入缓冲 (睡眠态用)
    float*             d_replay_injection;       // 55,000 × 4B
};

// -----------------------------------------------------------------------------
// 显存分配器
// -----------------------------------------------------------------------------
class MemoryAllocator {
public:
    explicit MemoryAllocator(size_t budget_bytes = DEFAULT_VRAM_BUDGET_MB * 1024ULL * 1024ULL)
        : d_bufs_{}, vram_used_(0), vram_peak_(0), budget_bytes_(budget_bytes),
          allocation_failed_(false) {}
    ~MemoryAllocator() { free_all(); }

    // 分配所有持久缓冲 (返回总字节数, 失败返回 0)
    size_t allocate_all();

    // 释放所有
    void free_all();

    // 获取缓冲结构
    PersistentBuffers& buffers() { return d_bufs_; }
    const PersistentBuffers& buffers() const { return d_bufs_; }

    // 显存统计
    size_t vram_used() const { return vram_used_; }
    size_t vram_peak() const { return vram_peak_; }
    size_t budget_bytes() const { return budget_bytes_; }

    // 检查显存是否超过本次运行预算 (90% 时警告)
    bool check_budget() const;

    // 打印显存预算表
    void print_budget_report() const;

private:
    PersistentBuffers d_bufs_;
    size_t vram_used_;
    size_t vram_peak_;
    size_t budget_bytes_;
    bool allocation_failed_;

    // 模板化分配助手
    template<typename T>
    T* alloc(size_t count, const char* name, size_t* accum_bytes);
};

} // namespace stage2e

#endif // SNN_STAGE2E_MEMORY_ALLOCATOR_CUH
