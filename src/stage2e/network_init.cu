// =============================================================================
// Stage 2e 网络初始化实现 (P1)
// =============================================================================
// 设计要点 (Phase R2 模块 C: 真实皮层层级结构 L4/L2-3/L5/L6):
//   1. 神经元布局: 50 柱 × 1000 神经元 = 50K 联合皮层 + 5K 前额叶
//      - 柱 c 的 L4 层:  [c*1000,       c*1000 + 200)    (丘脑输入)
//      - 柱 c 的 L2/3 层: [c*1000 + 200, c*1000 + 550)   (整合+跨柱)
//      - 柱 c 的 L5 层:  [c*1000 + 550, c*1000 + 750)   (输出)
//      - 柱 c 的 L6 层:  [c*1000 + 750, c*1000 + 1000)  (丘脑反馈)
//      - 前额叶:          [50000, 55000)
//   2. 80/20 兴奋/抑制: 每层 (L4/L2-3/L5/L6) 内前 80% 兴奋, 后 20% 抑制
//   3. 抑制亚型: 抑制性中 FS:LTS:SOM = 50:30:20 (每层内独立分配)
//   4. 突触拓扑 (host 端生成, 一次性上传 GPU):
//      - 柱内: 每神经元 150 个柱内突触 (按生物合理流向规则)
//          L4 → L2/3,  L2/3 → L2/3(横向)+L5,  L5 → L6+L2/3(反馈),  L6 → L4(反馈)+L6(横向)
//      - 跨柱: 仅 L2/3 神经元 ~30 个跨柱突触 (pre 和 post 都必须是 L2/3)
//      - 前额叶投射: 从 L5 发起 ~15 个 (替代旧 association 层)
//      - 总数 ~ 10.7M (通过补足联合皮层出度维持)
//   5. 突触延迟: 柱内 1-3, 跨柱 5-10, 长程 15-20
// =============================================================================

#include "network_init.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <curand_kernel.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernel: 初始化神经元 (cudaMemset 后再填字段)
// Phase R2 模块 C: 4 层皮层结构 (L4/L2-3/L5/L6) + 前额叶
// -----------------------------------------------------------------------------
__global__ void init_neurons_kernel(NeuronStateAdEx* neurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_TOTAL_NEURONS_2E) return;

    NeuronStateAdEx& n = neurons[i];

    // AdEx 静息状态
    n.membrane_potential   = 0.0f;       // V_norm = 0 (静息)
    n.synaptic_current     = 0.0f;
    n.nmda_current         = 0.0f;
    n.adaptive_conductance = 0.0f;       // w = 0 (无初始适应)
    n.last_spike_time      = -1000;      // 远古值
    n.fire_rate            = 0.0f;
    n.refractory_remaining = 0;
    n.ca_neuron            = 0.0f;
    n.wm_injection         = 0.0f;
    n.homeostatic_factor   = 1.0f;       // 初始缩放 = 1.0

    // 类型/区域/柱分配
    if (i < N_ASSOCIATION_NEURONS_2E) {
        // 联合皮层 (50 柱 × 1000, 4 层皮层结构)
        int col = i / NEURONS_PER_COLUMN_2E;       // 0..49
        int off = i % NEURONS_PER_COLUMN_2E;       // 0..999
        n.column_id = static_cast<uint8_t>(col);
        n.pf_group_id = -1;

        // 4 层 region 分配 (L4/L2-3/L5/L6)
        int layer_off, layer_size;
        if (off < COL_L4_SIZE_2E) {
            n.region = REGION_L4;             // L4 (丘脑输入层)
            layer_off = off;
            layer_size = COL_L4_SIZE_2E;
        } else if (off < COL_L4_SIZE_2E + COL_L23_SIZE_2E) {
            n.region = REGION_L23;            // L2/3 (整合+跨柱)
            layer_off = off - COL_L4_SIZE_2E;
            layer_size = COL_L23_SIZE_2E;
        } else if (off < COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E) {
            n.region = REGION_L5;             // L5 (输出层)
            layer_off = off - COL_L4_SIZE_2E - COL_L23_SIZE_2E;
            layer_size = COL_L5_SIZE_2E;
        } else {
            n.region = REGION_L6;             // L6 (丘脑反馈层)
            layer_off = off - COL_L4_SIZE_2E - COL_L23_SIZE_2E - COL_L5_SIZE_2E;
            layer_size = COL_L6_SIZE_2E;
        }

        // 80/20 兴奋/抑制 (每层内独立维持, 前 80% 兴奋)
        int exc_count = static_cast<int>(layer_size * EXCITATORY_RATIO_2E);
        if (layer_off < exc_count) {
            n.neuron_type = 0;  // EXCITATORY
            n.inhibitory_subtype = 0;  // NONE
        } else {
            n.neuron_type = 1;  // INHIBITORY
            // 抑制亚型分配: 50% FS, 30% LTS, 20% SOM (每层内独立分配)
            int inh_off = layer_off - exc_count;
            int inh_total = layer_size - exc_count;
            float frac = static_cast<float>(inh_off) / static_cast<float>(inh_total);
            if (frac < INHIB_FS_RATIO) {
                n.inhibitory_subtype = 1;  // FS
            } else if (frac < INHIB_FS_RATIO + INHIB_LTS_RATIO) {
                n.inhibitory_subtype = 2;  // LTS
            } else {
                n.inhibitory_subtype = 3;  // SOM
            }
        }
    } else {
        // 前额叶 (5K = 50 组 × 100)
        int pf_i = i - N_ASSOCIATION_NEURONS_2E;
        int pf_group = pf_i / NEURONS_PER_PF_GROUP;
        n.column_id = 255;       // 255 = 前额叶标记
        n.region = REGION_PREFRONTAL;  // 前额叶 (新枚举值 4)
        n.pf_group_id = static_cast<int16_t>(pf_group);
        // 前额叶 80/20 兴奋/抑制
        int off_in_group = pf_i % NEURONS_PER_PF_GROUP;
        int exc_count = static_cast<int>(NEURONS_PER_PF_GROUP * EXCITATORY_RATIO_2E);
        if (off_in_group < exc_count) {
            n.neuron_type = 0;
            n.inhibitory_subtype = 0;
        } else {
            n.neuron_type = 1;
            n.inhibitory_subtype = 1;  // 前额叶主要 FS 型 PV+
        }
    }

    // 阈值偏移初始 0
    n.threshold_offset = 0;

    n._reserved = 0;
    n._pad = 0;
}

// -----------------------------------------------------------------------------
// Host: 初始化神经元
// -----------------------------------------------------------------------------
void init_neurons(NeuronStateAdEx* d_neurons) {
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    init_neurons_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(d_neurons);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Host: 初始化突触拓扑 (CPU 端生成, 一次性上传)
// -----------------------------------------------------------------------------
// 拓扑策略 (Phase R2 模块 C: 真实皮层层级结构):
//   每神经元目标出度:
//     - 柱内突触: 150 (按生物合理流向规则, 见下表)
//     - 跨柱突触: 30  (仅 L2/3 神经元, pre 和 post 都必须是 L2/3)
//     - 前额叶投射: 15 (仅 L5 神经元, 替代旧 association 层)
//     - 前额叶自反馈: 100 + 50 到联合皮层
//   柱内流向规则 (替代旧 sensory→assoc→motor 循环):
//     L4  → L2/3
//     L2/3 → L2/3 (横向) + L5
//     L5  → L6 + L2/3 (反馈)
//     L6  → L4 (反馈) + L6 (横向)
//   延迟:
//     - 柱内: 1-3 步
//     - 跨柱: 5-10 步
//     - 长程(前额叶): 15-20 步
//   总突触数通过补足联合皮层出度维持 ~10.7M (N_TOTAL_SYNAPSES_2E)
// -----------------------------------------------------------------------------
void init_synapses_host(std::vector<BioSynapse>& h_synapses,
                       std::vector<int>& h_row_ptr,
                       std::vector<int>& h_col_idx,
                       std::vector<float>& h_weights_cache,
                       std::vector<uint8_t>& h_delay,
                       std::vector<float>& h_alpha,
                       std::vector<float>& h_beta,
                       uint32_t seed) {
    // 简单的确定性 PRNG (避免引入 curand 依赖到 host)
    // xorshift32 的零状态会永久锁死，用户传入 0 时映射到固定非零状态。
    uint32_t rng = seed == 0 ? 0x6D2B79F5u : seed;
    auto next_rng = [&rng]() -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    auto randf = [&next_rng](float lo, float hi) -> float {
        return lo + (hi - lo) * (next_rng() / 4294967296.0f);
    };
    auto randi = [&next_rng](int lo, int hi) -> int {  // [lo, hi)
        return lo + static_cast<int>(next_rng() % static_cast<uint32_t>(hi - lo));
    };

    // 平衡态网络权重缩放 (van Vreeswijk & Sompolinsky 1996)
    // w ∝ 1/√K, 调参 2/√K 保留平衡态特性但提升单突触驱动 2 倍
    const float w_scale = 2.0f / sqrtf((float)SYNAPSES_PER_NEURON_2E);  // ≈ 0.1414

    h_row_ptr.assign(N_TOTAL_NEURONS_2E + 1, 0);
    h_synapses.clear();
    h_synapses.reserve(N_TOTAL_SYNAPSES_2E);
    h_col_idx.clear(); h_col_idx.reserve(N_TOTAL_SYNAPSES_2E);
    h_weights_cache.clear(); h_weights_cache.reserve(N_TOTAL_SYNAPSES_2E);
    h_delay.clear(); h_delay.reserve(N_TOTAL_SYNAPSES_2E);
    h_alpha.clear(); h_alpha.reserve(N_TOTAL_SYNAPSES_2E);
    h_beta.clear();  h_beta.reserve(N_TOTAL_SYNAPSES_2E);

    // 临时统计每神经元出度
    std::vector<int> out_deg(N_TOTAL_NEURONS_2E, 0);
    int total_target = 0;

    // 预计算每神经元出度
    // - 联合皮层: 150 柱内 + 30 跨柱(仅 L2/3) + 15 前额叶投射(仅 L5)
    // - 前额叶: 100 自反馈 + 50 到联合皮层
    for (int pre = 0; pre < N_TOTAL_NEURONS_2E; ++pre) {
        int deg = 0;
        if (pre < N_ASSOCIATION_NEURONS_2E) {
            int off = pre % NEURONS_PER_COLUMN_2E;
            deg = 150;  // 柱内基线出度
            if (off >= COL_L4_SIZE_2E && off < COL_L4_SIZE_2E + COL_L23_SIZE_2E) {
                deg += 30;  // 仅 L2/3 神经元有跨柱突触
            }
            if (off >= COL_L4_SIZE_2E + COL_L23_SIZE_2E &&
                off < COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E) {
                deg += 15;  // 仅 L5 神经元投射到前额叶
            }
        } else {
            // 前额叶: 100 自反馈 + 50 到联合皮层
            deg = 100 + 50;
        }
        out_deg[pre] = deg;
        total_target += deg;
    }

    // 调整为精确 N_TOTAL_SYNAPSES_2E (补足联合皮层出度)
    while (total_target < N_TOTAL_SYNAPSES_2E) {
        int pre = randi(0, N_ASSOCIATION_NEURONS_2E);
        out_deg[pre]++;
        total_target++;
    }
    while (total_target > N_TOTAL_SYNAPSES_2E) {
        int pre = randi(0, N_ASSOCIATION_NEURONS_2E);
        if (out_deg[pre] > 50) {
            out_deg[pre]--;
            total_target--;
        }
    }

    // 填充 row_ptr (CSR)
    h_row_ptr[0] = 0;
    for (int i = 0; i < N_TOTAL_NEURONS_2E; ++i) {
        h_row_ptr[i + 1] = h_row_ptr[i] + out_deg[i];
    }

    // 生成突触
    h_synapses.resize(N_TOTAL_SYNAPSES_2E);
    h_col_idx.resize(N_TOTAL_SYNAPSES_2E);
    h_weights_cache.resize(N_TOTAL_SYNAPSES_2E);
    h_delay.resize(N_TOTAL_SYNAPSES_2E);
    h_alpha.resize(N_TOTAL_SYNAPSES_2E);
    h_beta.resize(N_TOTAL_SYNAPSES_2E);

    // 辅助 lambda: 柱内指定层的基址 (相对 col_base 的偏移)
    auto layer_offset_in_col = [](int layer) -> int {
        switch (layer) {
            case REGION_L4:  return 0;
            case REGION_L23: return COL_L4_SIZE_2E;
            case REGION_L5:  return COL_L4_SIZE_2E + COL_L23_SIZE_2E;
            case REGION_L6:  return COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E;
            default:         return 0;
        }
    };
    auto layer_sz = [](int layer) -> int {
        switch (layer) {
            case REGION_L4:  return COL_L4_SIZE_2E;
            case REGION_L23: return COL_L23_SIZE_2E;
            case REGION_L5:  return COL_L5_SIZE_2E;
            case REGION_L6:  return COL_L6_SIZE_2E;
            default:         return 0;
        }
    };
    // 辅助 lambda: 按 pre_layer 选择柱内目标层 (生物合理流向规则)
    //   L4 → L2/3
    //   L2/3 → L2/3 (横向) 或 L5
    //   L5 → L6 或 L2/3 (反馈)
    //   L6 → L4 (反馈) 或 L6 (横向)
    //   前额叶 → 前额叶 (同组)
    auto pick_target_layer = [&next_rng](int pre_layer) -> int {
        if (pre_layer == REGION_L4)  return REGION_L23;
        if (pre_layer == REGION_L23) return (next_rng() & 1) ? REGION_L23 : REGION_L5;
        if (pre_layer == REGION_L5)  return (next_rng() & 1) ? REGION_L6 : REGION_L23;
        if (pre_layer == REGION_L6)  return (next_rng() & 1) ? REGION_L4 : REGION_L6;
        return REGION_PREFRONTAL;
    };
    // 辅助 lambda: 在指定柱和层内随机选一个 post 神经元
    auto pick_intra_post = [&](int col_base, int target_layer) -> int {
        int base = col_base + layer_offset_in_col(target_layer);
        int size = layer_sz(target_layer);
        return base + randi(0, size);
    };
    // 辅助 lambda: 初始化 BioSynapse 通用字段 (避免重复)
    auto init_syn_fields = [&](int idx, int pre, int post, float weight, uint8_t delay,
                                bool pre_is_exc, bool is_feedforward = false) {
        BioSynapse& s = h_synapses[idx];
        s.pre_idx = pre;
        s.post_idx = post;
        s.weight = weight;
        s.delay_steps = static_cast<float>(delay);
        s.last_pre_spike = -1000.0f;
        s.last_post_spike = -1000.0f;
        s.x_pre_trace = 0.0f;
        s.x_post_trace = 0.0f;
        s.nmda_conductance = 0.0f;
        s.ampa_conductance = 0.0f;
        s.ca_concentration = 0.0f;
        s.resource = 1.0f;
        // 前馈连接使用易化型 STP 基线利用率, 其他用抑郁型
        s.utilization = is_feedforward ? STP_U_FEEDFORWARD
                                        : (pre_is_exc ? STP_U_SE : STP_U_SI);
        s.eligibility = 0.0f;
        s.eligibility_slow = 0.0f;
        s.scaling_factor = 1.0f;
        s.camkii_autophosph = 0.0f;
        s.da_receptor = pre_is_exc ? DA_RECEPTOR_INIT_EXC : DA_RECEPTOR_INIT_INH;
        s.ach_receptor = ACH_RECEPTOR_INIT;
        // receptor_flags: bit0=AMPA, bit1=NMDA, bit2=GABA_A, bit3=GABA_B, bit4=前馈标志
        uint8_t flags = pre_is_exc ? 0x03 : 0x0C;  // AMPA+NMDA or GABA_A+GABA_B
        if (is_feedforward) flags |= RECEPTOR_FLAG_FEEDFORWARD;  // bit4 标记前馈连接
        s.receptor_flags = flags;
        set_ne_receptor(s, NE_RECEPTOR_INIT);
        set_ht5_receptor(s, HT5_RECEPTOR_INIT);
        s._pad = 0;
        // PSW 初始化: α/(α+β) = |w|/W_MAX, 总证据 α+β=0.1 (弱先验)
        float w_ratio = fabsf(weight) / STDP_W_MAX_2E;
        if (w_ratio > 0.999f) w_ratio = 0.999f;
        h_alpha[idx] = w_ratio * PSW_EVIDENCE_INIT_TOTAL;
        h_beta[idx]  = (1.0f - w_ratio) * PSW_EVIDENCE_INIT_TOTAL;
        h_weights_cache[idx] = weight;
        h_delay[idx] = delay;
    };

    for (int pre = 0; pre < N_TOTAL_NEURONS_2E; ++pre) {
        int start = h_row_ptr[pre];
        int deg = out_deg[pre];
        int n_written = 0;

        bool is_pf = (pre >= N_ASSOCIATION_NEURONS_2E);
        int pre_col = is_pf ? -1 : (pre / NEURONS_PER_COLUMN_2E);
        int pre_off = is_pf ? 0 : (pre % NEURONS_PER_COLUMN_2E);
        // 判断 pre 的层 (0=L4, 1=L2/3, 2=L5, 3=L6, 4=prefrontal)
        int pre_layer;
        if (is_pf) {
            pre_layer = REGION_PREFRONTAL;
        } else if (pre_off < COL_L4_SIZE_2E) {
            pre_layer = REGION_L4;
        } else if (pre_off < COL_L4_SIZE_2E + COL_L23_SIZE_2E) {
            pre_layer = REGION_L23;
        } else if (pre_off < COL_L4_SIZE_2E + COL_L23_SIZE_2E + COL_L5_SIZE_2E) {
            pre_layer = REGION_L5;
        } else {
            pre_layer = REGION_L6;
        }

        // 兴奋/抑制类型 (前 80% 兴奋, 每层内独立)
        bool pre_is_exc;
        int l_size, l_off;
        if (pre_layer == REGION_PREFRONTAL) {
            l_size = NEURONS_PER_PF_GROUP;
            l_off = (pre - N_ASSOCIATION_NEURONS_2E) % NEURONS_PER_PF_GROUP;
        } else {
            l_size = layer_sz(pre_layer);
            l_off = pre_off - layer_offset_in_col(pre_layer);
        }
        int exc_count = static_cast<int>(l_size * EXCITATORY_RATIO_2E);
        pre_is_exc = (l_off < exc_count);

        // 按出度类型分配
        int n_intra = is_pf ? 100 : 150;
        int n_inter = (pre_layer == REGION_L23) ? 30 : 0;       // 仅 L2/3 有跨柱突触
        int n_pf    = is_pf ? 50 : (pre_layer == REGION_L5 ? 15 : 0);  // 仅 L5 投射前额叶

        // ---- 柱内目标 (n_intra 个, 按流向规则) ----
        for (int k = 0; k < n_intra && n_written < deg; ++k) {
            int post;
            bool is_feedforward = false;
            if (is_pf) {
                // 前额叶 → 同组前额叶 (自反馈)
                int group_base = N_ASSOCIATION_NEURONS_2E +
                    ((pre - N_ASSOCIATION_NEURONS_2E) / NEURONS_PER_PF_GROUP) * NEURONS_PER_PF_GROUP;
                post = group_base + randi(0, NEURONS_PER_PF_GROUP);
            } else {
                int col_base = pre_col * NEURONS_PER_COLUMN_2E;
                int target_layer = pick_target_layer(pre_layer);
                post = pick_intra_post(col_base, target_layer);
                // 前馈连接判断 (L4→L2/3, L2/3→L5, L5→L6): 使用增强权重打破"鸡生蛋"困境
                is_feedforward = (pre_layer == REGION_L4  && target_layer == REGION_L23) ||
                                 (pre_layer == REGION_L23 && target_layer == REGION_L5)  ||
                                 (pre_layer == REGION_L5  && target_layer == REGION_L6);
            }
            // 避免自环
            if (post == pre) post = (post + 1) % N_TOTAL_NEURONS_2E;

            int idx = start + n_written;
            h_col_idx[idx] = post;
            // 权重: 前馈连接 [2.5,3.5], 其他 [0.4,1.0] (1/√K 缩放)
            float w_exc_min = is_feedforward ? FEEDFORWARD_W_EXC_MIN : 0.4f;
            float w_exc_max = is_feedforward ? FEEDFORWARD_W_EXC_MAX : 1.0f;
            float w = pre_is_exc ? randf(w_exc_min, w_exc_max) * w_scale
                                  : randf(-w_exc_max, -w_exc_min) * w_scale;
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTRA_MIN, DELAY_INTRA_MAX + 1));
            init_syn_fields(idx, pre, post, w, delay, pre_is_exc, is_feedforward);
            n_written++;
        }

        // ---- 跨柱目标 (n_inter 个, 仅 L2/3 → L2/3) ----
        for (int k = 0; k < n_inter && n_written < deg; ++k) {
            int target_col = randi(0, N_COLUMNS_2E);
            if (target_col == pre_col) target_col = (target_col + 1) % N_COLUMNS_2E;

            // 跨柱约束: post 必须是目标柱的 L2/3 层 (pre 已经是 L2/3)
            int col_base = target_col * NEURONS_PER_COLUMN_2E;
            int post = col_base + layer_offset_in_col(REGION_L23) +
                       randi(0, layer_sz(REGION_L23));

            int idx = start + n_written;
            h_col_idx[idx] = post;
            // 跨柱大幅弱化 (R1 消融实验: [0.05,0.15]/[-0.15,-0.05], 抑制跨柱兴奋传播)
            float w = pre_is_exc ? randf(CROSS_COL_W_EXC_MIN, CROSS_COL_W_EXC_MAX) * w_scale
                                  : randf(CROSS_COL_W_INH_MIN, CROSS_COL_W_INH_MAX) * w_scale;
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTER_MIN, DELAY_INTER_MAX + 1));
            init_syn_fields(idx, pre, post, w, delay, pre_is_exc);
            n_written++;
        }

        // ---- 前额叶投射 (n_pf 个) ----
        for (int k = 0; k < n_pf && n_written < deg; ++k) {
            int post;
            uint8_t delay;
            if (is_pf) {
                // 前额叶 → 联合皮层 (长程投射)
                post = randi(0, N_ASSOCIATION_NEURONS_2E);
                delay = static_cast<uint8_t>(randi(DELAY_LONG_MIN, DELAY_LONG_MAX + 1));
            } else {
                // L5 → 前额叶 (长程投射, 替代旧 association → 前额叶)
                int pf_group = randi(0, PREFRONTAL_GROUPS);
                post = N_ASSOCIATION_NEURONS_2E + pf_group * NEURONS_PER_PF_GROUP +
                       randi(0, NEURONS_PER_PF_GROUP);
                delay = static_cast<uint8_t>(randi(DELAY_LONG_MIN, DELAY_LONG_MAX + 1));
            }

            int idx = start + n_written;
            h_col_idx[idx] = post;
            // 长程投射 (前额叶) - 维持与柱内同量级以保持信号传播
            float w = pre_is_exc ? randf(0.6f, 1.2f) * w_scale
                                  : randf(-1.2f, -0.6f) * w_scale;
            init_syn_fields(idx, pre, post, w, delay, pre_is_exc);
            n_written++;
        }

        // ---- 剩余突触 (补足出度, 用柱内流向规则填充) ----
        while (n_written < deg) {
            int post;
            bool is_feedforward = false;
            if (is_pf) {
                post = N_ASSOCIATION_NEURONS_2E + randi(0, N_PREFRONTAL_NEURONS);
            } else {
                int col_base = pre_col * NEURONS_PER_COLUMN_2E;
                int target_layer = pick_target_layer(pre_layer);
                post = pick_intra_post(col_base, target_layer);
                is_feedforward = (pre_layer == REGION_L4  && target_layer == REGION_L23) ||
                                 (pre_layer == REGION_L23 && target_layer == REGION_L5)  ||
                                 (pre_layer == REGION_L5  && target_layer == REGION_L6);
            }
            if (post == pre) post = (post + 1) % N_TOTAL_NEURONS_2E;

            int idx = start + n_written;
            h_col_idx[idx] = post;
            float w_exc_min = is_feedforward ? FEEDFORWARD_W_EXC_MIN : 0.4f;
            float w_exc_max = is_feedforward ? FEEDFORWARD_W_EXC_MAX : 1.0f;
            float w = pre_is_exc ? randf(w_exc_min, w_exc_max) * w_scale
                                  : randf(-w_exc_max, -w_exc_min) * w_scale;
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTRA_MIN, DELAY_INTRA_MAX + 1));
            init_syn_fields(idx, pre, post, w, delay, pre_is_exc, is_feedforward);
            n_written++;
        }
    }
}

// -----------------------------------------------------------------------------
// Host: 初始化突触 (上传到 GPU)
// -----------------------------------------------------------------------------
int init_synapses(BioSynapse* d_synapses,
                  int* d_csr_row_ptr,
                  int* d_csr_col_idx,
                  float* d_weights_cache,
                  uint8_t* d_synapse_delay,
                   float* d_synapse_alpha,
                   float* d_synapse_beta,
                   const NeuronStateAdEx* d_neurons,
                   uint32_t seed) {
    (void)d_neurons;  // P1 不依赖 d_neurons 内容做初始化

    // Host 端构造
    std::vector<BioSynapse> h_syn;
    std::vector<int> h_row;
    std::vector<int> h_col;
    std::vector<float> h_w;
    std::vector<uint8_t> h_d;
    std::vector<float> h_alpha;
    std::vector<float> h_beta;

    printf("[Stage2e P1] 生成突触拓扑 (host 端, ~10.7M 突触)...\n");
    init_synapses_host(h_syn, h_row, h_col, h_w, h_d, h_alpha, h_beta, seed);

    int n_syn = static_cast<int>(h_syn.size());
    printf("[Stage2e P1] 实际生成: %d 突触, %d row_ptr 项\n", n_syn, static_cast<int>(h_row.size()));

    // 上传 GPU
    CUDA_CHECK_2E(cudaMemcpy(d_synapses, h_syn.data(),
                              n_syn * sizeof(BioSynapse),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_csr_row_ptr, h_row.data(),
                              h_row.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_csr_col_idx, h_col.data(),
                              n_syn * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_weights_cache, h_w.data(),
                              n_syn * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_synapse_delay, h_d.data(),
                              n_syn * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
    // PSW: 上传 alpha/beta 数组
    CUDA_CHECK_2E(cudaMemcpy(d_synapse_alpha, h_alpha.data(),
                              n_syn * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_synapse_beta, h_beta.data(),
                              n_syn * sizeof(float),
                              cudaMemcpyHostToDevice));
    printf("[Stage2e P1] 突触拓扑已上传 GPU\n");
    return n_syn;
}

// -----------------------------------------------------------------------------
// Kernel: 初始化运动皮层 AdEx 神经元 (5K, 静息电位 -70mV)
// 与现有 init_neurons_kernel 模式一致, 但 region = REGION_MOTOR
// 神经元布局: 50 群 × 100 神经元 = 5000, 每群对应一个柱的 L5 输出
// -----------------------------------------------------------------------------
__global__ void init_motor_neurons_kernel(NeuronStateAdEx* neurons) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_MOTOR_NEURONS) return;

    NeuronStateAdEx& n = neurons[i];

    // AdEx 静息状态 (V_norm = 0, 对应 V_bio = -70 mV)
    n.membrane_potential   = 0.0f;
    n.synaptic_current     = 0.0f;
    n.nmda_current         = 0.0f;
    n.adaptive_conductance = 0.0f;       // w = 0 (无初始适应)
    n.last_spike_time      = -1000;       // 远古值
    n.fire_rate            = 0.0f;
    n.refractory_remaining = 0;
    n.ca_neuron            = 0.0f;
    n.wm_injection         = 0.0f;
    n.homeostatic_factor   = 1.0f;       // 初始缩放 = 1.0

    // 运动皮层分组: 50 群 × 100 神经元, 每群对应一个柱 (column_id = motor_group)
    int motor_group  = i / MOTOR_GROUP_SIZE;     // 0..49
    int off_in_group = i % MOTOR_GROUP_SIZE;     // 0..99
    n.column_id          = static_cast<uint8_t>(motor_group);
    n.pf_group_id        = -1;                    // 非前额叶
    n.region             = REGION_MOTOR;           // 5 = 运动皮层

    // 80/20 兴奋/抑制 (每群内独立维持, 前 80% 兴奋, 与现有层内分配模式一致)
    int exc_count = static_cast<int>(MOTOR_GROUP_SIZE * EXCITATORY_RATIO_2E);
    if (off_in_group < exc_count) {
        n.neuron_type         = 0;   // EXCITATORY
        n.inhibitory_subtype  = 0;   // NONE
    } else {
        n.neuron_type         = 1;   // INHIBITORY
        n.inhibitory_subtype  = 1;   // FS (运动皮层输出神经元主要为 PV+ 快速放电型)
    }

    // 阈值偏移初始 0
    n.threshold_offset = 0;
    n._reserved = 0;
    n._pad = 0;
}

// -----------------------------------------------------------------------------
// Host: 初始化运动皮层神经元
// -----------------------------------------------------------------------------
void init_motor_neurons(NeuronStateAdEx* d_motor_neurons) {
    int blocks = (N_MOTOR_NEURONS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    init_motor_neurons_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(d_motor_neurons);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Host: 生成 L5 → 运动皮层稀疏 CSR 突触
// -----------------------------------------------------------------------------
// 拓扑策略:
//   - 每个运动神经元 (全局索引 motor_global_base + i, i ∈ [0, 5000))
//     接收来自对应柱 (i / MOTOR_GROUP_SIZE) L5 层的 50 个突触
//   - L5 层柱内偏移: COL_L4_SIZE_2E + COL_L23_SIZE_2E = 550 (200 个 L5 神经元)
//   - 从该柱 L5 层 (200 个神经元) 随机选 50 个作为突触前神经元
//   - 突触类型: 兴奋性 (AMPA + NMDA, receptor_flags = 0x03)
//   - 初始权重: [0.3, 0.8] 均匀分布
//   - 延迟: 1-3 步 (柱内延迟, 与 DELAY_INTRA_MIN/MAX 一致)
//   CSR 布局:
//   - row_ptr[i] = i × L5_TO_MOTOR_SYNAPSES_PER_NEURON  (每个运动神经元固定 50 入度)
//   - col_idx[row_ptr[i] + k] = L5 突触前神经元索引
//   总突触数 = N_MOTOR_NEURONS × L5_TO_MOTOR_SYNAPSES_PER_NEURON = 5000 × 50 = 250,000
// -----------------------------------------------------------------------------
int init_l5_to_motor_synapses(BioSynapse* d_l5_to_motor_synapses,
                              float* d_l5_to_motor_weights,
                              int* d_l5_to_motor_csr_row_ptr,
                              int* d_l5_to_motor_csr_col_idx,
                              uint32_t seed) {
    // 简单的确定性 PRNG (与 init_synapses_host 一致, xorshift32)
    uint32_t rng = seed == 0 ? 0x6D2B79F5u : seed;
    auto next_rng = [&rng]() -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    auto randf = [&next_rng](float lo, float hi) -> float {
        return lo + (hi - lo) * (next_rng() / 4294967296.0f);
    };
    auto randi = [&next_rng](int lo, int hi) -> int {  // [lo, hi)
        return lo + static_cast<int>(next_rng() % static_cast<uint32_t>(hi - lo));
    };

    const int n_syn = N_MOTOR_NEURONS * L5_TO_MOTOR_SYNAPSES_PER_NEURON;  // 250,000

    // Host 端构造
    std::vector<BioSynapse> h_syn(n_syn);
    std::vector<float>      h_w(n_syn);
    std::vector<int>        h_row(N_MOTOR_NEURONS + 1);
    std::vector<int>        h_col(n_syn);

    // L5 层在柱内的偏移: COL_L4_SIZE_2E + COL_L23_SIZE_2E = 200 + 350 = 550
    // L5 层大小: COL_L5_SIZE_2E = 200 (柱内 [550, 750))
    const int l5_offset_in_col = COL_L4_SIZE_2E + COL_L23_SIZE_2E;
    const int l5_size          = COL_L5_SIZE_2E;

    // 运动神经元全局起始索引: N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS = 55,000
    const int motor_global_base = N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS;

    // CSR row_ptr: 每个运动神经元固定接收 L5_TO_MOTOR_SYNAPSES_PER_NEURON 个突触
    h_row[0] = 0;
    for (int i = 0; i < N_MOTOR_NEURONS; ++i) {
        h_row[i + 1] = h_row[i] + L5_TO_MOTOR_SYNAPSES_PER_NEURON;
    }

    // 生成突触
    for (int i = 0; i < N_MOTOR_NEURONS; ++i) {
        // 运动神经元 i 对应的柱 (i / MOTOR_GROUP_SIZE = i / 100)
        int col      = i / MOTOR_GROUP_SIZE;
        int col_base = col * NEURONS_PER_COLUMN_2E;
        int l5_base  = col_base + l5_offset_in_col;  // 该柱 L5 层基址

        // 运动神经元全局索引 (用于 BioSynapse.post_idx)
        int post_global = motor_global_base + i;

        int start = h_row[i];
        for (int k = 0; k < L5_TO_MOTOR_SYNAPSES_PER_NEURON; ++k) {
            // 从该柱 L5 层 (200 个神经元) 随机选一个作为突触前神经元
            int pre_local = randi(0, l5_size);
            int pre       = l5_base + pre_local;

            int idx = start + k;
            h_col[idx] = pre;

            // 权重 [0.3, 0.8] 均匀分布
            float w = randf(0.3f, 0.8f);
            h_w[idx] = w;

            // 初始化 BioSynapse 字段 (兴奋性: AMPA + NMDA)
            BioSynapse& s = h_syn[idx];
            s.pre_idx       = pre;
            s.post_idx      = post_global;
            s.weight        = w;
            // 延迟 1-3 步 (柱内延迟)
            s.delay_steps   = static_cast<float>(randi(DELAY_INTRA_MIN, DELAY_INTRA_MAX + 1));
            s.last_pre_spike  = -1000.0f;
            s.last_post_spike = -1000.0f;
            s.x_pre_trace     = 0.0f;
            s.x_post_trace    = 0.0f;
            s.nmda_conductance = 0.0f;
            s.ampa_conductance = 0.0f;
            s.ca_concentration = 0.0f;
            s.resource        = 1.0f;
            s.utilization     = STP_U_SE;             // 兴奋性基线利用率 U
            s.eligibility     = 0.0f;
            s.eligibility_slow = 0.0f;
            s.scaling_factor = 1.0f;
            s.camkii_autophosph = 0.0f;
            s.da_receptor    = DA_RECEPTOR_INIT_EXC;  // 兴奋性突触偏向 D1
            s.ach_receptor   = ACH_RECEPTOR_INIT;
            s.receptor_flags = 0x03;                   // AMPA + NMDA (兴奋性)
            set_ne_receptor(s, NE_RECEPTOR_INIT);
            set_ht5_receptor(s, HT5_RECEPTOR_INIT);
            s._pad = 0;
        }
    }

    // 上传 GPU
    CUDA_CHECK_2E(cudaMemcpy(d_l5_to_motor_synapses, h_syn.data(),
                              n_syn * sizeof(BioSynapse),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_l5_to_motor_weights, h_w.data(),
                              n_syn * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_l5_to_motor_csr_row_ptr, h_row.data(),
                              h_row.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK_2E(cudaMemcpy(d_l5_to_motor_csr_col_idx, h_col.data(),
                              n_syn * sizeof(int),
                              cudaMemcpyHostToDevice));

    return n_syn;
}

// -----------------------------------------------------------------------------
// Host: 初始化零缓冲 (调质, 输入电流, NMDA, etc.)
// -----------------------------------------------------------------------------
void init_buffers_zero(MemoryAllocator* alloc) {
    PersistentBuffers& b = alloc->buffers();
    CUDA_CHECK_2E(cudaMemset(b.d_input_current, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_nmda_current, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_inhibitory_current, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_da_concentration, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_ach_concentration, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_ne_concentration, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_ht5_concentration, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_spike_flags, 0, N_TOTAL_NEURONS_2E * sizeof(bool)));
    CUDA_CHECK_2E(cudaMemset(b.d_replay_injection, 0, N_TOTAL_NEURONS_2E * sizeof(float)));
    // 延迟环形队列清零
    CUDA_CHECK_2E(cudaMemset(b.d_delay_ring_indices, 0xFF,
                             DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY * sizeof(int)));
    CUDA_CHECK_2E(cudaMemset(b.d_delay_ring_current, 0, 20 * 500000 * sizeof(float)));
    // STDP trace 清零
    CUDA_CHECK_2E(cudaMemset(b.d_stdp_x_pre_trace, 0, N_TOTAL_SYNAPSES_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_camkii_activity, 0, N_TOTAL_SYNAPSES_2E * sizeof(float)));
    // Eligibility 清零
    CUDA_CHECK_2E(cudaMemset(b.d_eligibility, 0, N_TOTAL_SYNAPSES_2E * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_eligibility_slow, 0, N_TOTAL_SYNAPSES_2E * sizeof(float)));
    // 钙快照
    CUDA_CHECK_2E(cudaMemset(b.d_ca_snapshot, 0, N_TOTAL_SYNAPSES_2E * sizeof(float)));
    // 海马 / WM / 共激活 / DA 价值函数 / 字节直方图
    CUDA_CHECK_2E(cudaMemset(b.d_hippo_indices, 0, HIPP_INDEX_SIZE * sizeof(HippoIndex)));
    CUDA_CHECK_2E(cudaMemset(b.d_coact_trackers, 0, COACT_TRACKER_SIZE * sizeof(CoactTracker)));
    CUDA_CHECK_2E(cudaMemset(b.d_wm_slots, 0, WM_SLOTS * sizeof(WMSlot)));
    CUDA_CHECK_2E(cudaMemset(b.d_subcolumn_fr, 0, W_VALUE_DIM * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_baseline_fr, 0, W_VALUE_DIM * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_w_pred, 0, W_PRED_DIM * W_VALUE_DIM * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_w_value, 0, W_VALUE_DIM * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_pred_fr, 0, W_VALUE_DIM * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_byte_histogram, 0, 256 * sizeof(int)));
    // PCA 矩阵清零 (留待后续 PCA 训练)
    CUDA_CHECK_2E(cudaMemset(b.d_pca_W, 0, N_TOTAL_NEURONS_2E * PATTERN_DIM * sizeof(float)));
}

// -----------------------------------------------------------------------------
// Host: 完整初始化入口
// -----------------------------------------------------------------------------
void init_network(MemoryAllocator* alloc, uint32_t seed) {
    printf("[Stage2e P1] === 网络初始化 ===\n");

    PersistentBuffers& b = alloc->buffers();

    printf("[Stage2e P1] 初始化神经元 (60K AdEx 静息)...\n");
    init_neurons(b.d_neurons);

    printf("[Stage2e P1] 初始化突触拓扑 + 延迟 + STP + 调质受体 + PSW...\n");
    int n_syn = init_synapses(b.d_synapses, b.d_csr_row_ptr, b.d_csr_col_idx,
                              b.d_weights_cache, b.d_synapse_delay,
                               b.d_synapse_alpha, b.d_synapse_beta,
                               b.d_neurons, seed);
    if (n_syn != N_TOTAL_SYNAPSES_2E) {
        fprintf(stderr, "[Stage2e P1 WARN] 突触数 %d != 目标 %d\n", n_syn, N_TOTAL_SYNAPSES_2E);
    }

    printf("[Stage2e P1] 初始化缓冲为零...\n");
    init_buffers_zero(alloc);

    // -----------------------------------------------------------------
    // 语言运动皮层初始化 (在现有网络初始化之后)
    // -----------------------------------------------------------------
    printf("[Stage2e P1] 初始化运动皮层神经元 (5K AdEx 静息, region=MOTOR)...\n");
    init_motor_neurons(b.d_motor_neurons);

    printf("[Stage2e P1] 生成 L5 → 运动皮层稀疏 CSR 突触 (250K 突触)...\n");
    int n_l5_motor_syn = init_l5_to_motor_synapses(b.d_l5_to_motor_synapses,
                                                   b.d_l5_to_motor_weights,
                                                   b.d_l5_to_motor_csr_row_ptr,
                                                   b.d_l5_to_motor_csr_col_idx,
                                                   seed);
    const int expected_l5_motor_syn = N_MOTOR_NEURONS * L5_TO_MOTOR_SYNAPSES_PER_NEURON;
    if (n_l5_motor_syn != expected_l5_motor_syn) {
        fprintf(stderr, "[Stage2e P1 WARN] L5→Motor 突触数 %d != 目标 %d\n",
                n_l5_motor_syn, expected_l5_motor_syn);
    }

    // 零初始化解码权重矩阵 (虽然 alloc 已清零, 显式 memset 强调语义: 解码器从零权重起步)
    printf("[Stage2e P1] 零初始化解码权重矩阵 (60K × 256 = %.2f MB)...\n",
           (size_t)N_TOTAL_NEURONS_2E * 256 * sizeof(float) / (1024.0 * 1024.0));
    CUDA_CHECK_2E(cudaMemset(b.d_decode_weights, 0,
                              (size_t)N_TOTAL_NEURONS_2E * 256 * sizeof(float)));

    printf("[Stage2e P1] 网络初始化完成\n");
}

} // namespace stage2e
