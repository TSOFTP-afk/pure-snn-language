// =============================================================================
// Stage 2e 网络初始化实现 (P1)
// =============================================================================
// 设计要点:
//   1. 神经元布局: 50 柱 × 1000 神经元 = 50K 联合皮层 + 5K 前额叶
//      - 柱 c 的 sensory 层:    [c*1000,       c*1000 + 200)
//      - 柱 c 的 association 层: [c*1000 + 200, c*1000 + 800)
//      - 柱 c 的 motor 层:       [c*1000 + 800, c*1000 + 1000)
//      - 前额叶:                 [50000, 55000)
//   2. 80/20 兴奋/抑制: 每层前 80% 兴奋, 后 20% 抑制
//   3. 抑制亚型: 抑制性中 FS:LTS:SOM = 50:30:20
//   4. 突触拓扑 (host 端生成, 一次性上传 GPU):
//      - 柱内: 每神经元 ~150 个柱内突触 (sensory→assoc→motor 流向 + 同层横向)
//      - 跨柱: 每神经元 ~30 个跨柱突触 (随机选目标柱)
//      - 前额叶: 自反馈 + 接收 assoc 投射
//      - 总数 ~ 55K × 195 ≈ 10.7M
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
        // 联合皮层 (50 柱 × 1000)
        int col = i / NEURONS_PER_COLUMN_2E;       // 0..49
        int off = i % NEURONS_PER_COLUMN_2E;       // 0..999
        n.column_id = static_cast<uint8_t>(col);
        n.pf_group_id = -1;

        if (off < COL_SENSORY_SIZE_2E) {
            n.region = 0;  // SENSORY
        } else if (off < COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E) {
            n.region = 1;  // ASSOCIATION
        } else {
            n.region = 2;  // MOTOR
        }

        // 80/20 兴奋/抑制 (每层内前 80% 兴奋)
        int layer_off, layer_size;
        if (off < COL_SENSORY_SIZE_2E) {
            layer_off = off; layer_size = COL_SENSORY_SIZE_2E;
        } else if (off < COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E) {
            layer_off = off - COL_SENSORY_SIZE_2E;
            layer_size = COL_ASSOCIATION_SIZE_2E;
        } else {
            layer_off = off - COL_SENSORY_SIZE_2E - COL_ASSOCIATION_SIZE_2E;
            layer_size = COL_MOTOR_SIZE_2E;
        }
        int exc_count = static_cast<int>(layer_size * EXCITATORY_RATIO_2E);
        if (layer_off < exc_count) {
            n.neuron_type = 0;  // EXCITATORY
            n.inhibitory_subtype = 0;  // NONE
        } else {
            n.neuron_type = 1;  // INHIBITORY
            // 抑制亚型分配: 50% FS, 30% LTS, 20% SOM
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
        n.region = 3;            // PREFRONTAL
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
// 拓扑策略 (设计文档 §2.4 + v3 柱状拓扑):
//   每神经元目标出度 ~ 195 (10.7M / 55K ≈ 194.5)
//     - 柱内突触: 150 (sensory→assoc→motor 流向 + 横向)
//     - 跨柱突触: 30  (随机目标柱, 同流向)
//     - 前额叶投射: 15 (assoc → 前额叶)
//   延迟:
//     - 柱内: 1-3 步
//     - 跨柱: 5-10 步
//     - 长程(前额叶): 15-20 步
// -----------------------------------------------------------------------------
void init_synapses_host(std::vector<BioSynapse>& h_synapses,
                       std::vector<int>& h_row_ptr,
                       std::vector<int>& h_col_idx,
                       std::vector<float>& h_weights_cache,
                       std::vector<uint8_t>& h_delay) {
    // 简单的确定性 PRNG (避免引入 curand 依赖到 host)
    // 使用 xorshift32, 种子 = 42
    uint32_t rng = 42;
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

    h_row_ptr.assign(N_TOTAL_NEURONS_2E + 1, 0);
    h_synapses.clear();
    h_synapses.reserve(N_TOTAL_SYNAPSES_2E);
    h_col_idx.clear(); h_col_idx.reserve(N_TOTAL_SYNAPSES_2E);
    h_weights_cache.clear(); h_weights_cache.reserve(N_TOTAL_SYNAPSES_2E);
    h_delay.clear(); h_delay.reserve(N_TOTAL_SYNAPSES_2E);

    // 临时统计每神经元出度
    std::vector<int> out_deg(N_TOTAL_NEURONS_2E, 0);
    int total_target = 0;

    // 预计算每神经元的出度
    for (int pre = 0; pre < N_TOTAL_NEURONS_2E; ++pre) {
        int deg = 0;
        if (pre < N_ASSOCIATION_NEURONS_2E) {
            // 联合皮层: 150 柱内 + 30 跨柱 + 15 前额叶投射 (仅 assoc 层)
            deg = 150 + 30;
            int off = pre % NEURONS_PER_COLUMN_2E;
            if (off >= COL_SENSORY_SIZE_2E && off < COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E) {
                deg += 15;  // 仅 association 层投射到前额叶
            }
        } else {
            // 前额叶: 100 自反馈 + 50 接收
            deg = 100 + 50;
        }
        out_deg[pre] = deg;
        total_target += deg;
    }

    // 调整为精确 N_TOTAL_SYNAPSES_2E
    // 如果 total_target > N_TOTAL_SYNAPSES_2E, 截断
    // 如果 < , 补足 (给联合皮层加突触)
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

    // 填充 row_ptr (CSR: row_ptr[0]=0, row_ptr[i+1]-row_ptr[i]=out_deg[i])
    h_row_ptr[0] = 0;
    for (int i = 0; i < N_TOTAL_NEURONS_2E; ++i) {
        h_row_ptr[i + 1] = h_row_ptr[i] + out_deg[i];
    }

    // 生成突触
    h_synapses.resize(N_TOTAL_SYNAPSES_2E);
    h_col_idx.resize(N_TOTAL_SYNAPSES_2E);
    h_weights_cache.resize(N_TOTAL_SYNAPSES_2E);
    h_delay.resize(N_TOTAL_SYNAPSES_2E);

    for (int pre = 0; pre < N_TOTAL_NEURONS_2E; ++pre) {
        int start = h_row_ptr[pre];
        int deg = out_deg[pre];
        int n_written = 0;

        bool is_pf = (pre >= N_ASSOCIATION_NEURONS_2E);
        int pre_col = is_pf ? -1 : (pre / NEURONS_PER_COLUMN_2E);
        int pre_off = is_pf ? 0 : (pre % NEURONS_PER_COLUMN_2E);
        // 判断 pre 的层
        int pre_layer;  // 0=sensory, 1=assoc, 2=motor, 3=prefrontal
        if (is_pf) {
            pre_layer = 3;
        } else if (pre_off < COL_SENSORY_SIZE_2E) {
            pre_layer = 0;
        } else if (pre_off < COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E) {
            pre_layer = 1;
        } else {
            pre_layer = 2;
        }

        // 兴奋/抑制类型 (前 80% 兴奋)
        bool pre_is_exc;
        int layer_size, layer_off;
        if (pre_layer == 3) {
            layer_size = NEURONS_PER_PF_GROUP;
            layer_off = (pre - N_ASSOCIATION_NEURONS_2E) % NEURONS_PER_PF_GROUP;
        } else if (pre_layer == 0) {
            layer_size = COL_SENSORY_SIZE_2E;
            layer_off = pre_off;
        } else if (pre_layer == 1) {
            layer_size = COL_ASSOCIATION_SIZE_2E;
            layer_off = pre_off - COL_SENSORY_SIZE_2E;
        } else {
            layer_size = COL_MOTOR_SIZE_2E;
            layer_off = pre_off - COL_SENSORY_SIZE_2E - COL_ASSOCIATION_SIZE_2E;
        }
        int exc_count = static_cast<int>(layer_size * EXCITATORY_RATIO_2E);
        pre_is_exc = (layer_off < exc_count);

        // 按出度类型分配目标
        // 联合皮层: 150 柱内 + 30 跨柱 + 15 前额叶 (仅 assoc)
        // 前额叶: 100 自反馈 + 50 到联合皮层
        int n_intra = is_pf ? 100 : 150;
        int n_inter = is_pf ? 0  : 30;
        int n_pf    = is_pf ? 50 : (pre_layer == 1 ? 15 : 0);

        // 柱内目标 (n_intra 个)
        for (int k = 0; k < n_intra && n_written < deg; ++k) {
            int post;
            int target_layer;
            // 流向: sensory→assoc, assoc→motor, motor→assoc (循环), 前额叶→前额叶
            if (pre_layer == 0) {
                target_layer = 1;  // sensory → assoc
            } else if (pre_layer == 1) {
                target_layer = 2;  // assoc → motor
            } else if (pre_layer == 2) {
                target_layer = 1;  // motor → assoc
            } else {
                target_layer = 3;  // pf → pf
            }

            if (is_pf) {
                int group_base = N_ASSOCIATION_NEURONS_2E +
                    ((pre - N_ASSOCIATION_NEURONS_2E) / NEURONS_PER_PF_GROUP) * NEURONS_PER_PF_GROUP;
                post = group_base + randi(0, NEURONS_PER_PF_GROUP);
            } else {
                int col_base = pre_col * NEURONS_PER_COLUMN_2E;
                if (target_layer == 0) {
                    post = col_base + randi(0, COL_SENSORY_SIZE_2E);
                } else if (target_layer == 1) {
                    post = col_base + COL_SENSORY_SIZE_2E + randi(0, COL_ASSOCIATION_SIZE_2E);
                } else {
                    post = col_base + COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E + randi(0, COL_MOTOR_SIZE_2E);
                }
            }

            // 避免自环
            if (post == pre) post = (post + 1) % N_TOTAL_NEURONS_2E;

            int idx = start + n_written;
            h_col_idx[idx] = post;

            // 初始化 BioSynapse 字段
            BioSynapse& s = h_synapses[idx];
            s.pre_idx = pre;
            s.post_idx = post;
            // 权重: 兴奋性 [0.1, 0.5], 抑制性 [-0.5, -0.1]
            if (pre_is_exc) {
                s.weight = randf(0.1f, 0.5f);
            } else {
                s.weight = randf(-0.5f, -0.1f);
            }
            // 延迟: 柱内 1-3, 前额叶自反馈 1-3
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTRA_MIN, DELAY_INTRA_MAX + 1));
            s.delay_steps = static_cast<float>(delay);
            h_delay[idx] = delay;

            // STDP 双 trace
            s.last_pre_spike = -1000.0f;
            s.last_post_spike = -1000.0f;
            s.x_pre_trace = 0.0f;
            s.x_post_trace = 0.0f;

            // 电导 + 钙
            s.nmda_conductance = 0.0f;
            s.ampa_conductance = 0.0f;
            s.ca_concentration = 0.0f;
            // STP 资源 R=1 (满), 利用率 U = 兴奋 0.2 / 抑制 0.05
            s.resource = 1.0f;
            s.utilization = pre_is_exc ? STP_U_SE : STP_U_SI;

            // Eligibility
            s.eligibility = 0.0f;
            s.eligibility_slow = 0.0f;
            s.scaling_factor = 1.0f;

            // CaMKII + 受体
            s.camkii_autophosph = 0.0f;
            s.da_receptor = pre_is_exc ? DA_RECEPTOR_INIT_EXC : DA_RECEPTOR_INIT_INH;
            s.ach_receptor = ACH_RECEPTOR_INIT;
            s.receptor_flags = pre_is_exc ? 0x03 : 0x0C;  // AMPA+NMDA or GABA_A+GABA_B
            set_ne_receptor(s, NE_RECEPTOR_INIT);
            set_ht5_receptor(s, HT5_RECEPTOR_INIT);
            s._pad = 0;

            h_weights_cache[idx] = s.weight;
            n_written++;
        }

        // 跨柱目标 (n_inter 个, 仅联合皮层)
        for (int k = 0; k < n_inter && n_written < deg; ++k) {
            int target_col = randi(0, N_COLUMNS_2E);
            if (target_col == pre_col) target_col = (target_col + 1) % N_COLUMNS_2E;

            int col_base = target_col * NEURONS_PER_COLUMN_2E;
            // 跨柱投射: 同层为主 (sensory→sensory, assoc→assoc)
            int post;
            if (pre_layer == 0) {
                post = col_base + randi(0, COL_SENSORY_SIZE_2E);
            } else if (pre_layer == 1) {
                post = col_base + COL_SENSORY_SIZE_2E + randi(0, COL_ASSOCIATION_SIZE_2E);
            } else {
                post = col_base + COL_SENSORY_SIZE_2E + COL_ASSOCIATION_SIZE_2E + randi(0, COL_MOTOR_SIZE_2E);
            }

            int idx = start + n_written;
            h_col_idx[idx] = post;

            BioSynapse& s = h_synapses[idx];
            s.pre_idx = pre;
            s.post_idx = post;
            if (pre_is_exc) {
                s.weight = randf(0.05f, 0.3f);  // 跨柱较弱
            } else {
                s.weight = randf(-0.3f, -0.05f);
            }
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTER_MIN, DELAY_INTER_MAX + 1));
            s.delay_steps = static_cast<float>(delay);
            h_delay[idx] = delay;

            s.last_pre_spike = -1000.0f;
            s.last_post_spike = -1000.0f;
            s.x_pre_trace = 0.0f;
            s.x_post_trace = 0.0f;
            s.nmda_conductance = 0.0f;
            s.ampa_conductance = 0.0f;
            s.ca_concentration = 0.0f;
            s.resource = 1.0f;
            s.utilization = pre_is_exc ? STP_U_SE : STP_U_SI;
            s.eligibility = 0.0f;
            s.eligibility_slow = 0.0f;
            s.scaling_factor = 1.0f;
            s.camkii_autophosph = 0.0f;
            s.da_receptor = pre_is_exc ? DA_RECEPTOR_INIT_EXC : DA_RECEPTOR_INIT_INH;
            s.ach_receptor = ACH_RECEPTOR_INIT;
            s.receptor_flags = pre_is_exc ? 0x03 : 0x0C;
            set_ne_receptor(s, NE_RECEPTOR_INIT);
            set_ht5_receptor(s, HT5_RECEPTOR_INIT);
            s._pad = 0;

            h_weights_cache[idx] = s.weight;
            n_written++;
        }

        // 前额叶投射 (n_pf 个)
        for (int k = 0; k < n_pf && n_written < deg; ++k) {
            int post;
            uint8_t delay;
            if (is_pf) {
                // 前额叶 → 联合皮层 (长程投射)
                post = randi(0, N_ASSOCIATION_NEURONS_2E);
                delay = static_cast<uint8_t>(randi(DELAY_LONG_MIN, DELAY_LONG_MAX + 1));
            } else {
                // association → 前额叶 (长程投射)
                int pf_group = randi(0, PREFRONTAL_GROUPS);
                post = N_ASSOCIATION_NEURONS_2E + pf_group * NEURONS_PER_PF_GROUP +
                       randi(0, NEURONS_PER_PF_GROUP);
                delay = static_cast<uint8_t>(randi(DELAY_LONG_MIN, DELAY_LONG_MAX + 1));
            }

            int idx = start + n_written;
            h_col_idx[idx] = post;

            BioSynapse& s = h_synapses[idx];
            s.pre_idx = pre;
            s.post_idx = post;
            if (pre_is_exc) {
                s.weight = randf(0.1f, 0.4f);
            } else {
                s.weight = randf(-0.4f, -0.1f);
            }
            s.delay_steps = static_cast<float>(delay);
            h_delay[idx] = delay;

            s.last_pre_spike = -1000.0f;
            s.last_post_spike = -1000.0f;
            s.x_pre_trace = 0.0f;
            s.x_post_trace = 0.0f;
            s.nmda_conductance = 0.0f;
            s.ampa_conductance = 0.0f;
            s.ca_concentration = 0.0f;
            s.resource = 1.0f;
            s.utilization = pre_is_exc ? STP_U_SE : STP_U_SI;
            s.eligibility = 0.0f;
            s.eligibility_slow = 0.0f;
            s.scaling_factor = 1.0f;
            s.camkii_autophosph = 0.0f;
            s.da_receptor = pre_is_exc ? DA_RECEPTOR_INIT_EXC : DA_RECEPTOR_INIT_INH;
            s.ach_receptor = ACH_RECEPTOR_INIT;
            s.receptor_flags = pre_is_exc ? 0x03 : 0x0C;
            set_ne_receptor(s, NE_RECEPTOR_INIT);
            set_ht5_receptor(s, HT5_RECEPTOR_INIT);
            s._pad = 0;

            h_weights_cache[idx] = s.weight;
            n_written++;
        }

        // 剩余的突触 (如果 n_written < deg): 用柱内随机填充
        while (n_written < deg) {
            int post;
            if (is_pf) {
                post = N_ASSOCIATION_NEURONS_2E + randi(0, N_PREFRONTAL_NEURONS);
            } else {
                post = randi(0, N_ASSOCIATION_NEURONS_2E);
            }
            if (post == pre) post = (post + 1) % N_TOTAL_NEURONS_2E;

            int idx = start + n_written;
            h_col_idx[idx] = post;

            BioSynapse& s = h_synapses[idx];
            s.pre_idx = pre;
            s.post_idx = post;
            if (pre_is_exc) {
                s.weight = randf(0.1f, 0.4f);
            } else {
                s.weight = randf(-0.4f, -0.1f);
            }
            uint8_t delay = static_cast<uint8_t>(randi(DELAY_INTRA_MIN, DELAY_INTRA_MAX + 1));
            s.delay_steps = static_cast<float>(delay);
            h_delay[idx] = delay;

            s.last_pre_spike = -1000.0f;
            s.last_post_spike = -1000.0f;
            s.x_pre_trace = 0.0f;
            s.x_post_trace = 0.0f;
            s.nmda_conductance = 0.0f;
            s.ampa_conductance = 0.0f;
            s.ca_concentration = 0.0f;
            s.resource = 1.0f;
            s.utilization = pre_is_exc ? STP_U_SE : STP_U_SI;
            s.eligibility = 0.0f;
            s.eligibility_slow = 0.0f;
            s.scaling_factor = 1.0f;
            s.camkii_autophosph = 0.0f;
            s.da_receptor = pre_is_exc ? DA_RECEPTOR_INIT_EXC : DA_RECEPTOR_INIT_INH;
            s.ach_receptor = ACH_RECEPTOR_INIT;
            s.receptor_flags = pre_is_exc ? 0x03 : 0x0C;
            set_ne_receptor(s, NE_RECEPTOR_INIT);
            set_ht5_receptor(s, HT5_RECEPTOR_INIT);
            s._pad = 0;

            h_weights_cache[idx] = s.weight;
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
                  const NeuronStateAdEx* d_neurons) {
    (void)d_neurons;  // P1 不依赖 d_neurons 内容做初始化

    // Host 端构造
    std::vector<BioSynapse> h_syn;
    std::vector<int> h_row;
    std::vector<int> h_col;
    std::vector<float> h_w;
    std::vector<uint8_t> h_d;

    printf("[Stage2e P1] 生成突触拓扑 (host 端, ~10.7M 突触)...\n");
    init_synapses_host(h_syn, h_row, h_col, h_w, h_d);

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
    printf("[Stage2e P1] 突触拓扑已上传 GPU\n");
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
    CUDA_CHECK_2E(cudaMemset(b.d_delay_ring_indices, 0xFF, 20 * 500000 * sizeof(int)));  // -1 标记空槽
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
void init_network(MemoryAllocator* alloc) {
    printf("[Stage2e P1] === 网络初始化 ===\n");

    PersistentBuffers& b = alloc->buffers();

    printf("[Stage2e P1] 初始化神经元 (55K AdEx 静息)...\n");
    init_neurons(b.d_neurons);

    printf("[Stage2e P1] 初始化突触拓扑 + 延迟 + STP + 调质受体...\n");
    int n_syn = init_synapses(b.d_synapses, b.d_csr_row_ptr, b.d_csr_col_idx,
                              b.d_weights_cache, b.d_synapse_delay,
                              b.d_neurons);
    if (n_syn != N_TOTAL_SYNAPSES_2E) {
        fprintf(stderr, "[Stage2e P1 WARN] 突触数 %d != 目标 %d\n", n_syn, N_TOTAL_SYNAPSES_2E);
    }

    printf("[Stage2e P1] 初始化缓冲为零...\n");
    init_buffers_zero(alloc);

    printf("[Stage2e P1] 网络初始化完成\n");
}

} // namespace stage2e
