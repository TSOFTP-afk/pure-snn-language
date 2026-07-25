#ifndef SNN_STAGE2E_NETWORK_INIT_CUH
#define SNN_STAGE2E_NETWORK_INIT_CUH

// =============================================================================
// Stage 2e 网络初始化 (P1)
// =============================================================================
// 对应设计文档 §2.4, §2.1, §2.2:
//   - 神经元初始化: AdEx 状态全置静息, 80/20 兴奋/抑制, 3 种抑制亚型
//   - 突触拓扑: 50 柱 × 1000 神经元, 柱内 dense (p=0.1) + 跨柱 sparse (p=0.005)
//   - 突触初始化: 权重/电导/STP/调质受体密度
//   - 延迟分配: 柱内 1-3, 跨柱 5-10, 长程 15-20
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 初始化所有神经元状态 (AdEx 静息 + 类型/区域/柱分配)
// 必须在 scheduler.step() 之前调用
void init_neurons(NeuronStateAdEx* d_neurons);

// 初始化突触拓扑 + 突触状态
//   - 生成 CSR 连接 (柱内 dense + 跨柱 sparse + 前额叶自反馈)
//   - 填充 BioSynapse 字段 (weight, delay, conductance, STP, receptor)
//   - 初始化 PSW alpha/beta (Beta 分布先验参数)
//   - 返回实际生成的突触数 (应等于 N_TOTAL_SYNAPSES_2E)
int  init_synapses(BioSynapse* d_synapses,
                   int* d_csr_row_ptr,
                   int* d_csr_col_idx,
                   float* d_weights_cache,
                   uint8_t* d_synapse_delay,
                   float* d_synapse_alpha,    // PSW: LTP 证据累积
                   float* d_synapse_beta,     // PSW: LTD 证据累积
                   const NeuronStateAdEx* d_neurons,
                   uint32_t seed = 42);

// 初始化 GPU 缓冲为零 (调质浓度, 输入电流, nmda_current, etc.)
void init_buffers_zero(MemoryAllocator* alloc);

// -----------------------------------------------------------------------------
// 语言运动皮层初始化 (Stage 2e "语言运动皮层"基础设施)
// -----------------------------------------------------------------------------

// 初始化运动皮层 AdEx 神经元 (5K, 静息电位 -70mV, 与现有神经元初始化模式一致)
// 必须在 init_network() 中调用, d_motor_neurons 已分配
void init_motor_neurons(NeuronStateAdEx* d_motor_neurons);

// 生成 L5 → 运动皮层稀疏 CSR 突触
//   - 每个运动神经元 (索引 N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS + i)
//     接收来自对应柱 (i / MOTOR_GROUP_SIZE) L5 层的 L5_TO_MOTOR_SYNAPSES_PER_NEURON 个突触
//   - L5 层柱内偏移 = COL_L4_SIZE_2E + COL_L23_SIZE_2E = 550 (200 个 L5 神经元)
//   - 突触类型: 兴奋性 (AMPA + NMDA, receptor_flags = 0x03)
//   - 初始权重: [0.3, 0.8] 均匀分布
//   - 延迟: 1-3 步 (柱内延迟)
//   返回实际生成的突触数 (应等于 N_MOTOR_NEURONS × L5_TO_MOTOR_SYNAPSES_PER_NEURON = 250K)
int  init_l5_to_motor_synapses(BioSynapse* d_l5_to_motor_synapses,
                              float* d_l5_to_motor_weights,
                              int* d_l5_to_motor_csr_row_ptr,
                              int* d_l5_to_motor_csr_col_idx,
                              uint32_t seed = 42);

// P1 完整初始化入口 (调用上述三个 + 运动皮层初始化)
void init_network(MemoryAllocator* alloc, uint32_t seed = 42);

} // namespace stage2e

#endif // SNN_STAGE2E_NETWORK_INIT_CUH
