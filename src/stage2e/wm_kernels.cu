// =============================================================================
// Stage 2e 工作记忆 (WM) 完整闭环 kernel 实现 (Task 9)
// =============================================================================
// 设计要点:
//   - wm_write_kernel: 单 block, 50 线程 (每线程一个槽位)
//       阶段1: 每线程计算 cosine 相似度 (50 维点积 + 范数)
//       阶段2: thread 0 串行归约找 max_sim + argmax (50 元素, 串行高效)
//       阶段3: thread 0 决策 (新颖写入 LRU / 刷新已有)
//   - wm_maintain_kernel: 单 block, 50 线程 (每线程一个槽位)
//       阶段1: 衰减 activation, age++
//       阶段2: 活跃槽位内联 PCA 反投影, 注入前额叶组 (100 神经元)
//
// PCA 反投影数学 (内联实现, 避免跨 kernel 调用):
//   recon[j] = mean_fr[base + j] + Σ_k pattern[k] · W[base + j][k]
//   其中 base = pf_start + group_offset
//         pf_start = n_neurons - n_prefrontal  (前额叶起始索引 = 50000)
//         group_offset = prefrontal_group × group_size
//   W 为 row-major [N × K], W[i][k] = d_pca_W[i * K + k]
//
// 前额叶布局:
//   全局神经元索引: [0, 50000) = 联合皮层, [50000, 55000) = 前额叶 (50 组 × 100)
//   d_prefrontal_input 索引 [0, 5000) 对应前额叶神经元 [50000, 55000)
//   组 g 占用 d_prefrontal_input[g*100 .. (g+1)*100)
// =============================================================================

#include "wm_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// Task 9.1: WM 写入 kernel (新颖检测 + LRU 替换)
// =============================================================================
//   cosine(sig, pat) = (sig · pat) / (|sig| · |pat|)
//   - sig 为当前 PCA 签名 (L2 归一化, |sig| ≈ 1, 但仍计算以防数值漂移)
//   - pat 为槽位存储的签名 (可能为零向量 → cosine = 0, 可被覆盖)
//
// 决策逻辑:
//   - max_sim < novelty_threshold (0.7): 新颖模式
//     → 写入 LRU 游标位置 (覆盖最旧/最弱槽位)
//     → 复制签名, age=0, activation=1.0, prefrontal_group = cursor % 50
//     → 游标前进 (atomicAdd + mod)
//   - max_sim >= novelty_threshold: 已有模式
//     → 刷新匹配槽位: activation=1.0, age=0
//
// 边界处理:
//   - signature 为零 (网络静默期): 不做任何写入
//   - 空槽位 (pattern 全零): cosine = 0, 可被新颖模式覆盖
// =============================================================================
__global__ void wm_write_kernel(
    WMSlot* __restrict__ d_wm_slots,
    const float* __restrict__ d_signature,
    int* __restrict__ d_wm_write_cursor,
    int current_step,
    int n_slots,
    float novelty_threshold)
{
    const int tid = threadIdx.x;

    // shared memory: 相似度数组 + 归约结果
    __shared__ float s_sim[WM_SLOTS];       // 每槽位 cosine 相似度
    __shared__ float s_sig_norm_sq;         // signature 范数平方
    __shared__ float s_max_sim;             // 最大相似度
    __shared__ int   s_max_idx;             // 最大相似度槽位索引

    // ---------- 阶段1a: 计算 signature 范数平方 (thread 0 负责, 广播) ----------
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (int k = 0; k < WM_PATTERN_DIM; ++k) {
            norm_sq += d_signature[k] * d_signature[k];
        }
        s_sig_norm_sq = norm_sq;
        s_max_sim = -2.0f;   // cosine ∈ [-1, 1], 初始化低于 -1
        s_max_idx = 0;
    }
    __syncthreads();
    const float sig_norm_sq = s_sig_norm_sq;

    // ---------- 阶段1b: 每线程计算一个槽位的 cosine 相似度 ----------
    float my_sim = -2.0f;
    if (tid < n_slots) {
        if (sig_norm_sq > 1e-20f) {
            float dot = 0.0f;
            float pat_norm_sq = 0.0f;
            const WMSlot& slot = d_wm_slots[tid];
            for (int k = 0; k < WM_PATTERN_DIM; ++k) {
                float p = slot.pattern[k];
                dot += d_signature[k] * p;
                pat_norm_sq += p * p;
            }
            if (pat_norm_sq > 1e-20f) {
                my_sim = dot / (sqrtf(sig_norm_sq) * sqrtf(pat_norm_sq));
            } else {
                // 空槽位 (pattern 全零): cosine = 0, 可被新颖模式覆盖
                my_sim = 0.0f;
            }
        }
        // signature 为零时 my_sim 保持 -2.0 (不触发任何分支)
        s_sim[tid] = my_sim;
    }
    __syncthreads();

    // ---------- 阶段2: 串行归约找最大相似度 + 索引 ----------
    // 50 个元素, thread 0 串行扫描比 tree reduction 更高效 (无同步开销)
    if (tid == 0) {
        float max_sim = -2.0f;
        int max_idx = 0;
        for (int i = 0; i < n_slots; ++i) {
            if (s_sim[i] > max_sim) {
                max_sim = s_sim[i];
                max_idx = i;
            }
        }
        s_max_sim = max_sim;
        s_max_idx = max_idx;
    }
    __syncthreads();

    // ---------- 阶段3: 决策 (thread 0 执行写入) ----------
    if (tid == 0) {
        // signature 为零时跳过 (网络静默期不写入 WM)
        if (sig_norm_sq <= 1e-20f) {
            (void)current_step;  // 预留: 未来可用于步数相关写入策略
            return;
        }

        if (s_max_sim < novelty_threshold) {
            // 新颖模式: 写入 LRU 游标位置 (覆盖最旧/最弱槽位)
            // atomicAdd 返回旧值, 然后游标 +1; mod n_slots 实现环形循环
            int cursor = atomicAdd(d_wm_write_cursor, 1) % n_slots;
            WMSlot& slot = d_wm_slots[cursor];
            // 复制签名到 pattern
            for (int k = 0; k < WM_PATTERN_DIM; ++k) {
                slot.pattern[k] = d_signature[k];
            }
            slot.age = 0;
            slot.activation = 1.0f;
            slot.prefrontal_group = cursor % PREFRONTAL_GROUPS;
        } else {
            // 已有模式: 刷新匹配槽位 (重置 activation 和 age)
            WMSlot& slot = d_wm_slots[s_max_idx];
            slot.activation = 1.0f;
            slot.age = 0;
        }
        (void)current_step;  // 预留: 未来可用于步数相关写入策略
    }
}

// =============================================================================
// Task 9.2: WM 维持与注入 kernel (衰减 + PCA 反投影注入)
// =============================================================================
//   每线程处理一个 WM 槽位:
//     1. activation *= decay_factor (指数衰减, 模拟短期记忆遗忘)
//     2. age++
//     3. 若 activation > inject_threshold:
//        - 预加载 slot.pattern 到寄存器 (避免内层循环重复读 global memory)
//        - 内联 PCA 反投影重建前额叶组发放率 (100 神经元 × 50 分量)
//        - 注入电流到 d_prefrontal_input (atomicAdd, 防多槽位冲突)
//
// 注入电流公式:
//   current[j] = recon[j] × activation × WM_INJECT_GAIN
//   recon[j] = mean_fr[base + j] + Σ_k pattern[k] · W[base + j][k]
//   base = pf_start + group_offset
//   pf_start = n_neurons - n_prefrontal  (= 55000 - 5000 = 50000)
//   group_offset = prefrontal_group × group_size
//
// 注: 调用方需在调用前清零 d_prefrontal_input (本 kernel 仅做累加注入)
// =============================================================================
__global__ void wm_maintain_kernel(
    WMSlot* __restrict__ d_wm_slots,
    const float* __restrict__ d_pca_W,
    const float* __restrict__ d_mean_fr,
    float* __restrict__ d_prefrontal_input,
    int n_slots,
    int n_prefrontal,
    int group_size,
    float inject_threshold,
    float decay_factor,
    int n_pca_components,
    int n_neurons)
{
    const int slot_idx = threadIdx.x;
    if (slot_idx >= n_slots) return;

    WMSlot& slot = d_wm_slots[slot_idx];

    // ---------- 阶段1: 衰减 + 老化 ----------
    slot.activation *= decay_factor;
    slot.age += 1;

    // ---------- 阶段2: 活跃槽位注入前额叶 ----------
    if (slot.activation > inject_threshold) {
        // 前额叶在全局神经元数组中的起始索引
        // n_neurons = 联合皮层(50000) + 前额叶(5000) = 55000
        // pf_start = 55000 - 5000 = 50000
        const int pf_start = n_neurons - n_prefrontal;
        const int group_offset = slot.prefrontal_group * group_size;

        // 预加载 pattern 到寄存器 (50 float = 200B, 编译器可保持在寄存器中)
        // 避免内层 j 循环重复从 global memory 读取 slot.pattern[k]
        float pat[WM_PATTERN_DIM];
        for (int k = 0; k < WM_PATTERN_DIM; ++k) {
            pat[k] = slot.pattern[k];
        }

        // 内联 PCA 反投影 + 注入 (每组 100 神经元)
        for (int j = 0; j < group_size; ++j) {
            const int neuron_idx = pf_start + group_offset + j;
            if (neuron_idx >= n_neurons) break;  // 防越界

            // PCA 反投影: recon = mean_fr + Σ_k pattern[k] · W[neuron_idx][k]
            float recon = d_mean_fr[neuron_idx];
            const size_t w_base = (size_t)neuron_idx * n_pca_components;
            #pragma unroll 8
            for (int k = 0; k < n_pca_components; ++k) {
                recon += pat[k] * d_pca_W[w_base + k];
            }

            // 注入电流 = recon × activation × gain
            float current = recon * slot.activation * WM_INJECT_GAIN;
            // atomicAdd: 虽然各槽位 prefrontal_group 唯一 (cursor % 50),
            // 用原子操作保证未来扩展 (如多槽位共享组) 的安全性
            atomicAdd(&d_prefrontal_input[group_offset + j], current);
        }
    }
}

// =============================================================================
// Host 端 wrapper 函数
// =============================================================================
// 接受裸指针 (不依赖 MemoryAllocator), 由 scheduler 在集成时调用:
//   - n = WM_SLOTS = 50
//   - n_pf = N_PREFRONTAL_NEURONS = 5000
//   - group_sz = NEURONS_PER_PF_GROUP = 100
//   - n_comp = PCA_N_COMPONENTS = 50
//   - n_neurons = N_ASSOCIATION_NEURONS_2E + N_PREFRONTAL_NEURONS = 55000
// =============================================================================

void launch_wm_write(WMSlot* d_slots, const float* d_sig,
                     int* d_cursor, int step, int n, float thr)
{
    // 单 block, n 线程 (每线程一个 WM 槽位)
    // n = WM_SLOTS = 50, 单 block 足够 (shared memory 串行归约)
    wm_write_kernel<<<1, n>>>(d_slots, d_sig, d_cursor, step, n, thr);
    CUDA_CHECK_LAST_2E();
}

void launch_wm_maintain(WMSlot* d_slots, const float* d_pca_W,
                        const float* d_mean_fr, float* d_pf_input,
                        int n_slots, int n_pf, int group_sz,
                        float inject_thr, float decay, int n_comp, int n_neurons)
{
    // 单 block, n_slots 线程 (每线程一个 WM 槽位)
    // n_slots = 50, 每活跃槽位内联反投影 100 神经元 × 50 分量 = 5000 乘加
    wm_maintain_kernel<<<1, n_slots>>>(d_slots, d_pca_W, d_mean_fr, d_pf_input,
                                        n_slots, n_pf, group_sz, inject_thr, decay,
                                        n_comp, n_neurons);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
