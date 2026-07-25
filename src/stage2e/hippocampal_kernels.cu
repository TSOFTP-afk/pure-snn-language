// =============================================================================
// Stage 2e 海马索引编码 kernel 实现 (Task 3)
// =============================================================================
// 设计要点:
//   1. hippo_encode_kernel: 单 block (256 线程) 协作
//        阶段1: signature[50] 加载到 shared memory, 256 线程分块遍历 50K 索引,
//               每线程维护局部最大相似度 (cosine_sim = dot product, PCA 已归一化),
//               tree reduction 得全局最佳匹配 (sim, idx)
//        阶段2: tid==0 单线程判定新颖性并写入 (避免竞争):
//               - sim >= novelty_threshold → 已有模式, importance += 1/(1+replay_count)
//               - sim <  novelty_threshold → 新颖模式, 写入 d_write_cursor 槽位,
//                 推进环形游标, 递增 d_filled_count (上限 max_indices)
//   2. hippo_get_top_k_kernel: 单 block, K 轮 partial selection sort
//        - shared memory 维护 selected bitmap (HIPP_INDEX_SIZE/8 = 6250B)
//        - 每轮并行 argmax 跳过已选, tid==0 写入 d_top_k_indices[iter] 并置位
//        - filled_count < k 时多余位置写 -1
//   3. hippo_decay_importance_kernel: grid 跨步, 每线程一个被重放索引
//        - importance *= HIPP_REPLAY_DECAY (0.9), replay_count++
//        - 跳过 -1 无效索引
// =============================================================================

#include "hippocampal_kernels.cuh"
#include "pca_kernels.cuh"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace stage2e {

// selected bitmap 字节数 (HIPP_INDEX_SIZE bit → 6250 byte, 编译期常量)
// 注: HIPP_INDEX_SIZE 是 #define = 50000, 直接用于 static shared memory 大小
constexpr int kHippoBitmapBytes = (HIPP_INDEX_SIZE + 7) / 8;

// =============================================================================
// Task 3.1: 海马编码 kernel
// =============================================================================
__global__ void hippo_encode_kernel(
    HippoIndex* __restrict__ d_indices,
    const float* __restrict__ d_signature,
    int* __restrict__ d_write_cursor,
    int* __restrict__ d_filled_count,
    int current_step,
    int max_indices,
    float novelty_threshold)
{
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;     // = THREADS_PER_BLOCK_2E = 256

    // shared memory 布局:
    //   s_sig[50]        当前 PCA 签名 (全 block 共享, 避免每线程重复读 global)
    //   s_sim[256]       每线程局部最佳相似度 (tree reduction scratch)
    //   s_idx[256]       每线程局部最佳索引
    //   s_best_sim/s_best_idx/s_filled  归约结果 + filled_count 镜像
    __shared__ float s_sig[PATTERN_DIM];
    __shared__ float s_sim[THREADS_PER_BLOCK_2E];
    __shared__ int   s_idx[THREADS_PER_BLOCK_2E];
    __shared__ float s_best_sim;
    __shared__ int   s_best_idx;
    __shared__ int   s_filled;

    // ---------- 协同加载 signature[50] 到 shared memory ----------
    for (int k = tid; k < PATTERN_DIM; k += bs) {
        s_sig[k] = d_signature[k];
    }

    // ---------- 读取 filled_count (单线程, 广播) ----------
    if (tid == 0) {
        int f = d_filled_count[0];
        if (f < 0) f = 0;
        if (f > max_indices) f = max_indices;
        s_filled = f;
        s_best_sim = -2.0f;     // cosine_sim ∈ [-1, 1], 初值低于下界
        s_best_idx = -1;
    }
    __syncthreads();

    const int filled = s_filled;

    // ---------- 阶段1: 分块遍历 [0, filled), 每线程找局部最大相似度 ----------
    float my_best_sim = -2.0f;
    int   my_best_idx = -1;
    for (int i = tid; i < filled; i += bs) {
        const float* entry_sig = d_indices[i].pattern_signature;
        float dot = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < PATTERN_DIM; ++k) {
            dot += s_sig[k] * entry_sig[k];
        }
        // cosine_similarity = dot (PCA 签名已 L2 归一化)
        if (dot > my_best_sim) {
            my_best_sim = dot;
            my_best_idx = i;
        }
    }
    s_sim[tid] = my_best_sim;
    s_idx[tid] = my_best_idx;
    __syncthreads();

    // tree reduction 找全局最大相似度 (blockDim.x = 256 = 2^8, 满足 2 的幂要求)
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_sim[tid + stride] > s_sim[tid]) {
                s_sim[tid] = s_sim[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_best_sim = s_sim[0];
        s_best_idx = s_idx[0];
    }
    __syncthreads();

    // ---------- 阶段2: 新颖性判定 + 写入 (tid==0 单线程, 避免竞争) ----------
    if (tid == 0) {
        const float best_sim = s_best_sim;
        const int   best_idx = s_best_idx;

        // 语义: sim >= novelty_threshold → 已有模式 (相似); 否则 → 新颖
        // best_idx == -1 表示索引表为空 (filled == 0), 视为新颖
        if (best_idx >= 0 && best_sim >= novelty_threshold) {
            // ---- 已有模式: 刷新 importance ----
            // importance += 1.0 / (1.0 + replay_count)
            // 生物学: 反复出现的模式 importance 累积, 但增量随 replay_count 衰减
            HippoIndex& entry = d_indices[best_idx];
            float inc = 1.0f / (1.0f + static_cast<float>(entry.replay_count));
            entry.importance += inc;
        } else {
            // ---- 新颖模式: 写入 LRU 槽位 ----
            int slot = d_write_cursor[0];
            if (slot < 0 || slot >= max_indices) slot = 0;   // 防御性 clamp

            HippoIndex& entry = d_indices[slot];
            // 复制 signature 到 pattern_signature (从 shared memory 读, 合并写)
            for (int k = 0; k < PATTERN_DIM; ++k) {
                entry.pattern_signature[k] = s_sig[k];
            }
            entry.pattern_start_step = current_step;
            entry.replay_count       = 0;
            entry.importance         = 1.0f;

            // 推进 LRU 游标 (环形 % max_indices)
            // 单 block 单线程写, 无需 atomic; 保留语义以备未来多 block 扩展
            int next = slot + 1;
            if (next >= max_indices) next = 0;
            d_write_cursor[0] = next;

            // 若 filled_count < max_indices, 递增 (LRU 未填满时追加)
            int cur_filled = d_filled_count[0];
            if (cur_filled < max_indices) {
                d_filled_count[0] = cur_filled + 1;
            }
        }
    }
    // 无需 __syncthreads: kernel 末尾 CUDA 隐式同步
}

// =============================================================================
// Task 3.2: top-K 索引选取 kernel (partial selection sort + selected bitmap)
// =============================================================================
__global__ void hippo_get_top_k_kernel(
    const HippoIndex* __restrict__ d_indices,
    int* __restrict__ d_top_k_indices,
    int* __restrict__ d_filled_count,
    int k,
    int max_indices)
{
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;     // = 256

    // shared memory:
    //   s_selected[kHippoBitmapBytes]  已选索引 bitmap (1 bit/索引, 6250B)
    //   s_sim[256] / s_idx[256]        tree reduction scratch
    //   s_best_sim / s_best_idx        每轮全局 argmax
    //   s_filled                       filled_count 镜像
    __shared__ unsigned char s_selected[kHippoBitmapBytes];
    __shared__ float s_sim[THREADS_PER_BLOCK_2E];
    __shared__ int   s_idx[THREADS_PER_BLOCK_2E];
    __shared__ float s_best_sim;
    __shared__ int   s_best_idx;
    __shared__ int   s_filled;

    // ---------- 初始化 bitmap 为 0 ----------
    for (int i = tid; i < kHippoBitmapBytes; i += bs) {
        s_selected[i] = 0;
    }

    if (tid == 0) {
        int f = d_filled_count[0];
        if (f < 0) f = 0;
        if (f > max_indices) f = max_indices;
        s_filled = f;
    }
    __syncthreads();

    const int filled = s_filled;
    // 实际可选数 = min(k, filled), 多余位置写 -1
    const int actual_k = (k < filled) ? k : filled;

    // ---------- K 轮 partial selection sort ----------
    for (int iter = 0; iter < actual_k; ++iter) {
        // 每线程在自己负责的索引范围内找最大 importance (跳过已选)
        float my_best = -2.0f;
        int   my_idx  = -1;
        for (int i = tid; i < filled; i += bs) {
            // 检查 selected bitmap
            int byte = i >> 3;
            int bit  = i & 7;
            if (s_selected[byte] & (1u << bit)) continue;
            float imp = d_indices[i].importance;
            if (imp > my_best) {
                my_best = imp;
                my_idx  = i;
            }
        }
        s_sim[tid] = my_best;
        s_idx[tid] = my_idx;
        __syncthreads();

        // tree reduction 找全局 max
        for (int stride = bs / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_sim[tid + stride] > s_sim[tid]) {
                    s_sim[tid] = s_sim[tid + stride];
                    s_idx[tid] = s_idx[tid + stride];
                }
            }
            __syncthreads();
        }

        // tid==0 写入结果并置位 bitmap
        if (tid == 0) {
            int winner = s_idx[0];
            d_top_k_indices[iter] = winner;
            if (winner >= 0) {
                int byte = winner >> 3;
                int bit  = winner & 7;
                s_selected[byte] |= (1u << bit);
            }
        }
        __syncthreads();   // 确保 bitmap 写入对下一轮可见
    }

    // ---------- filled < k 时, 多余位置写 -1 (调用方需检查) ----------
    for (int i = tid + actual_k; i < k; i += bs) {
        d_top_k_indices[i] = -1;
    }
}

// =============================================================================
// Task 3.3: 重放后衰减 kernel (grid 跨步)
// =============================================================================
__global__ void hippo_decay_importance_kernel(
    HippoIndex* __restrict__ d_indices,
    const int* __restrict__ d_replayed_indices,
    int k)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;

    int idx = d_replayed_indices[tid];
    if (idx < 0) return;     // 跳过无效索引 (top-K 不足 K 时填充 -1)

    HippoIndex& entry = d_indices[idx];
    entry.importance *= HIPP_REPLAY_DECAY;   // 0.9
    entry.replay_count += 1;
}

// =============================================================================
// P1.2 修复: 海马 importance 时间衰减 kernel (grid 跨步遍历全部索引)
// =============================================================================
// 生物学意义: 海马记忆痕迹自然衰退 (memory trace decay), 与重放驱动的快速衰减互补
// 调用时机: 每 HIPP_ENCODE_INTERVAL 步 (与编码同步), 在 launch_hippo_encode 之后执行
// 效果: 所有已填充索引 importance *= HIPP_TIME_DECAY (0.9995)
//       新模式写入时重置为 1.0, 形成新旧对比, LRU 选择低 importance 的槽位淘汰
__global__ void hippo_time_decay_kernel(
    HippoIndex* __restrict__ d_indices,
    int filled_count,
    int max_indices)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    int n = filled_count;
    if (n > max_indices) n = max_indices;

    for (int i = tid; i < n; i += stride) {
        // 仅衰减已填充的槽位 (filled_count 范围内)
        // 跳过空槽位 (candidate_pre==0 且 replay_count==0 视为空)
        if (d_indices[i].replay_count == 0 && d_indices[i].importance == 0.0f) continue;
        d_indices[i].importance *= HIPP_TIME_DECAY;
    }
}

// =============================================================================
// Host 端 wrapper 函数
// =============================================================================
// 接受裸指针 (不依赖 MemoryAllocator), 由 scheduler 在集成时调用:
//   - max_idx = HIPP_INDEX_SIZE (50000)
//   - novelty_thr = HIPP_NOVELTY_THRESHOLD (0.7)
//   - k = HIPP_REPLAY_BATCH (200)
// =============================================================================

void launch_hippo_encode(HippoIndex* d_indices, const float* d_sig,
                         int* d_write_cursor, int* d_filled_count,
                         int step, int max_idx, float novelty_thr)
{
    // 单 block, 256 线程协作 (50K 量级, 单 block 可接受)
    hippo_encode_kernel<<<1, THREADS_PER_BLOCK_2E>>>(
        d_indices, d_sig, d_write_cursor, d_filled_count,
        step, max_idx, novelty_thr);
    CUDA_CHECK_LAST_2E();
}

void launch_hippo_get_top_k(const HippoIndex* d_indices, int* d_top_k,
                            int* d_filled_count, int k, int max_idx)
{
    // 单 block, 256 线程 + selected bitmap (K=200 轮 argmax)
    (void)max_idx;   // max_indices 由 d_filled_count 上限保证, kernel 内只读 filled_count
    hippo_get_top_k_kernel<<<1, THREADS_PER_BLOCK_2E>>>(
        d_indices, d_top_k, d_filled_count, k, max_idx);
    CUDA_CHECK_LAST_2E();
}

void launch_hippo_decay(HippoIndex* d_indices, const int* d_replayed, int k)
{
    // grid 跨步, 每线程一个被重放索引
    int blocks = (k + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    hippo_decay_importance_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_indices, d_replayed, k);
    CUDA_CHECK_LAST_2E();
}

// P1.2 修复: 时间衰减 host wrapper
// 调用时机: 每 HIPP_ENCODE_INTERVAL 步, 在 launch_hippo_encode 之后执行
void launch_hippo_time_decay(HippoIndex* d_indices, int filled_count, int max_indices)
{
    if (filled_count <= 0) return;
    int n = filled_count;
    if (n > max_indices) n = max_indices;
    int blocks = (n + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    if (blocks > 32) blocks = 32;   // 50K 量级, 32 blocks 足够
    hippo_time_decay_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_indices, n, max_indices);
    CUDA_CHECK_LAST_2E();
}

// =============================================================================
// Task 4-5: 睡眠重放电流注入 kernel (元素级)
// =============================================================================
// 睡眠重放 STDP 巩固 kernel: 对被重放的模式执行 STDP 巩固
// 对每个被重放的 top-K 索引 (来自 d_replayed_indices):
//   1. PCA 反投影已由 host 端 pca_back_project 完成, 结果在 d_replay_injection (N_ASSOCIATION_NEURONS_2E)
//   2. 注入 d_replay_injection 到联合皮层 (作为外部电流, 由调用方处理)
//   3. STDP 巩固: 对突触后神经元发放的突触, Δw = η_replay · pre · post · tag
//      简化版: 对每个突触后神经元 i, 若 d_replay_injection[i] > threshold,
//      则对其所有入突触执行 Δw += η_replay · tag (弱化的 STDP, 不依赖 pre 发放)
//
// 注意: 完整的突触级 STDP 巩固需要遍历 CSR 突触, 复杂度高。
// 此处采用简化方案: 重放注入后, 由现有 STDP kernel 在后续步骤自然学习。
// replay_kernel 仅负责注入重放电流到 d_replay_injection, 真正的 STDP 由主循环处理。
//
// 因此本 kernel 实际上是一个"重放电流注入" kernel:
//   对每个联合皮层神经元 i, d_replay_injection[i] = reconstructed[i] * REPLAY_INJECT_GAIN
//   其中 reconstructed 已由 host 端 pca_back_project 计算并写入 d_replay_injection
__global__ void replay_inject_kernel(
    float* __restrict__ d_replay_injection,      // [n_neurons] 重放注入电流 (in/out)
    const float* __restrict__ d_reconstructed,   // [n_neurons] PCA 反投影重建的发放率
    int n_neurons,                                // N_ASSOCIATION_NEURONS_2E = 50000
    float inject_gain)                           // REPLAY_INJECT_GAIN = 2.0f (10x 速度的简化实现)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_neurons) {
        d_replay_injection[i] = d_reconstructed[i] * inject_gain;
    }
}

// =============================================================================
// Task 4-5: 睡眠重放完整流程 (自由函数, 由 scheduler.launch_replay 调用)
// =============================================================================
// 睡眠重放: 完整流程
//   1. launch_hippo_get_top_k: 取 importance top-K 索引
//   2. 对每个 top-K 模式:
//      a. 提取 signature 到临时缓冲
//      b. pca_back_project 重建发放向量
//      c. replay_inject_kernel 注入到 d_replay_injection
//   3. launch_hippo_decay: 重放后衰减 importance, replay_count++
//
// 参数说明:
//   d_indices:          [HIPP_INDEX_SIZE] 海马索引表
//   d_top_k_indices:    [batch_size] top-K 索引输出
//   d_filled_count:     [1] 已填充条目数
//   d_replay_injection: [N_ASSOCIATION_NEURONS_2E] 重放注入电流 (已分配)
//   d_sig_buffer:       [PCA_N_COMPONENTS] 临时签名缓冲 (调用方分配)
//   d_pca_W:            [N × K] PCA 基矩阵
//   d_pca_mean:         [N] 滑动平均发放率
//   d_reconstructed_buffer: [N_ASSOCIATION_NEURONS_2E] PCA 反投影重建临时缓冲
//   step:               当前训练步 (未直接使用, 预留扩展)
//   max_indices:        HIPP_INDEX_SIZE
//   batch_size:         HIPP_REPLAY_BATCH
void launch_replay_cycle(
    HippoIndex* d_indices,
    int* d_top_k_indices,
    int* d_filled_count,
    float* d_replay_injection,
    float* d_sig_buffer,
    const float* d_pca_W,
    const float* d_pca_mean,
    float* d_reconstructed_buffer,
    int step, int max_indices, int batch_size)
{
    (void)step;   // 预留扩展, 当前未使用

    // 1. 取 importance top-K 索引
    launch_hippo_get_top_k(d_indices, d_top_k_indices, d_filled_count,
                           batch_size, max_indices);

    // 2. 拷贝 top-K 索引到 host (batch_size × int = 800B for batch=200)
    std::vector<int> h_top_k(batch_size, -1);
    CUDA_CHECK_2E(cudaMemcpy(h_top_k.data(), d_top_k_indices,
                              batch_size * sizeof(int),
                              cudaMemcpyDeviceToHost));

    // 3. 对每个有效 top-K 索引: 提取签名 → PCA 反投影 → 重放注入
    const int N = N_ASSOCIATION_NEURONS_2E;
    const int K = PCA_N_COMPONENTS;

    int blocks = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    for (int i = 0; i < batch_size; ++i) {
        int idx = h_top_k[i];
        if (idx < 0) continue;     // top-K 不足时填充 -1, 跳过

        // 3a. 从 d_indices[idx].pattern_signature 拷贝 50 float 到 d_sig_buffer
        //     HippoIndex 结构体首字段为 pattern_signature[50], 偏移 0
        CUDA_CHECK_2E(cudaMemcpy(d_sig_buffer,
                                  d_indices[idx].pattern_signature,
                                  K * sizeof(float),
                                  cudaMemcpyDeviceToDevice));

        // 3b. PCA 反投影: sig → reconstructed (N 维联合皮层发放率)
        launch_pca_back_project(d_pca_W, d_pca_mean, d_sig_buffer,
                                d_reconstructed_buffer, N, K);

        // 3c. 重放电流注入: d_replay_injection = reconstructed × REPLAY_INJECT_GAIN
        //     元素级赋值 (覆盖前一次): 最终 d_replay_injection 保留最后一个模式的重建
        //     调用方在循环外清零 d_replay_injection, 避免上一周期残留
        replay_inject_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
            d_replay_injection, d_reconstructed_buffer, N, REPLAY_INJECT_GAIN);
        CUDA_CHECK_LAST_2E();
    }

    // 4. 重放后衰减: importance *= HIPP_REPLAY_DECAY, replay_count++
    launch_hippo_decay(d_indices, d_top_k_indices, batch_size);
}

} // namespace stage2e
