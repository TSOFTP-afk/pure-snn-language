"""P1 数据评估脚本
分析 stage2e_p1 的 --csv 输出, 评估:
1. spike 时间序列模式 (注入步 vs 非注入步)
2. 簇状发放判定是否合理
3. NMDA 是否真实激活
4. STDP trace 是否累积
5. 网络活动周期性
"""
import csv
import sys
from pathlib import Path
import statistics

CSV_PATH = Path(r"f:\项目\THE TRUE AI\src\stage2e\p1_profile_fixed.csv")

def load_csv():
    rows = []
    with open(CSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'step': int(r['step']),
                'spikes': int(r['spikes']),
                'is_inject': int(r['is_inject_step']),
                'byte': int(r['byte']),
                'nmda_sum': float(r['nmda_sum']),
                'nmda_nz': int(r['nmda_nz']),
                'xpre_sum': float(r['xpre_sum']),
                'xpre_nz': int(r['xpre_nz']),
                'ca_sum': float(r['ca_sum']),
                'ca_nz': int(r['ca_nz']),
                'wsum': float(r['weight_sum']),
                'wabs_sum': float(r['weight_abs_sum']),
            })
    return rows

def analyze(rows):
    n = len(rows)
    spikes = [r['spikes'] for r in rows]
    inject_rows = [r for r in rows if r['is_inject']]
    non_inject_rows = [r for r in rows if not r['is_inject']]

    print("=" * 78)
    print("  Stage 2e P1 数据评估报告")
    print("=" * 78)
    print(f"\n采样步数: {n}")
    print(f"注入步数: {len(inject_rows)} (每 {rows[1]['step']-rows[0]['step']} 步一次)")
    print(f"非注入步数: {len(non_inject_rows)}")

    # ============== 1. spike 分布 ==============
    print("\n" + "─" * 78)
    print("  [1] Spike 时间序列分布")
    print("─" * 78)
    print(f"  全局: min={min(spikes)}, max={max(spikes)}, "
          f"mean={statistics.mean(spikes):.1f}, "
          f"median={statistics.median(spikes):.1f}, "
          f"stdev={statistics.stdev(spikes):.1f}")

    inj_spikes = [r['spikes'] for r in inject_rows]
    non_spikes = [r['spikes'] for r in non_inject_rows]
    print(f"\n  注入步 ({len(inj_spikes)} 个):")
    if inj_spikes:
        print(f"    min={min(inj_spikes)}, max={max(inj_spikes)}, "
              f"mean={statistics.mean(inj_spikes):.1f}, "
              f"median={statistics.median(inj_spikes):.1f}")
    print(f"  非注入步 ({len(non_spikes)} 个):")
    if non_spikes:
        print(f"    min={min(non_spikes)}, max={max(non_spikes)}, "
              f"mean={statistics.mean(non_spikes):.1f}, "
              f"median={statistics.median(non_spikes):.1f}")

    # 注入步与非注入步的差异
    if inj_spikes and non_spikes:
        ratio = statistics.mean(inj_spikes) / max(statistics.mean(non_spikes), 1)
        print(f"\n  注入/非注入 比: {ratio:.1f}×")
        # 这是关键: 如果非注入步几乎为 0, 说明网络无自主活动
        silent_non = sum(1 for s in non_spikes if s == 0)
        print(f"  非注入步中静默步数 (spike=0): {silent_non}/{len(non_spikes)} "
              f"({100*silent_non/len(non_spikes):.1f}%)")

    # ============== 2. 簇状发放判定核查 ==============
    print("\n" + "─" * 78)
    print("  [2] 簇状发放判定合理性")
    print("─" * 78)
    avg = statistics.mean(spikes)
    burst_thresh = 2 * avg
    burst_steps = sum(1 for s in spikes if s > burst_thresh)
    print(f"  burst 阈值 = 2 * avg = {burst_thresh:.1f}")
    print(f"  burst 步数: {burst_steps}/{n} ({100*burst_steps/n:.1f}%)")
    # 检查 burst 步是否几乎全是注入步
    burst_inject = sum(1 for r in inject_rows if r['spikes'] > burst_thresh)
    if burst_steps > 0:
        print(f"  其中注入步: {burst_inject}/{burst_steps} "
              f"({100*burst_inject/burst_steps:.1f}%)")
        print(f"\n  >>> 警告: 簇状发放判定把'输入注入同步'误判为'簇状发放'")
        print(f"  >>> 真正簇状发放应跟踪单神经元连续高频, 而非群体同步")

    # ============== 3. NMDA 是否激活 ==============
    print("\n" + "─" * 78)
    print("  [3] NMDA 受体激活情况")
    print("─" * 78)
    nmda_sums = [r['nmda_sum'] for r in rows]
    nmda_nzs = [r['nmda_nz'] for r in rows]
    print(f"  nmda_current_sum: min={min(nmda_sums):.4f}, max={max(nmda_sums):.4f}, "
          f"final={nmda_sums[-1]:.4f}")
    print(f"  nmda_current_nz (非零神经元数): min={min(nmda_nzs)}, max={max(nmda_nzs)}, "
          f"final={nmda_nzs[-1]}/{55000}")
    if max(nmda_sums) < 1e-6:
        print("\n  >>> 严重问题: NMDA 电流始终为 0, NMDA 受体未激活")
        print(f"  >>> 原因: embryonic 期 nmda_expr=0, 但 synapse_nmda_kernel 未读取此参数")
        print(f"  >>> 实际原因: NMDA 电导需要 pre spike 触发, 但 sensory→assoc 流向上")
        print(f"      pre 脉冲多, post 电压低 (静息 0), Mg²⁺ 阻塞严重 (mg_factor≈0)")

    # ============== 4. STDP trace 累积 ==============
    print("\n" + "─" * 78)
    print("  [4] STDP x_pre trace 累积")
    print("─" * 78)
    xpre_sums = [r['xpre_sum'] for r in rows]
    xpre_nzs = [r['xpre_nz'] for r in rows]
    print(f"  xpre_sum: initial={xpre_sums[0]:.4f}, final={xpre_sums[-1]:.4f}, "
          f"max={max(xpre_sums):.4f}")
    print(f"  xpre_nz: initial={xpre_nzs[0]}, final={xpre_nzs[-1]}, "
          f"max={max(xpre_nzs)}/10700000")
    if xpre_sums[-1] > xpre_sums[0]:
        print(f"  >>> STDP x_pre trace 在累积 (Δ={xpre_sums[-1]-xpre_sums[0]:.4f}), "
              f"pre 脉冲被记录")
    else:
        print(f"  >>> STDP x_pre trace 未累积, pre 脉冲未触发")

    # ============== 5. 钙浓度 (NMDA 钙内流) ==============
    print("\n" + "─" * 78)
    print("  [5] 突触钙浓度 (NMDA 钙内流)")
    print("─" * 78)
    ca_sums = [r['ca_sum'] for r in rows]
    ca_nzs = [r['ca_nz'] for r in rows]
    print(f"  ca_sum: initial={ca_sums[0]:.4f}, final={ca_sums[-1]:.4f}, "
          f"max={max(ca_sums):.4f}")
    print(f"  ca_nz: initial={ca_nzs[0]}, final={ca_nzs[-1]}, max={max(ca_nzs)}/10700000")
    if max(ca_sums) < 1e-6:
        print(f"  >>> 钙浓度始终为 0: NMDA Mg²⁺ 阻塞未解除, LTP/LTD 开关失效")
    else:
        print(f"  >>> 钙浓度有累积, NMDA 部分激活")

    # ============== 6. 权重变化 (STDP 是否在学习) ==============
    print("\n" + "─" * 78)
    print("  [6] 权重变化 (STDP 学习验证)")
    print("─" * 78)
    print(f"  weight_sum (镜像缓存): initial={rows[0]['wsum']:.2f}, "
          f"final={rows[-1]['wsum']:.2f}")
    print(f"  weight_abs_sum: initial={rows[0]['wabs_sum']:.2f}, "
          f"final={rows[-1]['wabs_sum']:.2f}")
    print(f"  注: d_weights_cache 是初始化镜像, 不随 STDP 更新")
    print(f"  >>> 真实权重在 BioSynapse.weight, 需要单独采样才能验证")
    print(f"  >>> 当前 embryonic 期 plast_gain=0, STDP 不更新权重 (设计预期)")

    # ============== 7. 周期性分析 ==============
    print("\n" + "─" * 78)
    print("  [7] 周期性分析 (前 100 步 spike 序列)")
    print("─" * 78)
    print("  step | spikes | inject | byte")
    print("  -----+--------+--------+-----")
    for r in rows[:100]:
        marker = " *" if r['is_inject'] else "  "
        print(f"  {r['step']:4d} | {r['spikes']:6d} | {r['is_inject']:6d} | "
              f"{r['byte']:3d}{marker}")

    # ============== 8. 综合评估 ==============
    print("\n" + "=" * 78)
    print("  [综合评估]")
    print("=" * 78)
    issues = []
    if non_spikes and statistics.mean(non_spikes) < 5:
        issues.append("非注入步几乎无活动 (mean<5), 网络无自主动力学")
    if burst_steps > 0 and burst_inject == burst_steps:
        issues.append("100% burst 步是注入步, 簇状发放判定有偏")
    if max(nmda_sums) < 1e-6:
        issues.append("NMDA 电流始终为 0, NMDA 受体未激活 (Mg²⁺ 阻塞未解除)")
    if max(ca_sums) < 1e-6:
        issues.append("钙浓度为 0, LTP/LTD 开关失效")
    if xpre_sums[-1] <= xpre_sums[0]:
        issues.append("STDP x_pre trace 未累积")

    if not issues:
        print("  所有指标正常")
    else:
        print(f"  发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"    [{i}] {issue}")

    print("\n" + "=" * 78)
    return issues

if __name__ == '__main__':
    rows = load_csv()
    analyze(rows)
