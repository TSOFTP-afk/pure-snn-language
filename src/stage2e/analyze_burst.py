"""P1 修复后 burst 分布详细分析"""
import csv
from pathlib import Path
import statistics

CSV_PATH = Path(r"f:\项目\THE TRUE AI\src\stage2e\p1_profile_fixed.csv")

rows = []
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'step': int(r['step']),
            'spikes': int(r['spikes']),
            'is_inject': int(r['is_inject_step']),
        })

non_inject = [r['spikes'] for r in rows if not r['is_inject']]
print(f"非注入步数: {len(non_inject)}")
print(f"非注入步 spikes: min={min(non_inject)}, max={max(non_inject)}, "
      f"mean={statistics.mean(non_inject):.1f}, median={statistics.median(non_inject):.1f}")
print(f"非注入步 stdev: {statistics.stdev(non_inject):.1f}")

# 分布直方图
buckets = [0, 1, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
print("\n非注入步 spike 分布直方图:")
print(f"{'range':<15} {'count':<8} {'pct':<8}")
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    c = sum(1 for s in non_inject if lo <= s < hi)
    pct = 100 * c / len(non_inject)
    print(f"[{lo:>4}, {hi:>4})    {c:<8} {pct:.1f}%")

# 当前 burst 判定
avg = statistics.mean(non_inject)
thresh = 2 * avg
bursts = sum(1 for s in non_inject if s > thresh)
print(f"\n当前 burst 阈值 = 2 * avg = {thresh:.1f}")
print(f"超过阈值的非注入步: {bursts}/{len(non_inject)} ({100*bursts/len(non_inject):.2f}%)")

# 不同阈值方案对比
print("\n不同 burst 判定方案的 burst%:")
for factor in [1.5, 1.8, 2.0, 2.5, 3.0]:
    t = factor * avg
    b = sum(1 for s in non_inject if s > t)
    print(f"  {factor}×avg ({t:.0f}): {b}/{len(non_inject)} = {100*b/len(non_inject):.2f}%")

# 用 stdev 而非 avg 的方案
print("\n用 stdev 的方案:")
for k in [1.0, 1.5, 2.0, 2.5]:
    t = avg + k * statistics.stdev(non_inject)
    b = sum(1 for s in non_inject if s > t)
    print(f"  avg + {k}σ ({t:.0f}): {b}/{len(non_inject)} = {100*b/len(non_inject):.2f}%")
