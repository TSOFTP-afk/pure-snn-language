#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2e Perplexity 评估脚本 (Task 11)
=====================================
从训练日志解析 perplexity 曲线, 绘制图表并输出统计信息。

支持的日志格式 (自动识别):
  1. [Eval] step=NNNN perplexity=X.XX decode_acc=X.XX%
     (任务规格描述的格式)
  2. [Stage2e Decode] step=NNNN  avg_loss=X.XX  perplexity=X.XX  accuracy=X.XX% (n/m)  last_pred=N
     (scheduler.cu 实际输出的格式)

用法:
    python eval_perplexity.py [training_log_path]

默认日志路径: training_freqnorm_100k.log
输出:
    - perplexity_curve.png (双子图: perplexity + decode_acc)
    - 统计信息打印到 stdout
"""

import sys
import re
import math
from pathlib import Path

# 使用 Agg 后端, 避免中文路径 / 无显示环境下的渲染问题
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# 常量
# -----------------------------------------------------------------------------
RANDOM_BASELINE_PERPLEXITY = math.log(256)   # ≈ 5.5452 (均匀分布 over 256 bytes)
TARGET_PERPLEXITY = 4.0                       # 设计目标
RANDOM_BASELINE_DECODE_ACC = 100.0 / 256.0    # ≈ 0.3906% (随机猜测)
DEFAULT_LOG_PATH = "training_freqnorm_100k.log"
OUTPUT_PNG = "perplexity_curve.png"


# -----------------------------------------------------------------------------
# 日志解析
# -----------------------------------------------------------------------------
# 正则 1: 任务规格格式 [Eval] step=NNNN perplexity=X.XX decode_acc=X.XX%
RE_EVAL = re.compile(
    r"\[Eval\]\s+step=(\d+)\s+perplexity=([\d.]+)\s+decode_acc=([\d.]+)%"
)

# 正则 2: scheduler.cu 实际格式 [Stage2e Decode] step=N  avg_loss=X  perplexity=X  accuracy=X% (n/m)  last_pred=N
RE_DECODE = re.compile(
    r"\[Stage2e Decode\]\s+step=(\d+)\s+avg_loss=([\d.]+)\s+perplexity=([\d.]+)\s+accuracy=([\d.]+)%"
)


def parse_log(log_path):
    """
    解析训练日志, 提取 (step, perplexity, decode_acc) 三元组列表。

    自动识别两种日志格式:
      - [Eval] step=... perplexity=... decode_acc=...%
      - [Stage2e Decode] step=... perplexity=... accuracy=...%

    返回:
        list of (step:int, perplexity:float, decode_acc:float) 三元组
        若解析失败返回空列表
    """
    results = []
    path = Path(log_path)
    if not path.exists():
        print(f"[ERROR] 日志文件不存在: {log_path}", file=sys.stderr)
        return results

    # UTF-8 编码读取 (训练日志含中文注释)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 尝试格式 1: [Eval] step=... perplexity=... decode_acc=...%
            m = RE_EVAL.search(line)
            if m:
                step = int(m.group(1))
                ppl = float(m.group(2))
                acc = float(m.group(3))
                results.append((step, ppl, acc))
                continue

            # 尝试格式 2: [Stage2e Decode] step=... perplexity=... accuracy=...%
            m = RE_DECODE.search(line)
            if m:
                step = int(m.group(1))
                ppl = float(m.group(3))
                acc = float(m.group(4))
                results.append((step, ppl, acc))
                continue

    return results


# -----------------------------------------------------------------------------
# 统计输出
# -----------------------------------------------------------------------------
def print_statistics(data):
    """
    输出 perplexity / decode_acc 统计信息到 stdout。

    data: list of (step, perplexity, decode_acc)
    """
    if not data:
        print("[WARN] 无可用数据, 跳过统计输出")
        return

    steps = [d[0] for d in data]
    ppls = [d[1] for d in data]
    accs = [d[2] for d in data]

    final_step = steps[-1]
    final_ppl = ppls[-1]
    final_acc = accs[-1]

    # 最佳 perplexity (最低)
    best_idx = min(range(len(ppls)), key=lambda i: ppls[i])
    best_ppl = ppls[best_idx]
    best_step = steps[best_idx]

    # 相比随机基线的改善百分比
    # 改善 = (baseline - final) / baseline * 100%  (正值表示优于基线)
    if RANDOM_BASELINE_PERPLEXITY > 0:
        improvement_pct = (RANDOM_BASELINE_PERPLEXITY - final_ppl) / RANDOM_BASELINE_PERPLEXITY * 100.0
    else:
        improvement_pct = 0.0

    print("=" * 60)
    print("  Stage 2e Perplexity 评估统计")
    print("=" * 60)
    print(f"  日志数据点数:          {len(data)}")
    print(f"  评估步数范围:          {steps[0]} - {steps[-1]}")
    print()
    print(f"  随机基线 perplexity:   {RANDOM_BASELINE_PERPLEXITY:.4f}  (ln(256))")
    print(f"  目标 perplexity:       {TARGET_PERPLEXITY:.4f}")
    print()
    print(f"  最终 perplexity:       {final_ppl:.4f}  (step={final_step})")
    print(f"  相比基线改善:          {improvement_pct:+.2f}%")
    print(f"  最佳 perplexity:       {best_ppl:.4f}  (step={best_step})")
    print()
    print(f"  随机基线 decode_acc:   {RANDOM_BASELINE_DECODE_ACC:.4f}%  (1/256)")
    print(f"  最终 decode_acc:       {final_acc:.4f}%")
    print(f"  最佳 decode_acc:       {max(accs):.4f}%")
    print()

    # 达标判定
    ppl_pass = final_ppl < RANDOM_BASELINE_PERPLEXITY
    ppl_target = final_ppl < TARGET_PERPLEXITY
    print(f"  优于随机基线:          {'YES' if ppl_pass else 'NO'}")
    print(f"  达到目标 (< {TARGET_PERPLEXITY}):    {'YES' if ppl_target else 'NO'}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 绘图
# -----------------------------------------------------------------------------
def plot_curves(data, output_path):
    """
    绘制双子图:
      上: perplexity vs steps (含随机基线 + 目标线)
      下: decode_acc vs steps (含随机基线)

    data: list of (step, perplexity, decode_acc)
    output_path: PNG 输出路径
    """
    if not data:
        print("[WARN] 无可用数据, 跳过绘图")
        return

    steps = [d[0] for d in data]
    ppls = [d[1] for d in data]
    accs = [d[2] for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---- 子图 1: perplexity vs steps ----
    ax1.plot(steps, ppls, "b-o", markersize=3, linewidth=1.5, label="Perplexity", zorder=3)
    # 随机基线 ln(256) ≈ 5.55
    ax1.axhline(y=RANDOM_BASELINE_PERPLEXITY, color="r", linestyle="--",
                linewidth=1.2, label=f"Random baseline (ln(256)={RANDOM_BASELINE_PERPLEXITY:.2f})", zorder=2)
    # 目标线 4.0
    ax1.axhline(y=TARGET_PERPLEXITY, color="g", linestyle="--",
                linewidth=1.2, label=f"Target ({TARGET_PERPLEXITY:.1f})", zorder=2)
    ax1.set_ylabel("Perplexity", fontsize=13)
    ax1.set_title("Stage 2e Decode Perplexity Curve", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, zorder=1)
    # Y 轴下限为 0, 上限留 20% 余量
    ppl_max = max(ppls) if ppls else RANDOM_BASELINE_PERPLEXITY
    ax1.set_ylim(bottom=0, top=max(ppl_max * 1.2, RANDOM_BASELINE_PERPLEXITY * 1.5))

    # ---- 子图 2: decode_acc vs steps ----
    ax2.plot(steps, accs, "m-s", markersize=3, linewidth=1.5, label="Decode Accuracy", zorder=3)
    # 随机基线 1/256 ≈ 0.39%
    ax2.axhline(y=RANDOM_BASELINE_DECODE_ACC, color="r", linestyle="--",
                linewidth=1.2, label=f"Random baseline (1/256={RANDOM_BASELINE_DECODE_ACC:.2f}%)", zorder=2)
    ax2.set_xlabel("Training Step", fontsize=13)
    ax2.set_ylabel("Decode Accuracy (%)", fontsize=13)
    ax2.set_title("Stage 2e Decode Accuracy Curve", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, zorder=1)
    # Y 轴下限为 0
    acc_max = max(accs) if accs else 1.0
    ax2.set_ylim(bottom=0, top=max(acc_max * 1.5, RANDOM_BASELINE_DECODE_ACC * 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图表已保存: {output_path}")


# -----------------------------------------------------------------------------
# 主入口
# -----------------------------------------------------------------------------
def main():
    # 命令行参数: 可选的日志路径
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = DEFAULT_LOG_PATH

    print(f"[*] 解析训练日志: {log_path}")

    # 1. 解析日志
    data = parse_log(log_path)
    if not data:
        print(f"[ERROR] 未能从日志中解析到任何 perplexity 数据: {log_path}", file=sys.stderr)
        print("[ERROR] 支持的格式:", file=sys.stderr)
        print("  [Eval] step=NNNN perplexity=X.XX decode_acc=X.XX%", file=sys.stderr)
        print("  [Stage2e Decode] step=NNNN avg_loss=X perplexity=X accuracy=X% ...", file=sys.stderr)
        sys.exit(1)

    print(f"[*] 已解析 {len(data)} 个数据点")

    # 2. 输出统计
    print_statistics(data)

    # 3. 绘制图表
    plot_curves(data, OUTPUT_PNG)


if __name__ == "__main__":
    main()
