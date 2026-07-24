#!/bin/bash
# =============================================================================
# Stage 2e 训练启动脚本 (Linux / DGX Spark)
# =============================================================================
# 用法:
#   ./run_train.sh                  # 默认 3M 步完整发育训练
#   ./run_train.sh 100000           # 自定义步数 (如 100K 烟雾测试)
#   ./run_train.sh 3000000 bg       # 后台运行 (nohup)
#   ./run_train.sh 3000000 bg src/stage2e/checkpoints/ckpt_step800000.snn2e
#
# 训练日志: training_<steps>.log
# 检查点:   checkpoints/ckpt_step*.bin (每 50K 步)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 参数
STEPS="${1:-3000000}"
MODE="${2:-fg}"
RESUME_PATH="${3:-}"
LOG_FILE="src/stage2e/training_${STEPS}.log"
RUN_ARGS=(--steps "$STEPS" --text data/lccc_sample_1mb.txt \
  --checkpoint-dir src/stage2e/checkpoints)
if [ -n "$RESUME_PATH" ]; then
    RUN_ARGS+=(--resume "$RESUME_PATH")
fi

# 检查二进制
if [ ! -f src/stage2e/build/snn_stage2e_p1 ]; then
    echo "ERROR: build/snn_stage2e_p1 not found. Run ./build_p1.sh first."
    exit 1
fi

# 检查数据文件 (相对于项目根目录)
if [ ! -f data/lccc_sample_1mb.txt ]; then
    echo "ERROR: data/lccc_sample_1mb.txt not found"
    echo "Please ensure LCCC corpus is in place."
    exit 1
fi

# 创建检查点目录
mkdir -p src/stage2e/checkpoints

# GPU 信息
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 启动训练
echo "=== Starting training: $STEPS steps ==="
echo "Log: $LOG_FILE"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "bg" ]; then
    # 后台运行 (适合长训练, SSH 断开不中断)
    nohup src/stage2e/build/snn_stage2e_p1 "${RUN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > src/stage2e/training.pid
    echo "Training started in background (PID: $PID)"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Stop safely: kill \$(cat src/stage2e/training.pid)"
else
    # 前台运行
    src/stage2e/build/snn_stage2e_p1 "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi
