#!/bin/bash
# =============================================================================
# Stage 2e 训练启动脚本 (Linux / DGX Spark)
# =============================================================================
# 用法:
#   ./run_train.sh                  # 默认 3M 步完整发育训练
#   ./run_train.sh 100000           # 自定义步数 (如 100K 烟雾测试)
#   ./run_train.sh 3000000 bg       # 后台运行 (nohup)
#
# 训练日志: training_<steps>.log
# 检查点:   checkpoints/ckpt_step*.bin (每 50K 步)
# =============================================================================
set -e

cd "$(dirname "$0")"

# 参数
STEPS="${1:-3000000}"
MODE="${2:-fg}"
LOG_FILE="training_${STEPS}.log"

# 检查二进制
if [ ! -f build/snn_stage2e_p1 ]; then
    echo "ERROR: build/snn_stage2e_p1 not found. Run ./build_p1.sh first."
    exit 1
fi

# 检查数据文件 (相对于项目根目录)
cd ../..
if [ ! -f data/lccc_sample_1mb.txt ]; then
    echo "ERROR: data/lccc_sample_1mb.txt not found"
    echo "Please ensure LCCC corpus is in place."
    exit 1
fi

# 创建检查点目录
mkdir -p src/stage2e/checkpoints

cd src/stage2e

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
    nohup ./build/snn_stage2e_p1 --steps "$STEPS" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > training.pid
    echo "Training started in background (PID: $PID)"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Stop:    kill \$(cat training.pid)"
else
    # 前台运行
    ./build/snn_stage2e_p1 --steps "$STEPS" 2>&1 | tee "$LOG_FILE"
fi
