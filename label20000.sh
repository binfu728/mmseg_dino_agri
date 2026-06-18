#!/usr/bin/env bash
# 修改后的同目录多卡训练脚本
set -e

# 1. 获取当前脚本所在的这个文件夹作为根目录（去掉了原先的 /..）
MMSEG_ROOT="$(cd "$(dirname "$0")" && pwd)"

GPUS=${1:-8}; shift || true
PORT=${PORT:-29555}

# 2. 直接指向当前目录下的 config 文件（请把下面这个名字换成你实际的 config 文件名）
CONFIG="$MMSEG_ROOT/configs/dinov3l_m2f_agri.py"

# 3. 在当前目录下创建 work_dirs 文件夹
WORK_DIR="$MMSEG_ROOT/work_dirs/123"
PY=${PY:-python}

mkdir -p "$WORK_DIR"
cd "$MMSEG_ROOT"
export PYTHONPATH="$MMSEG_ROOT:${PYTHONPATH:-}"

# 4. 直接运行当前目录下的 train.py（去掉了原先的 tools/）
nohup "$PY" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$GPUS" --master_port="$PORT" \
    train.py "$CONFIG" --launcher pytorch \
    --work-dir "$WORK_DIR" "$@" \
    > "$WORK_DIR/nohup.log" 2>&1 &

echo "PID: $!  GPUS: $GPUS  log: $WORK_DIR/nohup.log"