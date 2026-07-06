#!/bin/bash
# 为本地优化工具创建 Python 环境。
# 用法: ./install_env.sh [auto|cuda|rocm|cpu]
#   auto  — 检测硬件: NVIDIA -> cuda, AMD(/dev/kfd) -> rocm, 否则 cpu
#   cuda  — NVIDIA 显卡 (venv-cuda, torch cu124)
#   rocm  — AMD 显卡     (venv-rocm, torch rocm6.2; RX 6800/6900 XT 等)
#   cpu   — 无显卡       (venv-cpu)
# 同一台机器可以装多个环境, run_gui.sh 启动时按硬件自动挑选。
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-/usr/bin/python3.11}

detect() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo cuda
    elif [ -e /dev/kfd ] && lsmod 2>/dev/null | grep -q amdgpu; then
        echo rocm
    else
        echo cpu
    fi
}

FLAVOR=${1:-auto}
[ "$FLAVOR" = auto ] && FLAVOR=$(detect) && echo "检测到硬件类型: $FLAVOR"

case "$FLAVOR" in
    cuda) DIR=venv-cuda; IDX="https://download.pytorch.org/whl/cu124";;
    rocm) DIR=venv-rocm; IDX="https://download.pytorch.org/whl/rocm6.2";;
    cpu)  DIR=venv-cpu;  IDX="https://download.pytorch.org/whl/cpu";;
    *) echo "未知类型: $FLAVOR"; exit 1;;
esac

echo "创建 $DIR (torch index: $IDX) ..."
[ -d "$DIR" ] || "$PY" -m venv "$DIR"
"./$DIR/bin/pip" install -q --upgrade pip
"./$DIR/bin/pip" install torch --index-url "$IDX"
"./$DIR/bin/pip" install -q botorch gpytorch numpy requests pyyaml matplotlib PyQt5
env -u LD_LIBRARY_PATH "./$DIR/bin/python" -c "
import torch
print('torch', torch.__version__)
print('GPU 可用:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(将用 CPU)')"
echo "完成: local/$DIR"
