#!/bin/bash
# circuit-opt 本地 GUI（离线 BO/DKL 调参）。
# 按硬件自动选择 Python 环境（NVIDIA→venv-cuda, AMD→venv-rocm, 否则 cpu）。
# 覆盖: OPTLOCAL_ENV=cuda|rocm|cpu ./run_gui.sh
# LD_LIBRARY_PATH 必须清掉：Cadence 环境里的 libshm.so 会遮蔽 torch 同名库。
cd "$(dirname "$0")/local"

detect() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo cuda
    elif [ -e /dev/kfd ] && lsmod 2>/dev/null | grep -q amdgpu; then
        echo rocm
    else
        echo cpu
    fi
}

FLAVOR=${OPTLOCAL_ENV:-$(detect)}
# 候选顺序: 精确匹配 -> 旧版 venv(rocm) -> 任何存在的环境
for d in "venv-$FLAVOR" venv venv-cuda venv-rocm venv-cpu; do
    if [ -x "$d/bin/python" ]; then VENV=$d; break; fi
done
if [ -z "$VENV" ]; then
    echo "未找到 Python 环境。先运行: local/install_env.sh  (auto 检测硬件)"
    exit 1
fi
[ "$VENV" != "venv-$FLAVOR" ] && [ "$VENV" != venv ] && \
    echo "提示: 硬件类型 $FLAVOR 的环境不存在, 使用 $VENV (可能回退 CPU)。安装: local/install_env.sh $FLAVOR"
exec env -u LD_LIBRARY_PATH "./$VENV/bin/python" -m optlocal gui "$@"
