# vcal — 面向 Virtuoso 网表的、变差感知调参工具

**[English](README.en.md) · [中文] · [日本語](README.ja.md)**

基于**深度核高斯过程贝叶斯优化**的模拟/混合信号 IC **尺寸与偏置自动调参**
工具。指向一份 Virtuoso 导出的网表，声明"什么是好"，它就在器件尺寸空间
里搜索以达成目标，并把最优尺寸回写到原理图。

完全离线：仿真和优化都跑在你自己的机器上，不依赖云端或网络。

## 特性

- **自定义目标** — 最大化 / 最小化 / **逼近目标值**（如"增益 → 8×"），
  可多指标加权求和。
- **任意指标** — 保存节点的波形表达式：`avg`、`settle_time`、
  `overshoot`、相干 FFT 的 `enob` 等。
- **PVT 环境角点** — 定义温度/电压/工艺角，优化自动针对**最差角点**。
- **本地 GPU 自动检测** — NVIDIA(CUDA) / AMD(ROCm)，无则回退 CPU；也可把
  GP 计算**卸载到远程 GPU 机器**。
- **原理图回写** — 经 virtuoso-bridge 把最优尺寸写回原理图（预览+确认）。
- **PDK 可换** — 换 PDK 只改一行；电路配置里不出现 PDK 路径。
- **GUI 或 CLI** — 仿 gmid-tool 的桌面界面，或脚本化命令行。

## 算法

Sobol 初始化 → Matérn GP（样本足够后切换**深度核 GP**：MLP 特征提取器 +
GP，在 GPU 上联合训练）→ **TuRBO** 信赖域内的批量约束 **qLogNEI** 采集。
不收敛的点悲观填补，代理模型自动学会避开。

> 注：典型 sizing 问题（几十~几百点、10-20 维）下 GP/DKL 是小矩阵运算，
> **CPU 通常最快**，GPU 只在大规模问题上占优。`auto` 默认用 CPU。

## 安装

```bash
./vcal install            # 自动检测硬件 (cuda / rocm / cpu)
export PDK_ROOT=/path/to/your/pdk     # 供 pdks/*.yaml 引用
```

需 Python ≥ 3.10 及 SPICE 仿真器(HSpice/Spectre) 在 PATH。安装器按硬件
建 venv(`local/venv-<类型>`)，装 torch+botorch+PyQt5。

## 用法

```bash
./vcal gui                            # 图形界面
./vcal run examples/demo_ota.yaml --budget 100 --batch 4
./vcal serve-gpu --port 8494          # 在 GPU 机器上，把 GPU 借出
```

CLI 选项：`--budget N` 总评估数、`--batch q` 并行提案、
`--device auto|cuda|cpu`、`--remote-gpu URL`、`--resume`(从历史续跑)。

## 定义一个电路

一个电路就是一份 YAML（见 [`examples/demo_ota.yaml`](examples/demo_ota.yaml)）:

| 字段 | 说明 |
|------|------|
| `pdk` | `pdks/<name>.yaml` 的名字（电路文件里不含 PDK 路径） |
| `netlist` / `stimuli` | 你的器件网表 + 激励 deck |
| `params` | 优化变量：`devices`、`attr`(`w`/`l`/`mr`/`value`)、`lo`/`hi`/`log`/`integer`。一个变量可驱动多个器件(对称对共用) |
| `metrics` | 波形表达式，支持工程记数(`90n`)、`m['...']` 引用前面的指标 |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT 列表：`{name, temp, pdk_corner, tb_extra}` + `corner_worst` 归约 |

改 MOS 的 `w` 时版图衍生参数(as/ad/ps/pd…)自动同步缩放，语义等同改原理图。

## PDK 配置

PDK 定义在 `pdks/<name>.yaml`（见 [`pdks/example.yaml`](pdks/example.yaml)）。
一个 PDK 文件列出模型库路径（用 `${PDK_ROOT}` 避免把绝对路径入库）和各
工艺角要 include 的 `.lib` section。**换 PDK 只改 `pdk:` 一行**，电路配置
从不包含 PDK 路径。

## 目录结构

```
vcal                    统一入口 (gui / run / serve-gpu / install)
server/optserver/       网表改写 + 仿真 + 指标提取
client/optclient/       BoTorch 优化器 + 远程 GPU 计算服务
local/optlocal/         离线引擎 + PyQt5 GUI + 原理图回写
pdks/                   PDK 定义（example.yaml 是模板）
examples/               脱敏示例电路配置
```

## 许可证

见 [LICENSE](LICENSE)。
