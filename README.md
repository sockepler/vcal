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
./vcal check examples/demo_ota.yaml         # 只校验配置，不仿真
./vcal check examples/demo_ota.yaml --json  # 输出机器可读 spec
./vcal serve-gpu --port 8494          # 在 GPU 机器上，把 GPU 借出
```

CLI 选项：`--budget N` 总评估数（不会超过 N）、`--batch q` 并行提案、
`--device auto|cuda|rocm|cpu`、`--remote-gpu URL`、`--resume`(默认从历史续跑)
或 `--no-resume`（忽略历史）。

`check` 会检查配置结构、引用文件、器件、参数范围、指标表达式语法、符号、
指标依赖和 analysis 归属；它不运行真实仿真，保存信号是否存在仍在实际求值时
验证。

## 通用电路指标

GUI 的 **Circuit metrics** 页面可编辑 transient 的 stop/maxstep、Spectre DC
的 source/start/stop/step，以及指标表达式。它提供 24 个建立时间、DC gain、FFT、
DNL/INL、功率和能量模板；模板或自定义指标通过校验并应用后，才会出现在目标
和约束选择器中，已被目标或约束引用的指标不能直接删除。`tran`、`dc` 可以单独配置
或同时配置；指标值可为旧字符串，也可写成 `{analysis: dc, expr: "..."}`。完整签名、单位和边界见
[`docs/METRICS.md`](docs/METRICS.md)；配置到应用的流程见
[`docs/OTA_WORKFLOW.md`](docs/OTA_WORKFLOW.md)。

~~~yaml
tran: {stop: 200n, maxstep: 50p}
dc: {source: VBIAS, start: 0, stop: 1.8, step: 10m}
metrics:
  gain_vv: {analysis: dc, expr: "dc_gain(V('out'), V('in'))"}
  settle: {analysis: tran, expr: "settle_time(V('out'), start=0, end=200n, final=1.0)"}
  sndr: "sndr_fft(V('adc_out'), 1G, 7, 256, t0=0)"
constraints:
  - {metric: settle, max: 20n}
~~~

`settle_time` 默认未建立即失败；要保留窗口长度，必须显式使用
`on_unsettled: window` 并加 `settled(...) == 1` 约束。FFT 的 fund 是 bin 而非
Hz。`adc_static(method='histogram')` 只用于完整均匀 ramp code-density，
`adc_transitions` 要求完整转换边界，`decode_bits` 接收可由
`np.column_stack` 构成的 `(samples, bits)` 波形。

## 可选 gm/Id 与 scope 辅助

gm/Id 面板仍可离线加载 NPZ/CSV LUT，把尺寸建议作为优化初值；scope 仍可让
CLI/GUI 先检查一个 sub-cell 范围。合成演示表不含 PDK 数据：

~~~bash
./vcal gmid examples/gmid_demo.csv --length 180n --vds 0.6 --vsb 0 --gmid 15 --id 20u --json
~~~

这些功能不自动生成 PDK LUT、OTA AC testbench 或新的激励。当前优先级是通用
tran/dc、FFT、ADC 静态和功耗/能量测量；OTA 专用平台属于可选远期方向。

### 界面语言

GUI 顶部的语言下拉框可即时切换中文、日文或英文；选择语言后会立即刷新界面，
并保存为下次启动的偏好。CLI 可在子命令前或后指定语言，例如
`./vcal --lang ja gui`、`./vcal gui --lang en`，也可用
`VCAL_LANG=ja ./vcal gui`。语言优先级为显式 `--lang`、`VCAL_LANG`、已保存的
GUI 偏好，最后回退到中文。翻译只作用于界面和 CLI 文案；Spectre 等第三方日志、
用户定义的 metrics 名称以及网表标识符保持原文。
顶层帮助和 `install --help` 由 shell 直接输出，使用 `--lang` 或 `VCAL_LANG`，
不读取保存的 GUI 语言偏好。

## GUI 迭代图

GUI 的迭代图横轴是评估次数，指标下拉框包含配置中的全部 `metrics`。蓝色点
表示满足约束，橙色点表示违反约束，红色叉表示仿真失败；失败点不填入虚假
的纵坐标值。最佳可行趋势线只用于目标指标。点击点可查看该次评估的参数、
指标和错误信息，图表支持缩放、平移以及 PNG/CSV 导出。

GUI 默认把结果写到 YAML 旁的 `../work/<name>/gui_run/`；CLI 使用同一电路的
`cli_run/` 子目录。目录中会自动生成 `best.json`（存在成功评估时才生成，
`feasible` 表示是否满足约束）、`summary.json`、`state.json` 和
`history.csv`。`vcal run --resume` 会恢复已有记录，并跳过重复参数点的仿真。

## 定义一个电路

一个电路就是一份 YAML（见 [`examples/demo_ota.yaml`](examples/demo_ota.yaml)）:

| 字段 | 说明 |
|------|------|
| `pdk` | `pdks/<name>.yaml` 的名字（电路文件里不含 PDK 路径） |
| `netlist` / `stimuli` | 你的器件网表 + 激励 deck |
| `params` | 优化变量：`devices`、`attr`(`w`/`l`/`mr`/`value`)、`lo`/`hi`/`log`/`integer`；可用 `enabled: false` 禁用某个变量。一个变量可驱动多个器件(对称对共用) |
| `metrics` | 波形表达式，支持工程记数(`90n`)、`m['...']` 引用前面的指标 |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT 列表：`{name, temp, pdk_corner, tb_extra}` + `corner_worst` 归约 |

改 MOS 的 `w` 时版图衍生参数(as/ad/ps/pd…)自动同步缩放，语义等同改原理图。
顶层 `fixed` 可用 `{参数名: 数值}` 固定变量。配置中的相对路径优先相对该
YAML 文件解析，同时兼容旧的相对仓库根目录路径。

## PDK 配置

PDK 定义在 `pdks/<name>.yaml`（见 [`pdks/example.yaml`](pdks/example.yaml)）。
一个 PDK 文件列出模型库路径（用 `${PDK_ROOT}` 避免把绝对路径入库）和各
工艺角要 include 的 `.lib` section。**换 PDK 只改 `pdk:` 一行**，电路配置
从不包含 PDK 路径。

## 目录结构

```
vcal                    统一入口 (gui / run / check / serve-gpu / install)
server/optserver/       网表改写 + 仿真 + 指标提取
client/optclient/       BoTorch 优化器 + 远程 GPU 计算服务
local/optlocal/         离线引擎 + PyQt5 GUI + 原理图回写
pdks/                   PDK 定义（example.yaml 是模板）
examples/               脱敏示例电路配置
```

## 许可证

见 [LICENSE](LICENSE)。
