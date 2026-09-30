# 通用电路指标指南

vcal 把仿真结果先转换成带有分析类型的波形数据，再在这些波形上计算一个
有限的实数指标。同一套表达式可用于放大级、比较器、采样保持、时钟和偏置
等内部模块的建立时间、直流增益、时序、误差及功耗。指标计算本身是 NumPy 数值代码；它不
生成 testbench，也不替代 Spectre 对模型和电路的验证。

## 1. 配置分析和指标

一个配置至少有 "tran" 或 "dc"，也可以同时配置二者。DC 扫描当前只由
Spectre 支持；HSpice 配置不能包含 "dc"。两个分析都配置时，旧式字符串
指标默认使用 "tran"；只有 DC 时则默认使用 "dc"。需要明确选择分析时，使用
带 "analysis" 和 "expr" 的形式：

~~~yaml
simulator: spectre
stimuli: examples/my_tb.scs

tran: {stop: 200n, maxstep: 50p}
dc: {source: VBIAS, start: 0, stop: 1.8, step: 10m}

metrics:
  # 旧式字符串：两个分析都存在时默认使用 tran
  power: "avg(-I('VDD')) * 1.8"
  # 显式指定分析
  dc_gain: {analysis: dc, expr: "dc_gain(V('out'), V('in'))"}
  settle: {analysis: tran, expr: "settle_time(V('out'), start=0, end=200n, final=1.0)"}

objective: {metric: dc_gain, goal: maximize}
constraints:
  - {metric: settle, max: 100n}
~~~

"tran" 的 "stop" 和 "maxstep" 必须为正数。"dc" 只接受
"source/start/stop/step"：source 是网表中的 source instance 名，步长必须
朝向终点且扫描至少两个点。GUI 的 Circuit metrics 页面编辑这些字段、
指标表达式和常用模板；保存信号列表以空格分隔。点击“应用测量与分析”后，
指标才会出现在目标和约束选择器中；保存配置将保留这些修改。

指标值可以是字符串，也可以是包含 "expr"、"analysis" 和可选 "window" 的映射。表达式支持
工程后缀（如 50p、1.8、20u）和用 m["name"] 引用另一个指标。定义会
自动按依赖排序；未知指标、循环依赖、未配置的分析和不允许的语法在仿真前
报告。每个最终结果必须是有限的实标量；数组、字典和 inf 必须先选取字段
或归约。一个理想的无噪声 FFT 可能返回 inf，这会被外层指标检查拒绝，
不能用静默的数值夹紧掩盖它。

measurement_signature(defs, analyses, save=()) 把测量版本、指标定义、分析
配置和保存信号纳入身份。续跑时签名不一致的历史会跳过；没有旧签名的历史
也跳过，避免用不同公式混合优化数据。

多角点评估仍按 `corner_worst` 归约。对建立时间等越小越好的量请指定 `max`，
对增益等越大越好的量指定 `min`；默认归约为 `min`，不会猜测指标的优化方向。

## 2. 波形对象和单位

后端入口是：

~~~python
Waves(t, sigs, analysis="tran")
compute_metrics(defs, t=None, sigs=None, *, datasets=None,
                default_analysis="tran")
~~~

t 是严格单调、有限的一维轴。瞬态轴的单位是秒；DC 轴是扫描源的物理
单位，通常是伏特。每个保存信号必须与轴等长且有限。DC 轴如果严格递减会被
转为递增，其他非单调轴拒绝。V(name) 查找保存的电压，I(srcname) 查找
source current（兼容 :p、无后缀和 :1 名称）；电流正负号沿用仿真器的
保存结果，功耗表达式应明确写出所需符号。
Spectre PSF 转换使用 17 位有效数字保留双精度；转换器默认的 6 位有效数字
会给高分辨率 ADC 和小信号测量额外引入量化误差。
启动器将 EDA 的 `LD_LIBRARY_PATH` 保留到 `VCAL_SIM_LD_LIBRARY_PATH`，
只为 Spectre、HSpice 和 PSF 子进程恢复，避免与 Python／Qt 的库冲突。
若直接运行 Python，可显式提供 `VCAL_SIM_LD_LIBRARY_PATH`；正常使用
`vcal` 入口会自动传递已有环境。

窗口端点必须落在已保存的轴范围内，不能外推。窗口计算会在线性插值后把
端点加入积分/平均；没有足够的不同样本、空窗口、反向窗口和非有限波形都会
失败。函数的自然输出单位如下：

| 函数 | 输出 | 单位或含义 |
| --- | --- | --- |
| V(name) / I(srcname) | 波形数组 | 保存信号本身；V 或 A 等模型单位 |
| avg / rms / vmax / vmin | 标量 | 与输入信号相同 |
| std / pp | 标量 | 与输入信号相同；pp = vmax - vmin |
| integ | 标量 | 输入单位乘轴单位，例如 A·s 或 W·s=J |
| at | 标量 | 指定轴位置的线性插值，单位同输入 |
| slice | 数组 | 窗口内原始样本，不自动外推 |
| sample | 数组 | 重新采样后的波形，fs 为轴单位的倒数 |
| cross / delay | 标量 | 轴单位，例如秒 |
| rise_time / fall_time / settle_time | 标量 | 轴单位，例如秒 |
| slew_rate | 标量 | 输入单位/轴单位，例如 V/s |
| dc_gain | 标量 | 输出输入单位之比，例如 V/V；dB 要显式 20*log10(abs(...)) |
| dc_offset | 标量 | 输出单位，例如 V |
| settled | 0 或 1 | 是否在窗口内满足误差和 hold 条件 |
| settling_error | 标量 | 输出单位；relative=True 时为相对误差 |
| overshoot | 标量 | 相对最终步幅的无量纲比例 |

常用方法的完整签名如下。x、y 可以是 V(...)、I(...) 或相同长度的
有限数组。

~~~python
at(x, tq)
avg(x, t0=None, t1=None)
rms(x, t0=None, t1=None)
std(x, t0=None, t1=None)
integ(x, t0=None, t1=None)
vmax(x, t0=None, t1=None)
vmin(x, t0=None, t1=None)
pp(x, t0=None, t1=None)
slice(x, t0, t1)
sample(x, fs, nsamp, t0=None, method="linear")

cross(x, level, edge="rising", nth=1, start=None, end=None)
rise_time(x, start=None, end=None, low=.1, high=.9, initial=None, final=None)
fall_time(x, start=None, end=None, low=.1, high=.9, initial=None, final=None)
delay(x, y, level_x, level_y, edge_x="rising", edge_y="rising",
      nth=1, start=None, end=None, pairing="causal")
slew_rate(x, start=None, end=None, low=.1, high=.9,
          initial=None, final=None)

settle_time(x, start=None, end=None, tol=.005, final=None, atol=None,
            initial=None, hold=0, on_unsettled="raise")
settled(x, start=None, end=None, tol=.005, final=None, atol=None,
        initial=None, hold=0, on_unsettled="raise")
settling_error(x, start=None, end=None, final=None,
               relative=False, initial=None)
overshoot(x, start=None, end=None, initial=None, final=None)

dc_gain(y, x=None, at=None, start=None, end=None)
dc_offset(y, x=None, start=None, end=None)
~~~

rise_time、fall_time 和 slew_rate 以 low/high 定义初始到最终步幅的
比例，默认 10% 到 90%。cross 的 edge 可以是 rising 或 falling，
nth 从 1 开始。`cross` 返回绝对交越时刻；恰好在窗口边界的交越计一次，
阈值平台以首次到达阈值的时刻计，触碰阈值后回到同侧不计交越。
`delay` 默认将第 nth 个输入沿配给其后、下一个同向输入沿之前的第一个输出沿。
缺少该周期输出沿会报错，不借用前一周期或后一周期的边沿。
如确实需要有符号的第 nth 输入／输出沿之差，显式使用 `pairing="ordinal"`。
上升／下降时间会跳过未完成的早期毛刺，用首个完整转换的低、高阈值交越。
有振铃、多次交越时应收紧窗口并明确阈值；软件不会猜测哪个毛刺是真实事件。

settle_time 默认 on_unsettled="raise"：窗口结束前没有进入并保持误差带
时，指标失败；只有明确选择 on_unsettled="window" 才会返回整个窗口长度。
若用这种保留有限值的模式，必须同时加入 settled(...) == 1 的约束，否则
优化器可能把“未建立”当成可行的建立时间。没有给出 atol 时，误差带是
tol * abs(final - initial)；一旦给出 atol，它就是输出单位的绝对带宽并覆盖
tol。例如 ±1 mV 建立误差可写成 `atol=1m`。

dc_gain(y, x=None) 在没有 at 时对所选 DC 曲线作最小二乘斜率；有 at
时返回该点所在相邻采样段的局部斜率。x=None 时使用分析轴。dc_offset
返回相同线性拟合的截距。DC 输入必须严格单调且有非零范围。

## 3. 从指定时刻开始测量

GUI 为每个指标提供“开始时间 (s)”和“结束时间 (s)”。可输入 `5u`、`20n`
等工程数值，空白端点使用保存波形的对应边界。YAML 对应：

~~~yaml
metrics:
  settling_s:
    analysis: tran
    window: {start: 5u, end: 10u}
    expr: "settle_time(V('out'), final=1, initial=0, atol=1m, hold=1u)"
  ripple_v:
    analysis: tran
    window: {start: 9u, end: 10u}
    expr: "pp(V('out'))"
  comparator_delay_s:
    analysis: tran
    window: {start: 5u, end: 10u}
    expr: "delay(V('in'), V('out'), level_x=0, level_y=.5)"
  supply_power_w:
    analysis: tran
    window: {start: 5u}  # 到已保存波形末尾
    expr: "avg(-V('vdd') * I('VDD'))"
~~~

窗口在表达式求值**之前**裁剪全部信号，`V`、`I`、`np`、`t` 和 `axis`
都只看到所选区间，端点不在原采样点上时线性插值。时间轴保留绝对仿真时间。
因此 `cross(...)` 返回绝对时刻，`settle_time(...)` 返回距本次窗口开始的
建立用时，`delay`、脉宽和周期返回两沿间的时间差；不能混用这几种含义。

所有默认时间参数为 `None`，表示当前指标窗口边界；`sample` 默认从窗口首点
重新采样。公式中已有的 `start/end` 或 `t0/t1` 可进一步缩小窗口；如果超出
该指标窗口会报错。旧 YAML 不加 `window` 时仍读取全段数据，旧显式时间参数
保持可用。模板里的时间是可修改的例子，需要配合实际激励选择。

每项指标独立选窗，`m['name']` 引用的是另一个指标按其自身窗口得到的标量。
例如可引用 DC 增益确定瞬态最终值。改变窗口会改变测量签名并隔离旧历史。
窗口负数、反向、未知字段、超出 `tran.stop` 会在预检时拒绝；实际波形提前
结束或保存范围不足，在求值时拒绝。空窗口、单时刻窗口不能用于周期或积分。

**DC 扫描没有仿真时间轴。** DC 行禁用时间栏，`window` 用于 DC 会被拒绝。
`dc_gain(..., start=-10m, end=10m)` 的范围是扫描源数值，`at=0` 是输入值。
若需要在某个时刻之后估计准静态斜率，可对明确的单调瞬态输入窗口使用
`dc_gain(V('out'), V('in'))`，但动态滞后会进入该结果，不能当作 DC 扫描增益。

## 4. 内部模块补充指标

~~~python
pulse_width(x, level, polarity="high", nth=1, start=None, end=None)
period(x, level, edge="rising", start=None, end=None)
frequency(x, level, edge="rising", start=None, end=None)
duty_cycle(x, level, start=None, end=None)
period_jitter(x, level, edge="rising", start=None, end=None)
cycle_jitter(x, level, edge="rising", start=None, end=None)
slope(x, start=None, end=None)
value_at_cross(x, trigger, level, edge="rising", nth=1, start=None, end=None)
~~~

设阈值交越时刻为 `e_i`，同向边沿间隔为 `T_i=e_(i+1)-e_i`：

| 指标 | 本实现的数值定义 | 窗口要求 |
| --- | --- | --- |
| 脉宽 | 高脉冲上升→下降的时间差；`polarity='low'` 则相反 | nth 只计窗内完整脉冲；窗首／尾的半脉冲不计 |
| 周期 | `mean(T_i)` | 至少两个同向边沿 |
| 频率 | `1/mean(T_i)`，不是各周期频率的均值 | 至少两个同向边沿 |
| 占空比 | 完整上升沿→上升沿周期内，高时间总和／周期总时长 | 至少一个完整周期，返回 0–1 |
| 周期抖动 | `sqrt(mean((T_i-mean(T_i))**2))` | 至少两个周期，使用总体标准差 |
| 周期间抖动 | `sqrt(mean((T_(i+1)-T_i)**2))` | 至少两个相邻周期 |
| 保持下垂率 | `slope`：分段线性波形按时间加权的最小二乘斜率 | 选择进入保持模式后的窗口；V/s，有符号 |
| 触发时读数 | 在 trigger 第 nth 个指定方向的阈值交越时刻插值读取 x | 交越必须在窗口内；输出单位同 x |
| 跟踪误差 RMS | `rms(V('out')-V('in'))` | 选择跟踪模式的窗口，V |
| 峰值跟踪误差 | `vmax(abs(V('out')-V('in')))` | 选定窗口，V |
| 峰值电流 | `vmax(abs(I('VDD')))` | 选定窗口，A |
| 电荷 | `integ(-I('VDD'))` | 选定窗口，C；核对电流方向 |

边沿、脉宽、占空比的术语依据 [Tektronix TDSJIT2 手册附录 A](https://download.tek.com/manual/071081402.pdf)。
vcal 明确采用上表的聚合方式；改变窗口可能改变所含周期的数量，抖动不是只有
一个周期时的“0”。振荡器相位噪声、随机抖动需来自相应仿真或测量数据，普通
确定性 tran 不会自动加入噪声。

保持下垂、采集时间、保持切换误差适用于采样保持内部模块，参考
[ADI AN-1515](https://www.analog.com/en/resources/app-notes/an-1515.html)。
采集时间可用绝对误差带的 `settle_time`，保持切换误差可在切换前后分别用
`at` 或 `avg` 得到标量再相减；下垂用保持区间的 `slope`。拟合斜率按时间
加权，自适应采样点变密不会改变同一分段线性波形的拟合结果。

比较器可用 `value_at_cross(V('in'), V('out'), .5)` 测输出翻转时输入电压，
分别选择上升和下降输入扫描估算阈值差。该数值包含输入斜率与传播延迟的影响；
应声明过驱动、输出负载及门限，参见
[ADI 比较器传播延迟测量条件](https://www.analog.com/en/resources/technical-articles/parameters-that-affect-comparator-propagation-delay-measurements.html)。

## 5. 模板逐项覆盖与验证

GUI 提供 28 个内部模块模板：25 个瞬态指标均可设开始／结束时间，3 个 DC
指标使用扫描范围。模板仅是可编辑公式，不为任意电路自动选择激励或规格。

| 模板名称 | 是否可从指定时刻开始 | 数值验证依据 |
| --- | --- | --- |
| settling_s、settling_abs_s、settled_ok、settling_error_v | 是 | 明确初始值、最终值及误差带的阶跃 |
| rise_s、fall_s、slew_v_s、overshoot_ratio | 是 | 已知斜率、阈值与峰值的波形 |
| delay_s | 是 | 已知输入／输出沿；晚开始不配错周期 |
| ripple_vpp、noise_rms_v | 是 | 峰峰值与时间加权 RMS 的解析值 |
| power_w、energy_j、peak_current_a、charge_c | 是 | 已知电压、电流和积分时长 |
| pulse_width_s、period_s、frequency_hz、duty_ratio | 是 | 已知交越时刻的完整脉冲和周期 |
| period_jitter_s、cycle_jitter_s | 是 | 已知变化周期序列及其标准差／RMS差 |
| droop_v_s | 是 | 线性保持下垂及非均匀采样 |
| trigger_value_v | 是 | 已知触发交越位置及待测输入斜率 |
| tracking_rms_v、tracking_peak_v | 是 | 已知输出输入之差 |
| dc_gain_v_v、dc_gain_db、dc_offset_v | 无时间轴；设置扫描范围 | 已知线性传输及偏移；时间窗须拒绝 |

`tests/test_metric_windows.py` 对每个模板执行独立期望值、窗口前后污染及时间
整体平移检查，并验证非采样端点、单边界、非法窗口和旧格式兼容。
`tests/test_measurement_edges.py` 检查阈值边界、前一周期输出、缺少响应与毛刺。
GUI 测试覆盖保存／加载窗口、DC 时间栏和中日英切换。

[RC 参考配置](../examples/metrics_rc/rc.yaml) 可直接在 GUI 打开，包含 tran+DC
和跨分析指标依赖。验证夹具使用理想无源元件及已知源波形，不依赖 PDK；它验证
测量软件的数值与时间窗口行为，不代表用户内部模块已通过器件模型或 PVT 验证。

[时序／保持参考配置](../examples/metrics_blocks/blocks.yaml) 包含早期不规则脉冲，
将测量窗口设为 5–10 μs；延迟项从 5.25 μs 开始，刻意落在前一输入及其输出
之间。本地 Spectre 23.1 在 10 ns／5 ns 两个最大步长下的结果如下：

| 测量 | 独立预期 | Spectre 结果（10 ns／5 ns） |
| --- | --- | --- |
| 延迟 | 150 ns | 150／150 ns |
| 脉宽 | 400 ns | 400／400 ns |
| 平均周期 | 1.025 μs | 1.025／1.025 μs |
| 频率 | 975609.7561 Hz | 975609.7561／975609.7561 Hz |
| 占空比 | 0.4/1.025 | 0.3902439024／0.3902439024 |
| 周期／周期间抖动 | 25／50 ns | 两步长均为 25／50 ns |
| 保持斜率 | −2000 V/s | −2000／−2000 V/s |
| 触发时读数 | 0.785 V | 0.785／0.785 V |
| 平均功率 | 1.8 mW | 1.8／1.8 mW |
| RC 1% 建立时间 | 2.303085 μs | 2.303019／2.303071 μs |
| RC 上升时间 | 1.098612 μs | 1.098565／1.098599 μs |
| RC DC 增益／偏移 | 0.5 V/V、0 V | 两步长均为 0.5 V/V、0 V |

四次仿真均为 0 errors、0 warnings，原始波形完成时间、有限性、严格递增性及
最大采样间隔均检查通过；时间量的步长减半差异低于预先规定的 0.1%。
夹具中的抖动由已知源边沿序列注入，用来核验统计公式，不是器件随机噪声仿真。

旧 FFT、ADC 静态和解码函数继续支持已有配置，详见
[兼容接口说明](LEGACY_ADC_METRICS.md)。
