# 通用电路指标指南

vcal 把仿真结果先转换成带有分析类型的波形数据，再在这些波形上计算一个
有限的实数指标。这样同一套表达式可以用于建立时间、直流增益、ADC 动态
性能、静态线性度、功耗和组合 FoM。指标计算本身是 NumPy 数值代码；它不
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

指标值可以是字符串，也可以是只含 "expr"、"analysis" 的映射。表达式支持
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
sample(x, fs, nsamp, t0=0.0, method="linear")

cross(x, level, edge="rising", nth=1, start=None, end=None)
rise_time(x, start=0, end=None, low=.1, high=.9, initial=None, final=None)
fall_time(x, start=0, end=None, low=.1, high=.9, initial=None, final=None)
delay(x, y, level_x, level_y, edge_x="rising", edge_y="rising",
      nth=1, start=None, end=None)
slew_rate(x, start=0, end=None, low=.1, high=.9,
          initial=None, final=None)

settle_time(x, start=0, end=None, tol=.005, final=None, atol=None,
            initial=None, hold=0, on_unsettled="raise")
settled(x, start=0, end=None, tol=.005, final=None, atol=None,
        initial=None, hold=0, on_unsettled="raise")
settling_error(x, start=0, end=None, final=None,
               relative=False, initial=None)
overshoot(x, start=0, end=None, initial=None, final=None)

dc_gain(y, x=None, at=None, start=None, end=None)
dc_offset(y, x=None, start=None, end=None)
~~~

rise_time、fall_time 和 slew_rate 以 low/high 定义初始到最终步幅的
比例，默认 10% 到 90%。cross 的 edge 可以是 rising 或 falling，
nth 从 1 开始。delay 是 y 的 crossing time 减去 x 的 crossing time。

settle_time 默认 on_unsettled="raise"：窗口结束前没有进入并保持误差带
时，指标失败；只有明确选择 on_unsettled="window" 才会返回整个窗口长度。
若用这种保留有限值的模式，必须同时加入 settled(...) == 1 的约束，否则
优化器可能把“未建立”当成可行的建立时间。没有给出 atol 时，误差带是
tol * abs(final - initial)；一旦给出 atol，它就是输出单位的绝对带宽并覆盖
tol。ADC 的半 LSB 目标应把 atol 明确写成 0.5 * Vref / 2**bits。

dc_gain(y, x=None) 在没有 at 时对所选 DC 曲线作最小二乘斜率；有 at
时返回该点所在相邻采样段的局部斜率。x=None 时使用分析轴。dc_offset
返回相同线性拟合的截距。DC 输入必须严格单调且有非零范围。

## 3. FFT 动态指标

Waves 上的便捷包装是：

~~~python
enob(x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs)
sndr(x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs)
snr(x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs)
thd(x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs)
sfdr(x, fs, fund, nsamp, t0_lo=None, t0_hi=None, nphase=48, **kwargs)
~~~

它们固定 fs 和 nsamp 后采样 x，然后分别取 spectrum 返回字典中的
一个字段。fund 是 FFT bin，不是 Hz；若已知频率 f，应使用
fund = f * nsamp / fs。没有显式相位区间时，默认使用固定 t0（未给出
t0 时为波形轴起点），不会自动寻找最佳采样相位。

历史上显式传入 t0_lo 和 t0_hi 会在两者之间取 nphase 个采样起点，
默认选择最佳结果并发出 RuntimeWarning。这是兼容模式；可以传
phase_mode="worst" 选择最差相位，或者传 phase_mode="best" 明确保留旧
行为。固定相位使用 phase_mode="fixed"（只给一个 t0）。对 thd，数值
越大越差；对 enob/sndr/snr/sfdr，数值越小越差。

底层纯 NumPy API 是：

~~~python
spectrum(samples, fs, fund, window="rect", harmonics=5, bin_width=None)
~~~

返回 sndr、snr、thd、sfdr、enob 以及信号/噪声/失真功率、选中的
fundamental/harmonic bins 等元数据。thd 和 sfdr 是 dBc，enob 使用
(sndr - 1.76) / 6.02。计算会先去除记录均值，再进行加窗；Rect 窗要求整数
bin 并只计基波单 bin。周期 Hann 允许浮点 bin，默认积分基波和谐波的 ±2
bin 主瓣，SFDR 也按最强 spur 的主瓣积分（Rect 仍为单 bin）。单边谱会保留
Nyquist bin 的原始能量，不会把 DC 算入噪声。谐波折叠到 [0, fs/2] 后去重；
基波和谐波测量带重叠时明确报错。记录少于 16 点、fund 在 DC/Nyquist、零
AC 能量或常量波形也报错。

非相干 Hann 测量中，主瓣外的旁瓣残留仍计入噪声，因此窗口本身可能限制
可测 SNDR。高分辨率验证优先使用相干采样；改变记录长度或 `bin_width` 后
检查结果稳定性，不能把任意 Hann 结果当成 ADC 的真实噪声底。

表达式环境中 fft_metrics 直接映射到 spectrum，所以可选取字段：

~~~yaml
metrics:
  sndr: "sndr_fft(V('adc_out'), 1G, 7, 256, t0=0, window='hann')"
  # 或者先采样，再使用返回字典
  enob: "fft_metrics(sample(V('adc_out'), 1G, 256, t0=0), 1G, 7)['enob']"
~~~

理想闭式波形可能使 snr/sfdr 为 inf；目标指标仍必须有真实噪声或其他
有限误差，使最终结果可被 compute_metrics 接受。FFT 指标只描述所选窗口、
采样率、记录长度和带宽，不能直接等同于完整 OTA 的增益带宽或稳定性。

## 4. ADC 静态和 bit 解码

静态测量明确区分两种输入：

~~~python
adc_static(codes, bits, stimulus="ramp", method="endpoint")
code_density(codes, bits, stimulus="ramp")
transition_metrics(transitions, bits, *, ideal_step=1.0, ideal_start=0.0)
decode(bits, threshold=0.5, msb_first=True)
~~~

adc_static(..., method="histogram") 或 code_density 把 codes 当作完整均匀
ramp 的 ADC 输出码，返回 counts、dnl、inl、dnl_peak、inl_peak、
missing_codes。DNL 和 INL 以 LSB 为单位；直方图模式不能从没有已知激励
范围的样本推断绝对 offset/gain，因此这两个字段为 None。样本数至少为
2**bits，必须覆盖 code 0 和最大 code；端点数量相对内部中位数过高时会拒绝，
以避免端点饱和把采样噪声当作线性度。缺少内部码会列在 missing_codes，但
未覆盖端点或样本量不足是测量范围/激励错误，应先修正 ramp。

adc_static 默认的 method="endpoint" 和 transition_metrics 只接受完整的
2**bits + 1 个转换边界（含两端外边界）。边界允许非递减；零宽度才表示
missing code，某个 code 宽于一个 LSB 本身不再额外计为 missing code。端点
模式使用测量范围定义 LSB：LSB = (last - first) / 2**bits，DNL 是
width / LSB - 1，INL 是相对于连接两端点直线的归一化边界偏差，inl 数组
包含全部 2**bits + 1 个边界。这个归一化结果不从端点数组推导绝对 offset/gain；
若请求单独的参考 offset/gain 误差，ideal_step 和 ideal_start 只用于该参考。
内部阈值数组不会被自动外推成伪全码结果。

decode 接收形状为 (samples, bits) 的二维 bit 波形，每一列长度相同；
msb_first=True 时第一列是 MSB，返回无符号整数码。波形必须有限，ragged
数组、空列和非法阈值拒绝。表达式中可用
decode_bits(np.column_stack((V('b2'), V('b1'), V('b0'))), threshold=.5)；
再以 sample(..., method='previous') 在声明的有效采样时刻提取整数码，才用于
adc_static(..., method="histogram")。不能直接统计自适应瞬态时间点；这些点的
时间间隔不同，会扭曲码密度。输入必须来自覆盖完整范围的均匀 ramp。

## 5. 可复用指标例子

下面的例子只依赖波形和配置，不假设某个 PDK。

### ADC 半 LSB 建立时间

~~~yaml
metrics:
  adc_settle: >-
    settle_time(V('adc_in'), start=0, end=2u, final=0.9,
                atol=0.5*0.9/2**12, on_unsettled='raise')
  adc_settled: >-
    settled(V('adc_in'), start=0, end=2u, final=0.9,
            atol=0.5*0.9/2**12)
constraints:
  - {metric: adc_settled, min: 1}
~~~

adc_settle 的单位是秒，adc_settled 是 0/1。实际 ADC 的 LSB、参考范围、
采样保持时间和输入噪声必须一起定义；若噪声本身超过半 LSB，应复核指定带宽内
的噪声与误差预算，不能把未建立波形标记为通过。

### DC 增益和 offset

~~~yaml
metrics:
  gain_vv: {analysis: dc, expr: "dc_gain(V('out'), V('in'))"}
  offset_v: {analysis: dc, expr: "dc_offset(V('out'), V('in'))"}
~~~

gain_vv 是 V/V；若需要 dB，写成
20*log10(abs(m['gain_vv']))，并把它作为依赖指标或直接表达式。

### FFT 动态指标

~~~yaml
metrics:
  sndr_db: "sndr_fft(V('adc_out'), 1G, 7, 256, t0=0)"
  enob: "enob(V('adc_out'), 1G, 7, 256, t0=0)"
~~~

### DNL/INL

以下是独立的完整均匀 ramp 测试；时钟为 1 MHz，示例采样相位为 10 μs。
采样窗口必须位于所有 bit 都有效的转换输出中，并保证均匀覆盖全码范围。

~~~yaml
metrics:
  dnl_peak: >-
    adc_static(sample(decode_bits(np.column_stack((V('b2'), V('b1'), V('b0')))),
                      fs=1M, nsamp=4096, t0=10u, method='previous'),
               3, method='histogram')['dnl_peak']
  inl_peak: >-
    adc_static(sample(decode_bits(np.column_stack((V('b2'), V('b1'), V('b0')))),
                      fs=1M, nsamp=4096, t0=10u, method='previous'),
               3, method='histogram')['inl_peak']
~~~

至少一个样本/码只是最低输入检查，不保证统计精度；应增加采样数量并验证结果
稳定。missing_codes 返回未观察到的码，激励和样本量不足时不能据此断言硬件缺码。

### 功率、能量和 ADC FoM

以下另用已建立的单音动态测试；示例 fs=1 GHz、测量带宽 BW=500 MHz，
输出波形和功耗在同一观察窗口测量。按实际测试替换窗口、频点、fs 和 BW。

~~~yaml
metrics:
  sndr_db: "sndr_fft(V('adc_out'), 1G, 7, 256, t0=10u)"
  enob_bits: "(m['sndr_db'] - 1.76) / 6.02"
  power_w: "avg(-V('vdd') * I('VDD'), 10u, 10.256u)"
  energy_j: "integ(-V('vdd') * I('VDD'), 10u, 10.001u)"
  walden_fj: "1e15 * m['power_w'] / (2**m['enob_bits'] * 1G)"
  schreier_db: "m['sndr_db'] + 10*log10(500M / m['power_w'])"
objective: {metric: walden_fj, goal: minimize}
constraints: [{metric: sndr_db, min: 60}]
~~~

power_w 为 W，energy_j 为每次转换的 J；电流符号和正功耗由实际 testbench
确认。Walden FoM 的单位为 fJ/conversion-step，越低越好；Schreier FoM
越高越好。过采样 ADC 的实际信号带宽不能直接用 fs/2 代替。

## 6. 验证边界和参考

功能测试使用闭式波形、合成 ADC 码和明确的端点/噪声限制，验证 API、单位
和失败条件。另用手写的理想 RC 分压夹具做了本地 Spectre 23.1 的 tran+DC
验证：R1=R2=1 kΩ、C=1 nF，增益 0.5 V/V、偏移 0 V；1% 建立时间
2.303019 μs（maxstep=10 ns）和 2.303071 μs（5 ns），解析值 2.303085 μs。
两个步长均为 0 errors、0 warnings；步长减半的建立时间变化约 0.0023%。
这是通用测量后端验证；ADC 指标以合成数据验证，真实 ADC 仍需在目标模型、
负载、参考源、采样时序和 PVT 条件下复核。

[RC 参考配置](../examples/metrics_rc/rc.yaml) 可在 GUI 中直接打开；不需要 PDK。
它使用 DC 增益作为瞬态最终电平，演示不同分析间的指标依赖。名义参数下的
预期值见上文；调整 R1 后最终电平和时间常数随之改变。也可用
`./vcal check examples/metrics_rc/rc.yaml --json` 做无仿真预检。

- [Analog Devices MT-003](https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf)：SINAD、SNR、THD、SFDR 和 ENOB 的 FFT 定义。
- [Analog Devices MT-010](https://www.analog.com/media/en/training-seminars/tutorials/MT-010.pdf)：ADC 静态指标、LSB、DNL 和 INL 的背景。
- [Advanced Integrated Circuits: Oversampling and Sigma-Delta ADCs](https://analogicus.com/aic2024/2024/02/16/Lecture-6-Oversampling-and-Sigma-Delta-ADCs.html)：Walden 和 Schreier FoM 定义。
- [Texas Instruments SLYT262A](https://www.ti.com/lit/an/slyt262a/slyt262a.pdf)：以 ADC LSB 误差带定义建立时间，并说明采样驱动测量条件。
- [Texas Instruments SBAA535](https://www.ti.com/lit/an/sbaa535/sbaa535.pdf)：半 LSB 模拟建立时间和噪声/参考范围限制的工程注意事项。

这些资料用于定义术语和测量边界；vcal 的数值实现独立编写。
