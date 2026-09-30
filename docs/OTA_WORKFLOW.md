# 通用电路指标工作流（OTA／gm/Id 为可选方向）

当前优先级是把通用电路测量做成可复用、可校验的接口。OTA 的 gm/Id 初值
和 scope 仍可使用，但它们是可选的设计辅助；增益、建立时间、ADC、功耗等
指标都通过同一套 Circuit metrics 配置进入目标和约束。

完整函数签名、单位和失败条件见 [docs/METRICS.md](METRICS.md)。

## Circuit metrics 页面

GUI 的 **Circuit metrics** 页面提供四个连续步骤：

1. 选择要使用的分析。可以只配置 tran、只配置 dc，也可以同时配置。tran
   编辑 stop 和 maxstep；dc 编辑 source、start、stop、step。dc 扫描当前只
   支持 Spectre，source 必须是网表中的 source instance。
2. 在指标表中添加或编辑表达式。模板覆盖建立时间、DC gain、FFT、DNL/INL、
   功率和能量；每个指标可选择 tran 或 dc。旧的字符串值继续使用默认分析。
3. 运行语法、依赖和分析预检，通过后应用配置。依赖 m["name"] 会自动排序；
   未知引用、循环、未配置分析和非法表达式在仿真前报告。
4. 在目标和约束页面选择已验证的指标。最终每个指标必须是有限实标量；
   字典和数组要先用字段索引或归约，理想 FFT 的 inf 不能直接作为优化目标。

## YAML 最小结构

~~~yaml
simulator: spectre
stimuli: my_tb.scs
netlist: my_dut.scs

tran: {stop: 200n, maxstep: 50p}
dc: {source: VBIAS, start: 0, stop: 1.8, step: 10m}

save: [v(out), v(in), v(adc_in), i(VDD)]

metrics:
  settle_half_lsb:
    analysis: tran
    expr: "settle_time(V('adc_in'), start=0, end=200n, final=0.9)"
  gain_vv:
    analysis: dc
    expr: "dc_gain(V('out'), V('in'))"
  sndr_db:
    analysis: tran
    expr: "sndr_fft(V('adc_in'), 1G, 7, 256, t0=0)"

objective: {metric: gain_vv, goal: maximize}
constraints:
  - {metric: settle_half_lsb, max: 20n}
~~~

tran 和 dc 可以单独存在；两者都存在时，旧字符串指标默认使用 tran。
需要明确归属时使用 {analysis: dc, expr: "..."} 或 {analysis: tran, expr: "..."}。
dc 配置只接受 source/start/stop/step，步长必须朝向终点且包含至少两个点。
simulator: hspice 时不能使用 dc。

## 指标选择

### 建立时间和 transient

settle_time 的默认 on_unsettled="raise" 会把窗口内未建立视为失败。
如果明确使用 on_unsettled="window" 保留窗口长度，必须同时把
settled(...) == 1 加入约束。未给 atol 时使用 tol*abs(final-initial)；显式
atol 会覆盖 tol。ADC 半 LSB 可写成 atol=0.5*Vref/2**bits，输出单位是秒。
rise_time、fall_time、
delay、slew_rate、cross、overshoot、settling_error 适合组合成建立、
过冲和速度指标。

### DC gain

dc_gain(y, x) 对输入轴作最小二乘斜率，单位是输出/输入，例如 V/V；
dc_gain(y, x, at=...) 返回指定点所在采样段的局部斜率。
dc_offset(y, x) 返回拟合截距。若 x 省略，DC 分析轴会作为输入。

### FFT

enob、sndr_fft、snr_fft、thd_fft 和 sfdr_fft 固定采样起点时不自动
寻找最佳相位。fund 是 FFT bin，不是 Hz。旧式显式 t0_lo/t0_hi 仍可扫描
相位，并发出兼容性警告；需要保守约束时传 phase_mode="worst"。直接调用
fft_metrics 等价于调用 spectrum(samples, fs, fund, window="rect",
harmonics=5, bin_width=None)，返回字典后再选 ['sndr']、['enob'] 等标量。

### ADC 静态

adc_static(..., method="histogram") 和 code_density 只用于完整均匀
ramp code-density；必须有足够样本并覆盖 code 0 和最大 code，端点饱和会被
拒绝。任意瞬态码的直方图不能当作 INL。adc_transitions 映射完整的
transition_metrics 边界数组，要求 2**bits + 1 个非递减边界；零宽度才是
missing code，宽 code 不自动算缺码。端点模式以两端点定义 LSB，并把 INL
归一化到端点连线，不会自动合成未测的外端点。decode_bits 映射 decode，bit 波形可用
np.column_stack 组成 (samples, bits) 矩阵。

## 可复用例子

~~~yaml
metrics:
  # 12-bit ADC 的半 LSB 建立时间；settled 是必须的通过条件
  adc_settle: >-
    settle_time(V('adc_in'), start=0, end=2u, final=0.9,
                atol=0.5*0.9/2**12)
  adc_ok: >-
    settled(V('adc_in'), start=0, end=2u, final=0.9,
            atol=0.5*0.9/2**12)

  gain_vv: {analysis: dc, expr: "dc_gain(V('out'), V('in'))"}
  gain_db: "20*log10(abs(m['gain_vv']))"

  sndr_db: "sndr_fft(V('adc_in'), 1G, 7, 256, t0=0)"
  enob: "enob(V('adc_in'), 1G, 7, 256, t0=0)"

  power_w: "avg(-V('vdd') * I('VDD'), 0, 2u)"
  energy_j: "integ(-V('vdd') * I('VDD'), 0, 2u)"
  fom: >-
    m['sndr_db'] + m['gain_db'] - 10*log10(m['energy_j']/1p)

constraints:
  - {metric: adc_ok, min: 1}
~~~

DNL/INL 的一个表达式形态是：

~~~yaml
metrics:
  dnl_peak: >-
    adc_static(decode_bits(np.column_stack((V('b2'), V('b1'), V('b0')))),
               3, method='histogram')['dnl_peak']
  inl_peak: >-
    adc_static(decode_bits(np.column_stack((V('b2'), V('b1'), V('b0')))),
               3, method='histogram')['inl_peak']
~~~

这里的 bit 波形必须来自均匀 ramp 的采样，并满足端点覆盖和噪声限制；若只是
任意 transient code 序列，表达式应被拒绝或改为专门的逻辑波形统计，而不能
声称得到了 ADC INL。

功耗和能量的单位分别是 W 和 J。电压电流的方向由 testbench 决定，常见的
供电消耗写作 -V('vdd') * I('VDD')；请先用闭式或合成波形检查符号和范围。

## 校验、可复现性和验证边界

配置校验会检查分析块、表达式 AST、工程后缀、m 依赖和 analysis 归属；
运行时还检查窗口、波形有限性和最终实标量。测量签名包含指标定义、分析配置
和保存信号；签名变化或没有签名的旧结果会跳过，避免混用不同公式。

闭式 transient/DC 波形、合成 ADC 码、端点覆盖、采样噪声和 FFT 解析比值可
验证函数行为。它们不代表某个真实 PDK、OTA 或 Spectre 电路已经通过。真实
设计仍需在目标模型、负载、参考、采样时序和 PVT 下复核。术语和测量边界参考
[ADI MT-003](https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf)、
[ADI MT-010](https://www.analog.com/media/en/training-seminars/tutorials/MT-010.pdf)、
[TI SLYT262A](https://www.ti.com/lit/an/slyt262a/slyt262a.pdf) 和
[TI SBAA535](https://www.ti.com/lit/an/sbaa535/sbaa535.pdf)。

## 可选的 gm/Id 和 scope 辅助

已有的 gm/Id LUT 助手仍可读 NPZ/CSV，并把尺寸建议作为优化初值；scopes
可以让 CLI/GUI 先检查一个 sub-cell 范围。它们不改变 Circuit metrics 的
分析和有限标量规则，也不自动生成 PDK LUT、OTA AC testbench 或新的激励。

合成演示可以离线查询：

~~~bash
./vcal gmid examples/gmid_demo.csv --length 180n --vds 0.6 --vsb 0 --gmid 15 --id 20u --json
~~~

真实器件尺寸、短沟道效应、匹配、nf/m 和版图规则仍需相应 PDK 的仿真确认。

## 远期方向（可选）

在通用测量稳定后，可以考虑本地 Spectre OP/DC LUT 生成、OTA AC/稳定性测试
平台、噪声和 PVT 角色映射，以及按功能组的设计变量筛选。这些是后续方向，
当前优先级仍是让 tran、dc、FFT、ADC 静态和功耗/能量指标在明确边界下可复用。
