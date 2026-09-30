# 兼容的 FFT 与 ADC 数值接口

现有 YAML 的这些函数继续可用；本次开发范围是内部模块测量。通用窗口规则见 [指标指南](METRICS.md)。

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


兼容性回归覆盖这些 API；当前界面模板优先展示内部模块指标。
