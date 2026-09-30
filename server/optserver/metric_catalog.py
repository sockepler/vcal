"""Editable measurement examples for the desktop interface.

Examples declare measurement conditions, not performance specifications. The
user supplies signal names, windows, target levels and sampling conditions.
"""

CATALOG = [
    ("settling_s", "Settling time", "tran", "s",
     "settle_time(V('out'), final=1, tol=.01, hold=1u)",
     "Time to enter and remain within 1% of the step. Set the input edge, final level and observation window."),
    ("settling_abs_s", "Settling within absolute error", "tran", "s",
     "settle_time(V('out'), final=1, atol=1m, hold=1u)",
     "Time to enter and remain within an absolute ±1 mV band. Adjust the error band and final level."),
    ("settled_ok", "Settled flag", "tran", "0/1",
     "settled(V('out'), final=1, tol=.01, hold=1u)",
     "Use a minimum constraint of 1 when a finite capped settling-time objective is selected."),
    ("settling_error_v", "Settling error", "tran", "V",
     "settling_error(V('out'), final=1)",
     "Largest absolute error in the chosen observation window. Useful as a smooth optimization target."),
    ("rise_s", "Rise time", "tran", "s",
     "rise_time(V('out'), initial=0, final=1)",
     "10% to 90% of the specified rising transition; interpolate threshold crossings."),
    ("fall_s", "Fall time", "tran", "s",
     "fall_time(V('out'), initial=1, final=0)",
     "90% to 10% of the specified falling transition; select one transition window."),
    ("delay_s", "Propagation delay", "tran", "s",
     "delay(V('in'), V('out'), level_x=.5, level_y=.5)",
     "Delay from the selected input edge to its next output response, before the next input edge. Set actual thresholds and directions."),
    ("slew_v_s", "Slew rate", "tran", "V/s",
     "slew_rate(V('out'), initial=0, final=1)",
     "Average absolute slope over the 10–90% transition interval."),
    ("overshoot_ratio", "Overshoot", "tran", "V/V",
     "overshoot(V('out'), initial=0, final=1)",
     "Peak beyond the final level, divided by step amplitude; 0.05 means 5%."),
    ("dc_gain_v_v", "DC gain", "dc", "V/V",
     "dc_gain(V('out'), V('in'), at=0)",
     "Signed local slope at the specified input voltage. Omit at for a fit over the chosen range."),
    ("dc_gain_db", "DC gain in dB", "dc", "dB",
     "db(dc_gain(V('out'), V('in'), at=0))",
     "20 log10 of the magnitude of DC transfer slope; requires a configured DC sweep."),
    ("dc_offset_v", "DC output offset", "dc", "V",
     "dc_offset(V('out'), V('in'))",
     "Intercept of the fitted output-versus-input transfer line; not input-referred offset."),
    ("ripple_vpp", "Peak-to-peak ripple", "tran", "V",
     "pp(V('out'))",
     "Maximum minus minimum in the selected settled window."),
    ("noise_rms_v", "RMS deviation", "tran", "V",
     "std(V('out'))",
     "Time-weighted RMS after removing the mean. A deterministic trace does not include device noise automatically."),
    ("power_w", "Average supply power", "tran", "W",
     "avg(-V('vdd') * I('VDD'))",
     "Supply voltage times delivered source current. Verify the simulator current sign convention."),
    ("energy_j", "Energy per operation", "tran", "J",
     "integ(-V('vdd') * I('VDD'))",
     "Integrate delivered supply power over one declared operation window."),
    ("pulse_width_s", "Pulse width", "tran", "s",
     "pulse_width(V('out'), level=.5)",
     "High time between a rising and falling threshold crossing; only complete pulses inside the window count."),
    ("period_s", "Clock period", "tran", "s",
     "period(V('out'), level=.5)",
     "Mean interval between like edges inside the window; at least two edges are required."),
    ("frequency_hz", "Clock frequency", "tran", "Hz",
     "frequency(V('out'), level=.5)",
     "Reciprocal of the mean period, measured from complete edge intervals."),
    ("duty_ratio", "Duty cycle", "tran", "0–1",
     "duty_cycle(V('out'), level=.5)",
     "Total high time divided by total duration of complete rising-to-rising cycles; 0.5 means 50%."),
    ("period_jitter_s", "Period jitter", "tran", "s RMS",
     "period_jitter(V('out'), level=.5)",
     "Population standard deviation of measured periods; at least three edges. Includes only variation present in the trace."),
    ("cycle_jitter_s", "Cycle-to-cycle jitter", "tran", "s RMS",
     "cycle_jitter(V('out'), level=.5)",
     "RMS difference between adjacent measured periods; at least three edges. This is not a phase-noise analysis."),
    ("droop_v_s", "Hold droop rate", "tran", "V/s",
     "slope(V('out'))",
     "Signed time-weighted fitted slope in the hold window. Use abs(...) to minimize droop magnitude."),
    ("trigger_value_v", "Input at output crossing", "tran", "V",
     "value_at_cross(V('in'), V('out'), level=.5)",
     "Input voltage at the selected output edge. For a comparator, declare the input ramp, output threshold and edge direction."),
    ("tracking_rms_v", "RMS tracking error", "tran", "V",
     "rms(V('out') - V('in'))",
     "Time-weighted RMS output-minus-input error in the selected tracking window."),
    ("tracking_peak_v", "Peak tracking error", "tran", "V",
     "vmax(abs(V('out') - V('in')))",
     "Largest absolute output-minus-input error in the selected window."),
    ("peak_current_a", "Peak supply current", "tran", "A",
     "vmax(abs(I('VDD')))",
     "Largest absolute supply current in the selected operation window."),
    ("charge_c", "Delivered charge", "tran", "C",
     "integ(-I('VDD'))",
     "Signed delivered source charge in the selected window. Verify the simulator current sign convention."),
]


def metric_catalog():
    items = [dict(zip(("name", "title", "analysis", "unit", "expr", "description"), row))
             for row in CATALOG]
    for item in items:
        if item["analysis"] == "tran":
            start, end = "1u", "10u"
            if item["name"] in {"settling_error_v", "ripple_vpp", "noise_rms_v", "droop_v_s"}:
                start = "9u"
            if item["name"] in {"energy_j", "charge_c"}:
                end = "2u"
            item["window"] = {"start": start, "end": end}
    return items
