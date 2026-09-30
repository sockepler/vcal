# vcal — Variation-aware Calibration for Virtuoso netlists

**[English] · [中文](README.md) · [日本語](README.ja.md)**

Automatic transistor **sizing and bias tuning** for analog / mixed-signal
ICs, driven by **deep-kernel Gaussian-process Bayesian optimization**.
Point it at a Virtuoso-exported netlist, declare what "good" means, and it
searches the device-size space to get there — then writes the winning
sizes back to the schematic.

Fully offline: simulations and the optimizer both run on your own machine;
no cloud, no network required.

## Features

- **Custom objectives** — maximize, minimize, or *approach a target value*
  (e.g. "gain → 8×"), as a weighted sum of several metrics.
- **Arbitrary metrics** — waveform expressions over saved nodes: `avg`,
  `settle_time`, `overshoot`, coherent-FFT `enob`, … or your own.
- **PVT corners** — define temperature / voltage / process corners; the
  optimizer targets the **worst case** automatically.
- **Local GPU, auto-detected** — NVIDIA (CUDA) or AMD (ROCm); falls back to
  CPU. Optionally **offload the GP math to a remote GPU box**.
- **Schematic write-back** — push optimized sizes to the Virtuoso schematic
  via virtuoso-bridge (preview + confirm).
- **PDK-agnostic** — swap PDK with a single line; no PDK path in configs.
- **GUI or CLI** — a gmid-tool-style desktop GUI, or scriptable CLI.

## Algorithm

Sobol initialization → Matérn GP (switching to a **deep-kernel GP**: MLP
feature extractor + GP, jointly trained on the GPU) once enough data
exists → batched constrained **qLogNEI** acquisition optimized inside a
**TuRBO** trust region. Failed / non-converging simulations are imputed
pessimistically so the surrogate learns to avoid them.

> Note: on typical sizing problems (tens–hundreds of points, ~10-20 dims)
> the GP/DKL math is small-matrix work and **CPU is usually fastest** — the
> GPU only wins on much larger problems. `auto` defaults to CPU.

## Install

```bash
./vcal install            # auto-detect hardware (cuda / rocm / cpu)
# or explicitly:  ./vcal install cuda | rocm | cpu

export PDK_ROOT=/path/to/your/pdk     # referenced by pdks/*.yaml
```

Requires Python ≥ 3.10 and a SPICE simulator (HSpice / Spectre) on PATH.
The installer creates a per-hardware virtualenv (`local/venv-<flavor>`)
with torch + botorch + PyQt5.

## Usage

```bash
./vcal gui                            # desktop GUI
./vcal run examples/demo_ota.yaml --budget 100 --batch 4
./vcal check examples/demo_ota.yaml         # validate only; no simulation
./vcal check examples/demo_ota.yaml --json  # machine-readable spec
./vcal serve-gpu --port 8494          # on a GPU box, to lend its GPU
```

CLI options: `--budget N` total evals (never more than N), `--batch q` parallel proposals,
`--device auto|cuda|rocm|cpu`, `--remote-gpu URL`, `--objective '<yaml>'`
(override), and `--resume` (the default, warm-start from history) or
`--no-resume` (ignore history).

`check` validates configuration structure, referenced files, devices, parameter
ranges, and metric references. It does not validate waveform-expression syntax
or run a real simulation.

## OTA workflow: gm/Id, scopes, and review

See [`docs/OTA_WORKFLOW.md`](docs/OTA_WORKFLOW.md) for the complete workflow. The
GUI gm/Id panel loads an existing NPZ/CSV LUT; the synthetic table in this
repository is only a flow demo and contains no PDK data:

```bash
./vcal gmid examples/gmid_demo.csv \
  --length 180n --vds 0.6 --vsb 0 --gmid 15 --id 20u --json
```

The CLI can inspect parameter groups, run one sub-cell scope, and review a
history offline:

```bash
./vcal scopes <circuit.yaml> --json
./vcal run <circuit.yaml> --scope <scope-name> \
  --stagnation-rounds 6 --initial-points points.json --budget 100
./vcal review <history.jsonl> --config <circuit.yaml> --json
```

Without `--scope`, all enabled parameters are optimized. With a scope selected,
only parameters whose every `devices` entry belongs to that scope remain active;
shared and unselected-scope parameters use configured fixed/nominal values in
the CLI, or the initial-value column in the GUI.
`--initial-points` takes a JSON list of physical SI-value mappings, and
`--stagnation-rounds` is the non-negative number of stagnant rounds before global
exploration is queued. The GUI path is: open the gm/Id LUT → map width/length
parameters → calculate and apply initial values → select a scope → start the
optimization with the existing configuration → review the history with `review`.

In `params.devices`, `scope/instance` identifies a device inside a subcircuit
master; changing that master affects all of its instances. Scope selection does
not create new stimuli or a new testbench: simulations still use the configured
netlist and existing testbench. This version does not automatically generate a
PDK LUT or an OTA AC testbench; the synthetic LUT and examples here do not claim
real Spectre testing or performance results.

### Interface language

The language drop-down at the top of the GUI switches between Chinese, Japanese,
and English immediately, then saves the selection as the preference for the next
launch. The CLI accepts the option before or after its subcommand, for example
`./vcal --lang ja gui` or `./vcal gui --lang en`; you can also use
`VCAL_LANG=ja ./vcal gui`. The priority is explicit `--lang`, `VCAL_LANG`, the
saved GUI preference, and Chinese as the final fallback. Translation applies to
the interface and CLI messages only; third-party logs such as Spectre output,
user-defined metric names, and netlist identifiers remain unchanged.
Top-level help and `install --help` are generated by the shell and use `--lang`
or `VCAL_LANG`; they do not read the saved GUI preference.

## GUI iteration chart

The GUI chart uses evaluation count on the x-axis and a drop-down containing all
configured `metrics` for the y-axis. Blue points are feasible, orange points
violate constraints, and red crosses are failed simulations; failures have no
fabricated y-value. The best-feasible trend is shown only for the objective
metric. Click a point to inspect its parameters, metrics, and error, and use
zoom, pan, PNG export, or CSV export as needed.

GUI results go by default to `../work/<name>/gui_run/` beside the YAML; CLI runs
use the same circuit's `cli_run/` directory. The directory receives
`best.json` (only when a successful evaluation exists, with `feasible` recording
whether constraints are met), `summary.json`, `state.json`, and `history.csv`.
`vcal run --resume` restores existing records and avoids simulating duplicate
parameter points.

## Define a circuit

A circuit is one YAML file (see [`examples/demo_ota.yaml`](examples/demo_ota.yaml)):

| Field | Meaning |
|-------|---------|
| `pdk` | name of a `pdks/<name>.yaml` (no PDK path in the circuit file) |
| `netlist` / `stimuli` | your device netlist + the excitation deck |
| `params` | optimization variables: `devices`, `attr` (`w`/`l`/`mr`/`value`), `lo`/`hi`/`log`/`integer`; set `enabled: false` to disable one. One variable can drive several devices (symmetric pairs share one). |
| `metrics` | named waveform expressions; supports eng-notation (`90n`) and referencing earlier metrics via `m['...']` |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT list: `{name, temp, pdk_corner, tb_extra}` + `corner_worst` reduction |

Editing a MOS `w` auto-scales the layout-derived params (as/ad/ps/pd/…),
same semantics as editing the schematic.
Top-level `fixed` can pin parameters with `{parameter_name: value}`. Relative
paths in the config are resolved relative to the YAML first, while legacy
repository-root-relative paths remain compatible.

## PDK configuration

PDKs are defined in `pdks/<name>.yaml` — see
[`pdks/example.yaml`](pdks/example.yaml). A PDK file lists the model
library (use `${PDK_ROOT}` so no absolute path is committed) and, per
process corner, the `.lib` sections to include. **Swapping PDK is a
one-line `pdk:` change**; circuit configs never contain a PDK path.

## Layout

```
vcal                    unified launcher (gui / run / check / serve-gpu / install)
server/optserver/       netlist rewrite + simulate + metric extraction
client/optclient/       BoTorch optimizer + remote-GPU compute server
local/optlocal/         offline engine + PyQt5 GUI + schematic write-back
pdks/                   PDK definitions (example.yaml is the template)
examples/               sanitized example circuit config
```

## License

See [LICENSE](LICENSE).
