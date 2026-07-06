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
./vcal serve-gpu --port 8494          # on a GPU box, to lend its GPU
```

CLI options: `--budget N` total evals, `--batch q` parallel proposals,
`--device auto|cuda|cpu`, `--remote-gpu URL`, `--objective '<yaml>'`
(override), `--resume` (warm-start from history).

## Define a circuit

A circuit is one YAML file (see [`examples/demo_ota.yaml`](examples/demo_ota.yaml)):

| Field | Meaning |
|-------|---------|
| `pdk` | name of a `pdks/<name>.yaml` (no PDK path in the circuit file) |
| `netlist` / `stimuli` | your device netlist + the excitation deck |
| `params` | optimization variables: `devices`, `attr` (`w`/`l`/`mr`/`value`), `lo`/`hi`/`log`/`integer`. One variable can drive several devices (symmetric pairs share one). |
| `metrics` | named waveform expressions; supports eng-notation (`90n`) and referencing earlier metrics via `m['...']` |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT list: `{name, temp, pdk_corner, tb_extra}` + `corner_worst` reduction |

Editing a MOS `w` auto-scales the layout-derived params (as/ad/ps/pd/…),
same semantics as editing the schematic.

## PDK configuration

PDKs are defined in `pdks/<name>.yaml` — see
[`pdks/example.yaml`](pdks/example.yaml). A PDK file lists the model
library (use `${PDK_ROOT}` so no absolute path is committed) and, per
process corner, the `.lib` sections to include. **Swapping PDK is a
one-line `pdk:` change**; circuit configs never contain a PDK path.

## Layout

```
vcal                    unified launcher (gui / run / serve-gpu / install)
server/optserver/       netlist rewrite + simulate + metric extraction
client/optclient/       BoTorch optimizer + remote-GPU compute server
local/optlocal/         offline engine + PyQt5 GUI + schematic write-back
pdks/                   PDK definitions (example.yaml is the template)
examples/               sanitized example circuit config
```

## License

See [LICENSE](LICENSE).
