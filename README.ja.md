# vcal — Virtuoso ネットリスト向け ばらつき考慮キャリブレーション

**[English](README.en.md) · [中文](README.md) · [日本語]**

**深層カーネル・ガウス過程ベイズ最適化**による、アナログ/ミックスド
シグナル IC のトランジスタ**サイズ・バイアス自動調整**ツール。Virtuoso
から出力したネットリストを指定し、「良い」とは何かを宣言すると、素子
サイズ空間を探索して目標に近づけ、最適解を回路図へ書き戻します。

完全オフライン：シミュレーションも最適化もローカルマシンで実行、
クラウド・ネットワーク不要。

## 特長

- **自由な目的関数** — 最大化・最小化・**目標値への接近**（例「利得 → 8倍」）。
  複数指標の重み付き和も可能。
- **任意の指標** — 保存ノードの波形式：`avg`・`settle_time`・
  `overshoot`・コヒーレント FFT の `enob` など。
- **PVT コーナー** — 温度/電圧/プロセスコーナーを定義すると、最適化は
  自動的に**最悪ケース**を対象にします。
- **ローカル GPU 自動検出** — NVIDIA(CUDA)/AMD(ROCm)、無ければ CPU。
  **リモート GPU への計算オフロード**も可能。
- **回路図への書き戻し** — virtuoso-bridge 経由で最適サイズを反映
  （プレビュー＋確認）。
- **PDK 非依存** — PDK 切替は 1 行。回路設定に PDK パスは現れません。
- **GUI または CLI** — gmid-tool 風のデスクトップ GUI、あるいは CLI。

## アルゴリズム

Sobol 初期化 → Matérn GP（データが増えると **深層カーネル GP**：MLP
特徴抽出器 + GP を GPU 上で同時学習）→ **TuRBO** 信頼領域内でバッチ制約
付き **qLogNEI** 獲得関数を最適化。収束しない点は悲観的に補完し、代理
モデルが自動的に回避を学習します。

> 注：典型的なサイジング問題（数十〜数百点・10〜20 次元）では GP/DKL は
> 小行列演算で **CPU が最速**の場合が多く、GPU が有利になるのは大規模
> 問題のみ。`auto` は CPU を既定にします。

## インストール

```bash
./vcal install            # ハードウェア自動検出 (cuda / rocm / cpu)
export PDK_ROOT=/path/to/your/pdk     # pdks/*.yaml が参照
```

Python ≥ 3.10 と SPICE シミュレータ(HSpice/Spectre)が PATH に必要。
インストーラがハード別 venv(`local/venv-<種別>`)に torch+botorch+PyQt5
を用意します。

## 使い方

```bash
./vcal gui                            # GUI
./vcal run examples/demo_ota.yaml --budget 100 --batch 4
./vcal serve-gpu --port 8494          # GPU マシンで GPU を貸し出す
```

主なオプション：`--budget N`(総評価数)、`--batch q`(並列提案数)、
`--device auto|cuda|cpu`、`--remote-gpu URL`、`--resume`(履歴から再開)。

## 回路の定義

回路は 1 つの YAML ファイル（[`examples/demo_ota.yaml`](examples/demo_ota.yaml) 参照）:

| 項目 | 意味 |
|------|------|
| `pdk` | `pdks/<name>.yaml` の名前（回路ファイルに PDK パスを書かない） |
| `netlist` / `stimuli` | 素子ネットリスト + 励起デック |
| `params` | 最適化変数：`devices`・`attr`(`w`/`l`/`mr`/`value`)・`lo`/`hi`/`log`/`integer`。1 変数で複数素子を駆動可（対称ペアで共有） |
| `metrics` | 名前付き波形式。工学表記(`90n`)、`m['...']` で前の指標を参照可 |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT リスト：`{name, temp, pdk_corner, tb_extra}` + `corner_worst` 集約 |

MOS の `w` を変更するとレイアウト由来パラメータ(as/ad/ps/pd…)も自動
スケール。回路図編集と同じ意味論です。

## PDK 設定

PDK は `pdks/<name>.yaml` で定義します（[`pdks/example.yaml`](pdks/example.yaml)
参照）。モデルライブラリ（`${PDK_ROOT}` を使い絶対パスを非公開に）と、
プロセスコーナーごとに include する `.lib` セクションを列挙します。
**PDK 切替は `pdk:` の 1 行**。回路設定に PDK パスは現れません。

## 構成

```
vcal                    統合ランチャ (gui / run / serve-gpu / install)
server/optserver/       ネットリスト書換 + シミュレーション + 指標抽出
client/optclient/       BoTorch 最適化器 + リモート GPU 計算サーバ
local/optlocal/         オフラインエンジン + PyQt5 GUI + 回路図書戻し
pdks/                   PDK 定義（example.yaml がテンプレート）
examples/               サニタイズ済みサンプル回路設定
```

## ライセンス

[LICENSE](LICENSE) を参照。
