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
./vcal check examples/demo_ota.yaml         # 設定のみ検証（シミュレーションなし）
./vcal check examples/demo_ota.yaml --json  # 機械可読な spec
./vcal serve-gpu --port 8494          # GPU マシンで GPU を貸し出す
```

主なオプション：`--budget N`(総評価数、N を超えない)、`--batch q`(並列提案数)、
`--device auto|cuda|rocm|cpu`、`--remote-gpu URL`、`--resume`(既定で履歴から再開)、
または `--no-resume`(履歴を無視)。

`check` は設定の構造、参照ファイル、素子、パラメータ範囲、指標の参照だけを
検証します。波形式の構文や実際のシミュレーションは検証しません。

## OTA ワークフロー：gm/Id、scope、レビュー

全体の手順は [`docs/OTA_WORKFLOW.md`](docs/OTA_WORKFLOW.md) を参照してください。
GUI の gm/Id パネルは既存の NPZ/CSV LUT を読み込みます。リポジトリの合成テーブルは
フロー確認用で、PDK データを含みません。

```bash
./vcal gmid examples/gmid_demo.csv \
  --length 180n --vds 0.6 --vsb 0 --gmid 15 --id 20u --json
```

CLI ではパラメータのグループ確認、1 つの sub-cell scope の実行、履歴のオフライン
レビューができます。

```bash
./vcal scopes <circuit.yaml> --json
./vcal run <circuit.yaml> --scope <scope-name> \
  --stagnation-rounds 6 --initial-points points.json --budget 100
./vcal review <history.jsonl> --config <circuit.yaml> --json
```

`--scope` を省略すると、すべての有効なパラメータを最適化します。scope を指定すると、
`devices` の全エントリがその scope に属するパラメータだけが有効になり、共有パラメータと
選択していない scope のパラメータには、CLI では設定の固定値／公称値、GUI では初期値欄の
値を使います。`--initial-points` は物理 SI
値のマッピングを並べた JSON リスト、`--stagnation-rounds` は全体探索をキューに入れる
までの停滞ラウンド数（0 以上）です。GUI の流れは、gm/Id LUT を開く → 幅/長さの対応
パラメータを選ぶ → 計算結果を初期値として適用 → scope を選ぶ → 既存設定で最適化を開始
→ `review` で履歴を確認、です。

`params.devices` の `scope/instance` は subcircuit master 内の素子を指します。その master
を変更すると、同じ master の全インスタンスに影響します。scope の選択で新しい刺激や
テストベンチは生成されず、シミュレーションは設定済みのネットリストと既存の testbench
を使います。このバージョンは PDK LUT や OTA AC テストベンチを自動生成しません。合成 LUT
とここでの例は、実際の Spectre テストや性能を示すものではありません。

### 表示言語

GUI 上部の言語プルダウンで中国語・日本語・英語を即時に切り替えられます。
選択した言語は直ちに画面へ反映され、次回起動用の設定として保存されます。
CLI ではサブコマンドの前後に指定でき、`./vcal --lang ja gui` や
`./vcal gui --lang en` のように使います。`VCAL_LANG=ja ./vcal gui` も利用
できます。優先順位は明示した `--lang`、`VCAL_LANG`、保存済み GUI 設定、最後に
中国語です。翻訳対象は画面と CLI メッセージだけで、Spectre など第三者のログ、
ユーザー定義の metrics 名、ネットリスト識別子は原文のまま保持されます。
トップレベルのヘルプと `install --help` はシェルが直接出力し、`--lang` または
`VCAL_LANG` を使用します。保存済みの GUI 言語設定は読み込みません。

## GUI の反復グラフ

GUI のグラフは横軸が評価回数で、縦軸のプルダウンには設定されたすべての
`metrics` が表示されます。青は制約を満たす点、橙は制約違反、赤い×は
シミュレーション失敗を表し、失敗点には架空の y 値を入れません。最良の可行
トレンドは目的指標だけに表示されます。点をクリックするとその評価のパラメータ、
指標、エラーを確認でき、ズーム、パン、PNG/CSV 出力を利用できます。

GUI の結果は既定で YAML の隣の `../work/<name>/gui_run/` に保存され、CLI は
同じ回路の `cli_run/` を使います。`best.json`（成功した評価がある場合だけ生成、
`feasible` は制約達成を示す）、`summary.json`、`state.json`、`history.csv` が
自動生成されます。`vcal run --resume` は既存記録を復元し、同じパラメータ点の
シミュレーションを繰り返しません。

## 回路の定義

回路は 1 つの YAML ファイル（[`examples/demo_ota.yaml`](examples/demo_ota.yaml) 参照）:

| 項目 | 意味 |
|------|------|
| `pdk` | `pdks/<name>.yaml` の名前（回路ファイルに PDK パスを書かない） |
| `netlist` / `stimuli` | 素子ネットリスト + 励起デック |
| `params` | 最適化変数：`devices`・`attr`(`w`/`l`/`mr`/`value`)・`lo`/`hi`/`log`/`integer`。`enabled: false` で変数を無効化できます。1 変数で複数素子を駆動可（対称ペアで共有） |
| `metrics` | 名前付き波形式。工学表記(`90n`)、`m['...']` で前の指標を参照可 |
| `objective` | `{metric, goal: maximize/minimize/target, target, tol, weight}` |
| `constraints` | `[{metric, max/min}]` |
| `corners` | PVT リスト：`{name, temp, pdk_corner, tb_extra}` + `corner_worst` 集約 |

MOS の `w` を変更するとレイアウト由来パラメータ(as/ad/ps/pd…)も自動
スケール。回路図編集と同じ意味論です。
トップレベルの `fixed` には `{パラメータ名: 値}` を指定して変数を固定できます。
設定内の相対パスはまず YAML ファイルを基準に解決し、旧来のリポジトリルート基準
の相対パスも互換のため利用できます。

## PDK 設定

PDK は `pdks/<name>.yaml` で定義します（[`pdks/example.yaml`](pdks/example.yaml)
参照）。モデルライブラリ（`${PDK_ROOT}` を使い絶対パスを非公開に）と、
プロセスコーナーごとに include する `.lib` セクションを列挙します。
**PDK 切替は `pdk:` の 1 行**。回路設定に PDK パスは現れません。

## 構成

```
vcal                    統合ランチャ (gui / run / check / serve-gpu / install)
server/optserver/       ネットリスト書換 + シミュレーション + 指標抽出
client/optclient/       BoTorch 最適化器 + リモート GPU 計算サーバ
local/optlocal/         オフラインエンジン + PyQt5 GUI + 回路図書戻し
pdks/                   PDK 定義（example.yaml がテンプレート）
examples/               サニタイズ済みサンプル回路設定
```

## ライセンス

[LICENSE](LICENSE) を参照。
