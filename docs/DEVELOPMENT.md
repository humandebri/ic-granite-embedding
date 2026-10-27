# ビルドと検証の再現手順

リポジトリのルートから実行します。以下の例は`CARGO_TARGET_DIR`と`GRANITE_WASM`が未設定のshellを前提とします。
カスタムビルド先を使う場合は、末尾の[環境変数](#環境変数)を参照してください。

## 使用した環境

| 対象 | 固定値・準備方法 |
| --- | --- |
| Rust | 1.97.1。[rust-toolchain.toml](../rust-toolchain.toml)で指定 |
| Wasm target | `wasm32-unknown-unknown`。SIMD128は[.cargo/config.toml](../.cargo/config.toml)で有効化 |
| Python | 3.12。以下の例では`uv`で仮想環境を作る |
| Python依存 | [requirements.txt](../requirements.txt)でバージョン固定 |
| PocketIC server | 11.0.0の実行ファイルを別途用意し、`POCKET_IC_BIN`で指定 |
| モデルrevision | `2ab6fa8ea2d674564defd37171ae19079b864b33` |

PocketICのRust clientは[crates/bench/Cargo.toml](../crates/bench/Cargo.toml)で11.0.0に固定しています。
付属ハーネスは専用のApplication subnetを作り、終了時に破棄します。既存のicp-cliネットワークやmainnet canisterは操作しません。

## 通常Wasmをビルドする

```sh
cargo build --release --locked -p granite-canister --target wasm32-unknown-unknown
```

成果物は`target/wasm32-unknown-unknown/release/granite_canister.wasm`です。
重みは別途ロードします。ビルドだけならPython環境や参照データは不要です。
通常ビルドに`profileStep`は含まれません。

## モデルと参照データを準備する

初回は固定revisionの公開モデルを取得します。
`scripts/prepare.py`は元のsafetensors SHA256を検査し、INT8重み・token ID・FP32/INT8参照を生成します。

```sh
uv venv --python 3.12 .cache/venv
uv pip install --python .cache/venv/bin/python -r requirements.txt

.cache/venv/bin/python scripts/prepare.py \
  --lengths=32,64,128,160,176,256,512,640,768,1024,1280,1365 \
  --stress --stress-lengths=160,256,512,1280,1365
```

主な出力は次のとおりです。

| `.cache/`内のファイル | 用途 |
| --- | --- |
| `model.pack` | controllerがアップロードする固定モデルの重み |
| `tokens-<case>.json` | CLS/SEP込みのtoken ID列 |
| `reference-<case>.f32` | FP32参照、little-endian F32、384要素 |
| `reference-int8-<case>.f32` | W8A8・BF16 K/Vの独立参照 |
| `references.json` | revision、重み・PACKのhash、各参照caseの記録 |

`--output <directory>`で出力先を変更できます。再生成はその出力先の同名ファイルを上書きします。

検証スクリプトは、初回実験で使った2つのprovenanceパスも参照します。
上の1回の生成には必要な長文caseをすべて含めているため、同じmanifestを両方へ配置できます。重みPACKの複製は不要です。

```sh
mkdir -p .cache/frontier/references .cache/frontier/references-max
cp .cache/references.json .cache/frontier/references/references.json
cp .cache/references.json .cache/frontier/references-max/references.json
```

## 通常ビルドを検証する

```sh
export POCKET_IC_BIN=/absolute/path/to/pocket-ic
"$POCKET_IC_BIN" --version
python3 scripts/validate_frontier.py
```

この実行は、unit test、通常Wasmとハーネスのビルド後、次のsuiteを並行して実行します。

| 検査 | 範囲 |
| --- | --- |
| 旧入力域 | 2〜512の全511長、追加57ケース、公開API 16ケース。保存済み旧版出力とのbit一致 |
| 長文 | 32長×3入力の96ケース。公開APIの上限・有限値・384次元・L2 norm |
| 長文参照 | 11ケース。FP32参照とのcosine、request/responseサイズ |
| 1 query境界 | 289〜320の各長×3入力、計96ケース。成功と実際の命令上限超過を区別 |

Candid一致、通常Wasmの`profileStep`非公開、上限超過入力・不正な進捗・NaN stateの拒否も検査します。
結果は[benchmarks/validation](../benchmarks/validation/)に保存します。
検証suite同士が並行するため、その実行時間はレイテンシ比較には使いません。

計測版のAV 44ケースと1,365 tokensの層別結果は、保存済みの[av.json](../benchmarks/full-context/av.json)・[qk.json](../benchmarks/full-context/qk.json)・[profile-1365.json](../benchmarks/full-context/profile-1365.json)を照合します。
通常検証が、この2つのprofiling測定を自動で再実行するわけではありません。
現在のソースからすべて測り直す場合は、先に[計測版の手順](#計測版を再実行する)でこれらを更新してください。
過去段階との比較に使う`benchmarks/exhaustive-optimization/`の記録は保持します。

### 既存のビルド・結果を使う場合

- `--already-built`は、通常Wasm・ハーネス・`.cache/frontier/final-tests.log`が準備済みの場合に使います。suiteの推論は再実行します。
- `--summarize-only`は、完了済みsuiteのJSONを再照合する用途です。推論を再実行せず、最新ソースの動作確認の代わりにはなりません。
- `scripts/validate_regression.py`は旧入力域の比較を担当します。長文まで検証する入口は`validate_frontier.py`です。

いずれも測定WasmのSHA256を照合し、異なるmoduleの結果を混ぜないようにしています。

## 個別のqueryを実行する

モデル・参照データとPocketIC実行ファイルを準備した後、通常Wasmとハーネスをビルドします。

```sh
cargo build --release --locked -p granite-canister --target wasm32-unknown-unknown
cargo build --release --locked -p granite-bench

target/release/granite-bench 256
target/release/granite-bench 1365
```

`target/release/granite-bench --help`で現在のオプションを確認できます。
引数なしの既定長は256です。2番目の数値でtileを指定する旧CLIは撤去しました。ハーネスがモデルをロードし、公開APIの継続呼び出しまで処理します。
最初の引数の長さに対応するfixtureが必要です。`--case=space`なら`tokens-space-<length>.json`などを読みます。
各実行は`.cache/`の対応する測定JSONを上書きします。保存したい結果は別directoryへコピーしてください。

```sh
# 語彙内の3入力で1 queryの境界を調べる。公開APIの結果ともbit比較する。
target/release/granite-bench 256 --capacity-range=289:320 --capacity-patterns=3

# 同じRust宣言から通常Candidを出力する。
cargo run --release --locked -p granite-canister --example export_candid
```

## 計測版を再実行する

計測版は別targetへビルドします。以下の`GRANITE_WASM`は1コマンドだけに適用し、通常検証への指定が残らない形にしています。

```sh
cargo build --release --locked -p granite-bench
CARGO_TARGET_DIR=.cache/profiling-target cargo build --release --locked \
  -p granite-canister --target wasm32-unknown-unknown --features profiling

GRANITE_WASM="$PWD/.cache/profiling-target/wasm32-unknown-unknown/release/granite_canister.wasm" \
  target/release/granite-bench 1365 --micro-av --micro-attention --micro-qk --micro --micro-extended --probe
```

| 出力 | 内容 |
| --- | --- |
| `.cache/micro-av.json` | 512/513境界から5,000要素まで、i64参照と比較する44ケース |
| `.cache/micro-qk.json` | QKを旧SIMD経路・独立したi64積和と比較する44ケース |
| `.cache/micro-linear.json` | 独立したnative整数参照と比較するGEMM 112ケース（通常版でも実行可能） |
| `.cache/probe-profile.json` | 指定した長さの12層別命令数と最終embedding |
| `.cache/probe-single.json` | 256 tokensの4文書を通常診断`step`で実行した結果 |

AVの5,000要素検査はAttentionの積和単体です。全文推論は別途[分割queryの8ケース](SHARDED.md)で検証します。QKは`--micro-qk`で44ケースを旧経路・i64と比較でき、`--micro-attention`は256/512/1,365/5,000 keysの1-head Attentionを層別に計測します。

`validate_frontier.py`で照合する計測データを更新する場合は、結果を次のパスへ保存します。
過去の測定を残す場合は、コピー前に退避してください。

```sh
cp .cache/micro-av.json benchmarks/full-context/av.json
cp .cache/micro-linear.json benchmarks/full-context/linear.json
cp .cache/micro-qk.json benchmarks/full-context/qk.json
cp .cache/probe-profile.json benchmarks/full-context/profile-1365.json

python3 scripts/summarize_profile.py .cache/probe-profile.json
```

256 tokensでphaseを比較する場合は`--probe`の入力長を256にして再実行します。
計測版はcodeや最適化結果も変わるため、その命令数を通常版へ足し合わせません。
採否は通常ビルドで再確認します。

## 時間を比較する

全ビルド・ほかのベンチマークが終了した後、通常Wasmを指定して実行します。

```sh
GRANITE_WASM="$PWD/target/wasm32-unknown-unknown/release/granite_canister.wasm" \
  target/release/granite-bench 256 --probe-single --repeat-single
```

4文書×5回を測り、診断requestの未使用fieldのnonceを変えてqueryキャッシュを避けます。
出力一致も確認し、`.cache/repeat-single.json`に保存します。
比較対象ごとに結果を退避して、順番に測定してください。同一requestの再送時間を推論速度として扱わないでください。

## 数値仕様とfeature

Linearは出力channelごとのINT8重みとtokenごとの動的INT8活性値を使います。
重みを有効化時にi16へ展開し、Wasm SIMDの整数積和で計算します。
Embedding table、LayerNorm、残差、pooling、state転送はF32です。K/VはBF16へ丸めます。
GELUにはerf近似、Attentionには整数QK/AVを使うため、元のFP32モデルとのbit一致を目指す構成ではありません。

default featuresは[crates/canister/Cargo.toml](../crates/canister/Cargo.toml)の7つです。

| feature | 処理 |
| --- | --- |
| `approx-gelu` | SIMDのerf近似GELU |
| `integer-attention` | QKは符号付き14 bit、Vは符号付き12 bit、確率は整数化。長いAVはi64で合算 |
| `strassen` | Wiに1段Strassen。整数で復元してから元のF32 scaleを適用 |
| `chunked-gemm` | 大きいkernelのK方向をブロック化 |
| `strassen-wo` | Woにも1段Strassen。重みの和差は有効化時に準備 |
| `strassen-qkv` | QKV射影にも1段Strassen |
| `strassen-out` | Attention出力射影にも1段Strassen |

上書きするbufferのゼロ初期化省略、LayerNormの順序を保ったSIMD化、RoPEの事前計算、softmaxと整数化の融合も行います。
最後の3層ではCLSに必要なprefixを129→65→1へ絞る既存の計算方法を維持します。
実際の生成形状は[build.rs](../crates/core/build.rs)、数値差と命令数は[最新レポート](../benchmarks/frontier/RESULTS.md)を参照してください。

Wasmのkernel形状は採用済みの384列32×64、1536列32×32に固定しています。端数は8×32・4×32・1×16です。
`profiling`は計測専用です。旧圧縮stateのdecode互換は残し、新しく返すstateはraw F32に統一しています。
公開上限の資源検証はdefault構成に対するもので、ほかのfeature構成では命令数を再測定する必要があります。

## 環境変数

| 変数 | 意味 |
| --- | --- |
| `POCKET_IC_BIN` | PocketIC server 11.0.0の実行ファイル。ハーネス・検証で必須 |
| `CARGO_TARGET_DIR` | Cargoと検証スクリプトのビルド先。指定時はハーネスもそのdirectoryの`release/`を使う |
| `GRANITE_WASM` | 測定するWasmのパス。未指定のハーネスはworkspaceの`target/wasm32-unknown-unknown/release/granite_canister.wasm`を使う |

`CARGO_TARGET_DIR`を指定したビルドを手動でハーネスへ渡す場合は、`GRANITE_WASM`も対応するパスへ指定します。
通常検証へprofiling版を渡すと、公開exportの検査で拒否します。

## 5,000 tokensの分割query

[分割クライアントとバイナリ形式](SHARDED.md)に、追加の参照生成・実行・8ケースの検証手順をまとめています。通常の公開APIの1,365 tokens上限と、分割経路の5,000 tokens上限は別です。

全域Attentionのキー長が512以下のときは、4つのQ行でKのロードを共有します。長文経路のコンパイル結果への影響を避けるため、同じ行kernelのソースを通常経路とbatch経路で別々にコンパイルします。profiling版の差を通常版の削減量として加算せず、通常Wasmで命令数を確認してください。

## GitとCI

モデル重み・`.cache/`・`target/`・ローカル環境設定はGitの管理対象外です。重みと参照fixtureはこの文書の手順で生成します。`benchmarks/`には測定結果とそのSHAを保存しています。

GitHub Actionsはformat・Rust unit test・通常Wasm build・Candid一致を検査します。モデルをロードするPocketIC検証はCIには含めず、ローカルの検証手順で実行します。
