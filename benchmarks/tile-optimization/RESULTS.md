# 行列積タイルの追加比較と命令数優先オプション

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26。PocketIC server 11.0.0、Application subnet / Production、1 query上限50億命令。

既定構成を維持し、`instruction-gemm` featureを追加した。
256 tokensの公開APIは **4,274,785,648 → 4,230,646,590命令（約1.03%削減）**。
一方、最終バイナリを順番に測った中央値は **0.476155 → 0.565728秒（約18.8%増加）**だった。
速度低下を避けるため、命令数優先オプションは既定では無効にした。
INT8、入力全体、全12層、CLS pooling、既存の近似は維持し、今回の変更で追加の近似は導入していない。

両構成とも2〜288 tokensは1 query、289〜512 tokensは2 query `[5,7]`。
上限・query数は前回と同じ。今回の変更で5,000 tokens全体のembeddingを可能にしたわけではない。

## 通常公開APIの結果

| tokens | query数 | 既定・合計命令 | 命令数優先・合計命令 | 命令数優先・最大/query |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 4,274,785,648 | 4,230,646,590 | 4,230,646,590 |
| 288 | 1 | 4,826,633,982 | 4,776,151,904 | 4,776,151,904 |
| 512 | 2 | 8,836,065,163 | 8,751,099,369 | 4,655,320,677 |

288 tokensは固定seedの長さ走査入力、256/512は既存のtokenized文書。
256 tokensでは4文書とも同じ命令数になった。
256 tokensの上限への余裕は約14.50% → 約15.39%。
全長走査の最大/queryは両構成とも287 tokensで、4,871,607,769 → 4,812,141,180命令、余裕は約2.57% → 約3.76%。
固定seedの各長1入力と追加の反復入力による検証であり、あらゆる入力の最悪値の証明ではない。

[比較集計](comparison.json)、[既定構成](default/summary.json)、[命令数優先](instruction-gemm/summary.json)。

## 変更した部分

- 1536列・32 tokensの行列積を16出力から32出力へ広げ、入力ロードを共有する。
- 384列・32 tokensの射影を32出力から64出力へ広げる。
- Strassenの192列leafを64出力から128出力へ広げる。
- 8/4 tokensの端数処理を16出力から32出力へ広げる。1 token用は悪化したため16出力を維持する。
- 分割しないkernelでは4出力channelを計算するごとに結果を書き出す。
  入力は共有したまま、重みと累積値の寿命を短くする。整数和とF32 scaleの積の順序を維持する。
- 上記の生成・呼び出しを`instruction-gemm` featureで一緒に切り替える。既定構成では従来のkernelを生成する。

生成kernelの入力・重み・出力は呼び出し元で別々のallocationに置く。
未初期化出力は従来どおり全要素を書いてからVecの長さを設定する。
[生成処理](../../crates/core/build.rs)、[通常GEMMのdispatch](../../crates/core/src/lib.rs)、[Strassen leafのdispatch](../../crates/core/src/strassen.rs)。

Strassenの入力準備は前回のプロファイル全体の約0.10%、合成・scale適用は約1.20%だった。
バッファ再利用より行列積本体を優先した。[準備の内訳](strassen-preparation-analysis.json)。

## 比較した候補

下表はprofiling feature付きWasmの通常診断step・256 tokensであり、上の通常公開APIの命令数とは区別する。
比較元は4,302,540,574命令。各候補はGEMM 112ケースで独立したnative整数参照と一致し、文書4件も前回とbit単位で一致した。

| 候補 | 命令 | 比較元からの削減 |
| --- | ---: | ---: |
| wide | 4,290,742,210 | 11,798,364 |
| wider | 4,281,436,438 | 21,104,136 |
| leaf | 4,273,924,570 | 28,616,004 |
| tails | 4,274,067,438 | 28,473,136 |
| selected | 4,273,924,620 | 28,615,954 |
| streamed | 4,258,401,420 | 44,139,154 |
| balanced | 4,287,783,196 | 14,757,378 |

`wide`は1536列の拡大、`wider`は384列も拡大、`leaf`はStrassen leafも拡大。
`tails`は1 token用まで拡大した案、`selected`は1 token用を戻した案。
`streamed`は4出力ごとの書き出しを追加した最終オプションと同じ形状、`balanced`は主タイルを従来サイズへ戻した比較案。
小タイル案も初回の比較では0.531秒と既定の0.476秒より遅かったため、既定には採用しなかった。
[段階別測定](experiments.json)、[当初の通常バイナリ比較](latency-comparison.json)。

早期書き出しで、宣言されたSIMD local数の最大は7,692 → 2,608へ減った。
大タイル案の初回installは書き出し変更前の約100秒から約47秒へ短くなったが、最大関数のbodyは約1.35MBで、従来の約0.68MBより大きい。
これらはWasmのコード量・local数の測定で、実行速度悪化のCPU上の原因を断定するものではない。
[kernelのサイズとlocal数](kernel-local-counts.json)。

12層を分けた計測のGEMM・水平加算・scaleは3,166,768,421 → 3,122,629,267命令。
[最終プロファイル](final-profile-summary.json)、[各層の生データ](final-profile.json)、[計測ビルドの環境](profile-environment.json)。

## 時間・メモリ

最終測定は全ビルドと検証の終了後に、既定→命令数優先の順に実行した。
各構成で4文書×5回。計算に使わないrequest fieldのnonceを毎回変更してqueryキャッシュを避け、各出力が一致することも検査した。
OS負荷や温度は管理しておらず、mainnetのレイテンシ保証ではない。

| 構成 | 256 tokens中央値 | 最小〜最大 | 初回install |
| --- | ---: | ---: | ---: |
| 既定 | 0.476155秒 | 0.470582〜0.526431秒 | 21.53秒 |
| 命令数優先 | 0.565728秒 | 0.561634〜0.588866秒 | 40.62秒 |

先行比較では大タイル案の中央値が0.635秒だった。上表は公開する最終バイナリでの再測定値。
[最終時間比較](final-latency-comparison.json)、[既定20試行](final-paired-default-repeat.json)、[命令数優先20試行](final-paired-instruction-repeat.json)。

有効化直後のWasm heapは両構成とも312,868,864 bytes。
コード量は約3.19MB → 約5.43MBに増える。query中のpeak heapを保証する測定ではない。
512 tokensのquery数・raw F32 carry・転送量は同じで、往復1,576,654 bytes。
[既定のmemory](default/model-memory.json)、[命令数優先のmemory](instruction-gemm/model-memory.json)。

## 検証

- 両構成で2〜512 tokensの全511長を公開APIで実行し、各queryが50億命令以内。
- 両構成の全511長、追加57ケース、既存API 16ケースで、前回版とF32出力がbit単位で一致。
- native unit testは両構成で10件通過。Strassen 1〜3段の整数上限・i64直接計算との一致を含む。
- 最終計測ビルドのGEMM 112ケースはnative整数参照と一致。通常/計測ビルドの4文書と層別traceの最終出力も一致。
- 513 tokens、不正な継続位置、正しい継続位置5でのNaN stateを拒否。
- Candid一致、通常WasmのprofileStep非公開、request/responseサイズ、384次元・有限値・L2 normを確認。
- FP32参照との最小cosineは0.999486899で変わらない。retrievalのRecall@kは評価していない。

[既定の環境・source SHA](default/environment.json)、[命令数優先の環境・source SHA](instruction-gemm/environment.json)。
[既定の全長走査](default/sweep.json)、[命令数優先の全長走査](instruction-gemm/sweep.json)。

## 再現と成果物

READMEのfixture準備後に以下を実行する。

```sh
python3 scripts/validate_exhaustive.py
python3 scripts/validate_exhaustive.py --instruction-gemm
```

`CARGO_TARGET_DIR`でビルド先、`GRANITE_WASM`で検証する通常Wasmを指定できる。
検証先はそれぞれ`benchmarks/tile-optimization/default/`、`instruction-gemm/`。
`--output`で変更可能。過去の比較元は`benchmarks/exhaustive-optimization/`に固定している。
profiling版は`--features instruction-gemm,profiling`でbuildし、`granite-bench 256 384 --micro --micro-extended --probe`で計測する。

今回の検証済み通常artifactは`target/wasm32-unknown-unknown/release/`に別名で配置した。
既定は`granite_canister.wasm`、命令数優先は`granite_canister_instruction.wasm`。
Cargoで再buildした場合は、選んだ構成が標準名`granite_canister.wasm`に出力される。

| 構成 | bytes | SHA256 |
| --- | ---: | --- |
| default | 3,194,652 | `c1ba65ae15c1868d210394e10123bc7c80d6942a1c003def9ac714f6214ee8f4` |
| instruction-gemm | 5,434,900 | `faa1c76da109a72f59cddef91e2e889c6c471d7b2cf65188a443ec44c3d27121` |

[artifact一覧](artifacts.json)。mainnetにはdeployしていない。
