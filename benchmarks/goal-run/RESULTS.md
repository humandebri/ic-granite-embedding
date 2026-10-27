# Goal達成: 256 tokensを通常の1 queryへ

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26。PocketIC server 11.0.0、Application subnet / Production設定。
**CLS/SEP込み256 tokens全体・全12層を、公開APIの通常query 1回で処理した。**
`startEmbedding` が `next_layer=12` と384次元embeddingを返し、中間stateは空になる。
4文書すべて **4,737,545,056命令**で完走。50億命令の上限への余裕は約5.25%。
命令制限の引き上げ、updateへの移行、入力の切り捨て、層の削除は行っていない。

| 入力tokens | 公開API query数 | 合計命令 | 最大/query | ローカル時間 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1 | 5.81億 | 5.81億 | 0.096秒 |
| 64 | 1 | 12.00億 | 12.00億 | 0.182秒 |
| 128 | 1 | 24.10億 | 24.10億 | 0.334秒 |
| 160 | 1 | 29.77億 | 29.77億 | 0.406秒 |
| 176 | 1 | 32.83億 | 32.83億 | 0.459秒 |
| 256 | 1 | 47.38億 | 47.38億 | 0.628秒 |
| 512 | 5 | 109.61億 | 24.50億 | 1.329秒 |

前版の256 tokensは2 query・7,307,496,652命令だった。合計命令は約35.2%削減した。
往復転送量は679,268から2,645 bytesへ減った。query間の中間state転送を省けた効果が大きい。
256 tokensの4文書のローカル時間は0.590〜0.628秒。
代表文書は前版0.582秒、今回0.628秒。時間は各ケース1回の測定で、統計的な速度比較やmainnetのレイテンシ保証ではない。

## ボトルネックと変更

[初回の段階別計測](../goal-profile/baseline-summary.json)では、GEMMとスケール適用が約57%、活性値のINT8量子化とバッファ準備が約12%、GELUが約8%、QKとAVが計約16%だった。
各層・各射影を分けた計測から、最大値計算、丸め、部分和の合成、F32変換、スケール適用まで調べた。

- 重みのINT8化はオフライン、i16展開はモデル有効化時に行う。query内で重みを量子化しない。
- 活性値は入力依存なので、最大値・丸め・整数変換をSIMD化した。32個の単独GEMM fixtureでnative参照と完全一致した。
- 32 tokens × 16出力channelの固定形状kernelを生成し、ロードを共有する。整数積和の合成からF32変換とスケール適用までSIMDでつなぐ。
- RoPEのsin/cosをモデル有効化時に5000位置分準備する。tokenごとに係数を一度コピーしてhead間で再利用する。query中の三角関数計算を省いた。
- QKを符号付き14bit、AVを符号付き12bitの整数演算へ変更した。QKは32要素、AVは512 keysまでの積和がi32に収まる。
- GELUをSIMDのerf近似へ変更した。単体Wasm fixtureで最大絶対誤差は約9.54×10⁻⁷。240,001点のscalar gridも検査した。
- QKVを直接渡し、内部でのバイト列化・復元を省いた。K/VのBF16丸めとCLS依存範囲の剪定は維持した。

最終計測用ビルドの層別合算内訳。通常の単一query実測とは区別する。

| 段階 | 命令 | 比率 |
| --- | ---: | ---: |
| INT8 GEMM・水平加算・スケール | 33.414億 | 69.63% |
| QK積 | 3.812億 | 7.94% |
| AV積 | 3.248億 | 6.77% |
| その他・計測処理・境界入出力 | 1.856億 | 3.87% |
| GELU | 1.718億 | 3.58% |
| 活性値量子化・バッファ準備 | 1.607億 | 3.35% |
| Softmax | 1.268億 | 2.64% |
| QK量子化 | 0.348億 | 0.72% |
| V転置・量子化 | 0.307億 | 0.64% |
| RoPE回転 | 0.222億 | 0.46% |
| QKVの分割・BF16丸め | 0.187億 | 0.39% |

[最終プロファイル](../goal-profile/final-summary.json)と[層別の全計測値](../goal-profile/final-256.json)を保存した。
計測用ビルドは層別queryの合算で、計測処理・層境界の入出力を含む。上表の公開APIは計測コードを含まない通常ビルドで測った。
一部の中間版は行ごとの文字列集計コストを含むため、最終版ではcounterを数値で累積してから集計する方式へ改めた。
計測用ビルドと通常ビルドの4文書のF32出力はbit単位で一致した。[計測ビルドの環境](../goal-profile/environment-final.json)。

Strassenも比較した。3段分解は合成・バッファ処理が乗算削減分を打ち消した。
1段分解とSIMD合成では一部改善したが、今回の通常kernelを採用した。実験記録は `../goal-profile/linear-strassen*.json`、ソース断片は `../goal-profile/rejected/`。

## 精度と検証

高速GELUと整数Attentionは追加の近似を含み、前版から出力が少し変わる。
公開API 16ケースのFP32参照との最小cosineは **0.999486899**。
前版と比較できた12ケースの最小cosineは **0.999542187**。
参照は同じ固定revisionのFP32 ModernBERTで、語彙・層数・CLS pooling・L2正規化を維持している。
fixtureは繰り返しを含む技術説明・天文・料理・天候の文書で、retrieval benchmarkではない。

- unit test 9件、公開API 16ケースを通過。
- 256 tokensの4文書で、診断用の通常queryと型付き公開APIのF32出力がbit単位で一致。
- RoPE事前準備前後の4文書もF32出力がbit単位で一致。
- 過大入力、不正な継続位置、非有限値の中間stateを拒否する検査を通過。
- 384次元、全要素有限、L2 norm、request/responseサイズ、実際の50億命令制限を検査。
- 512 tokensの5 query経路も4文書で回帰検証した。

[実行環境とsource hash](environment.json)、[公開APIの要約](summary.json)、[単一queryの4文書](ordinary-single-256.json)、[入力と参照](references.json)。
通常Wasm SHA256: `fdb32f8817d475ffe73b3c9239910b1369006f52955d9d3e060d29a482e345ff`、2,638,097 bytes。

## 再現

READMEのfixture生成とPocketIC設定後、`python3 scripts/validate_goal.py`。
計測用は `profiling` featureと`granite-bench --probe`、集計は `scripts/summarize_profile.py` を使う。

大きいkernelの初回Wasmコンパイルは重い。手元の旧`candid-extractor`ではCranelift 0.117.2のassertionで失敗したため、同じRust宣言をnativeで出力する `cargo run --release --locked -p granite-canister --example export_candid` で既存Candidとの一致を検査した。

外部サービスへのデプロイは行っていない。重み・RoPE tableはheapに保持し、upgrade時の復元は今後の実装対象。
