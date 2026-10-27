# INT8 query canister 最適化の実測

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26。通常のAPI上限を512 tokensに設定し、入力長に応じて1〜6回のqueryで処理する実装へ変更した。
256 tokensでは3回、512 tokensでは6回で全12層の推論が完了する。

## 同じ512 tokensでの比較

入力長を短くした効果とkernel最適化の効果を分けるため、同じ512 tokensの初版も測定した。

| 指標 | 初版 | 最適化後のquery API |
| --- | ---: | ---: |
| 入力tokens | 512 | 512 |
| 実行した層 | 12 | 12 |
| query数 | 99 | **6** |
| ローカル経過時間 | 3.702秒 | **1.857秒** |
| 往復転送量 | 99,890,890 bytes | **7,868,474 bytes** |
| 総計測命令数 | 59,549,691,355 | **25,033,165,865** |
| 最大query命令数 | 2,575,130,062 | 4,520,189,986 |
| 最大request | 1,277,996 bytes | 786,454 bytes |
| 最大response | 1,769,518 bytes | 786,507 bytes |
| F32参照とのコサイン類似度 | 0.9995573850 | 0.9995917930 |

query回数は16.5分の1、転送量は約12.7分の1、総命令数は約58%減。
この測定ではローカル時間が約2分の1になった。
初版の5,000 tokens・687 query・74秒と、入力長をそろえずに速度改善率を比較していない。

## 入力長ごとの計画

| 入力tokens | 1 queryの層数 | query数 | 実測時間 | 往復bytes | 最大query命令数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 12 | 1 | 0.069秒 | 1,748 | 1,178,371,400 |
| 64 | 12 | 1 | 0.131秒 | 1,876 | 2,467,356,409 |
| 128 | 6 | 2 | 0.314秒 | 395,446 | 2,636,582,521 |
| 256 | 4 | 3 | 0.669秒 | 1,575,703 | 3,824,219,764 |
| 512 | 2 | 6 | 1.857秒 | 7,868,474 | 4,520,189,986 |

APIは長さ2〜64、65〜128、129〜256、257〜512の各範囲でこの層数を選ぶ。
短い検索queryは1回で処理し、文書では512 tokensを通常の上限にする。
速度・通信量を優先する使い方では256 tokensが候補になる。

513 tokens以上はエラーで返し、暗黙に切り捨てない。
長文を段落等に分けて検索用embeddingを作る場合、それぞれを独立した入力として扱う。
独立した部分のembeddingを、元の全文embeddingと同じものだとは扱わない。

## 変更した計算

1. INT8行列計算を4 tokens × 4出力channelでブロック化。入力・重みのSIMDロードを共有し、K方向を展開した。
2. QKV、attention、output projection、MLPの中間値をqueryの内部に保持。K/Vをquery間で繰り返し送る初版の方式をやめた。
3. 複数層を1 queryにまとめた。query間のcarryは層境界のF32 hidden stateのみで、最終queryは384次元のembeddingを直接返す。
4. RoPEのfrequencyを準備し、同じtokenのsin/cosを複数headで共有した。
5. softmaxのexpを範囲縮約付き6次多項式に変更し、確率の正規化に使う逆数を共有した。
6. 検証したshapeから安全性を確認できるattention内側のアクセスで、反復するslice境界処理を減らした。

元の12層を省略していない。受け付けた入力全体に対するglobal attentionと、前後64 tokensのlocal attentionを保持する。
LinearのINT8量子化、F32のnorm・GeGLU・残差、CLS poolingとL2正規化も保持する。
attention内部のK/VのBF16丸めは、初版との比較条件をそろえるため残している。

expの近似はlosslessではない。
native unit testでは[-80,0]の80,001点で標準expに対する相対誤差2e-6未満を確認した。
-80未満は0として扱う。最終embeddingについても別途F32/PyTorch参照と比較した。

## 異なる文書の確認

512 tokensの内容を変えて、公開query APIで処理した。

| 文書 | query数 | ローカル時間 | 最大query命令数 | F32参照とのコサイン類似度 |
| --- | ---: | ---: | ---: | ---: |
| 技術説明 | 6 | 1.857秒 | 4,520,189,986 | 0.9995917930 |
| 天文 | 6 | 1.909秒 | 4,518,184,512 | 0.9996220160 |
| 料理 | 6 | 2.236秒 | 4,519,359,440 | 0.9996805621 |
| 天候 | 6 | 2.438秒 | 4,521,453,056 | 0.9995877031 |

すべて50億命令上限内で完走した。最も大きいものでも上限まで約9.6%の余裕がある。
これは測定した固定サンプルについての値であり、全入力に対する最悪命令数の証明ではない。
各サンプルは説明文を反復して所定のtoken数にしたもので、retrieval benchmarkではない。
コサイン類似度を検索精度の維持率として扱わない。

## 検証

- 固定revisionの配布重みをF32に展開したHugging Face `AutoModel` と比較。
- export済みINT8重みと動的活性量子化を使う、独立したPyTorch参照とも比較。
- 512 tokensのtyped query APIと内部fused queryの出力は全要素で完全一致。
- 32 tokensではtyped API、内部診断graph、細分化したquery graphの出力が全要素で完全一致。
- 長さ超過、不正な進捗、NaNを含む中間状態をAPIがエラーで返すことを確認。
- 6つのunit testでINT8累積、16出力すべてのtile計算、BF16丸め、exp近似、attention範囲、長さごとの実行計画を検査。
- Wasmビルド、Candid抽出、各queryの命令・request・responseサイズの検査が通過。

計測はApple M6 / RAM 32 GiB / arm64 macOS、Rust 1.97.1、PocketIC server 11.0.0。
Application subnetのProduction設定を使い、命令上限を増やすBenchmarking設定は使用していない。
計測中に別のベンチマークを並行実行していない。
既存のシステム負荷は制御しておらず、時間は単回測定値。mainnetの時間は未測定。
モデル取得・ビルド・ロード・不正入力テストの時間は推論時間に含めない。
bytesはCandid payloadの合計で、HTTPヘッダー等は含まない。
命令数はメソッド内部のcounter差分であり、入出力Candid処理の全量を含まない。
実行そのものにはPocketICの50億命令制限が適用されている。

## APIと再現情報

`startEmbedding` と `continueEmbedding` の2メソッドを追加した。
返り値に進捗、state、最終embedding、命令数が入り、クライアントは完了までqueryを繰り返す。
長さごとの計画はcanister側で選ぶ。
中間状態の改変防止、upgrade後の重み復元、公開サービス向けの運用機能は今回の範囲外。

- [Candid](../../crates/canister/granite.did)
- [再実行手順](../../README.md)
- 当時の`validate_optimized.py`（旧runnerは撤去済み）
- [環境、実行したWasmとソースのhash](environment.json)
- [参照出力、モデルrevision、重みhash](references.json)
- [512 tokens API](api-512.json) / [256 tokens API](api-256.json)
- [同じ512 tokensの初版](baseline-512.json)

旧版の5,000 tokens実験は [以前のレポート](../RESULTS.md) に履歴として残す。
