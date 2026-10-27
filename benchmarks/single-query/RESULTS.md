# 256 tokensの1 query化を目指した追加実験

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26、PocketIC 11.0.0 / Application subnet / Production設定。
**256 tokensの通常の1 query化は未達**。最終Wasmで全12層を1 queryに指定すると、実際に50億命令制限で拒否された。
[最終Wasmでの拒否ログ](rejected-single-256.log)を保存した。命令上限は変更していない。

公開APIで1 queryにする上限は128から**160 tokens**に拡張した。token数はCLS/SEP込み。
176 tokensも4文書で1 query完走したが、最大4,934,870,193命令で余裕が約1.3%しかないため、公開APIでは2 queryに分ける。

| 入力tokens | 公開API query数 | 合計命令（十億） | 最大/query（十億） | ローカル時間 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1 | 0.866 | 0.866 | 0.084秒 |
| 64 | 1 | 1.800 | 1.800 | 0.157秒 |
| 128 | 1 | 3.631 | 3.631 | 0.291秒 |
| 160 | 1 | 4.493 | 4.493 | 0.363秒 |
| 256 | 2 | 7.307 | 3.714 | 0.582秒 |
| 512 | 5 | 16.211 | 3.655 | 1.327秒 |

256 tokensは前版の9,339,567,602から7,307,496,652命令へ約21.8%削減した。
同じ代表文書の時間は0.668秒から0.582秒へ改善した。時間は各ケース1回のローカル測定で、mainnetのレイテンシ保証や統計的な速度比較ではない。
256 tokensの4文書の時間は0.58〜0.66秒。1 queryに収めるには、残る計算をさらに約3割減らす必要がある。

## 採用した変更

- INT8 GEMMを8 tokens × 8出力channelの固定形状kernelへ変更。K方向を展開し、activationとweightを再利用する。build.rsから384/1536幅のkernelを生成する。
- i32の加算を木構造にし、Wasmのaccumulatorの読み書きを減らす。最大絶対積和は1536×127²でi32に収まる。
- AttentionのQを再利用し、Vの加重和をSIMDレジスタに保持する。
- Softmaxの既存多項式を4要素ずつSIMDで計算する。分母の加算順は従来どおりにする。
- 全12層、256 tokens全体、既存のCLS依存範囲、INT8量子化、K/VのBF16丸め、F32中間状態を維持する。

Strassenの2段分解も試したが、256 tokensで約107.7億命令へ悪化したため採用しなかった。
[不採用実験の記録](strassen2-rejected-256.json)、unrolled-256.json、tree8-256.json、simd-softmax-256.json、attempt-*.logは探索途中の別Wasmの記録。それぞれのJSON内のWasm hashを参照。最終結果と混同しない。

## 検証

最終Wasmでunit test 8件、公開APIの入力検査、32/64/128/160/256/512 tokens、160/256/512 tokensの追加3文書、176 tokensの診断query 4文書、診断経路と公開APIの一致を確認した。
前版と比較できる公開API 11ケースでembeddingが全要素完全一致した。
最終Wasmの256 tokensの単一queryは制限超過を確認し、成功例として数えていない。

160 tokensの4文書は最大4,498,927,152命令。入力内容・長さ全域に対する最悪命令数の証明ではない。
176 tokensの測定値は診断用 `step` の1 query実測であり、公開APIの1 query対応長ではない。
FP32参照とのcosineは公開APIの全ケースで0.9995以上。文書は繰り返しを含む合成fixtureで、検索品質benchmarkではない。

最終Wasm SHA256: `ed5c9c31e5d370fac95dece4082b48085ad21a22d138dd47dfbc229a999330b9`、1,066,064 bytes。
[環境・source hash](environment.json)、[入力と参照](references.json)、[256 tokens実測](api-256.json)、[160 tokens実測](api-160.json)、[176 tokens診断実測](fused-176.json)。

再現: READMEの環境準備後、`python3 scripts/validate_single_query.py`。
主な変更: `crates/core/build.rs`、`crates/core/src/lib.rs`。外部サービスへのデプロイは行っていない。
