# 計測結果の入口と履歴

現在の公開API・default feature構成は、[README](../README.md)と[現在の実装の検証](query-fusion/RESULTS.md)を参照してください。
下の履歴には、当時のquery plan・Wasm・転送形式・計測方法が含まれます。現在の上限として読み替えないでください。

## 最新の検証データ

[最新の性能・5,000 tokens検証](query-fusion/RESULTS.md)と[現在のsource・Wasm検証](validation/environment.json)が最新です。
以下のfrontierは直前の最適化・長文対応の性能記録です。

| 記録 | 内容 |
| --- | --- |
| [query-fusion/RESULTS.md](query-fusion/RESULTS.md) | query境界融合と全域Attention batch、5,000 tokens / 161 queries |
| [query-fusion/sharded/summary.json](query-fusion/sharded/summary.json) | 新しい分割経路の8参照ケース |
| [full-context/RESULTS.md](full-context/RESULTS.md) | 全射影Strassen、kernel縮小、5,000 tokensの分割query |
| [full-context/sharded/summary.json](full-context/sharded/summary.json) | 1,365〜5,000 tokensの8参照ケース |
| [frontier/RESULTS.md](frontier/RESULTS.md) | 300 tokens / 1 query、1,365 tokens / 7 query、命令数・速度・上限の比較 |
| [frontier/final/environment.json](frontier/final/environment.json) | 通常WasmのSHA256、feature、source SHA、検証範囲 |
| [frontier/artifacts.json](frontier/artifacts.json) | 検証した通常版・計測版のパスとSHA256 |
| [frontier/final/regression/sweep.json](frontier/final/regression/sweep.json) | 旧入力域2〜512の全511長 |
| [frontier/final/stress-long.json](frontier/final/stress-long.json) | 長文32長×3入力、計96ケース |
| [frontier/final/summary.json](frontier/final/summary.json) | 長文11参照ケースのFP32比較・query数・転送量 |
| [frontier/final/capacity.json](frontier/final/capacity.json) | 289〜320の各長×3入力、1 query境界の96ケース |
| [frontier/av-final.json](frontier/av-final.json) | AVの512/513境界などをi64参照と比較した44ケース |
| [frontier/latency-comparison.json](frontier/latency-comparison.json) | ビルド・他の測定が終了した後の、キャッシュを避けた順次比較 |

検証suite内の時間と、性能比較用の時間は分けています。
sourceやWasmが異なる記録を比較する場合は、各environment・SHA・featureを確認してください。
`target/`の標準ファイル名は再ビルドで変わるため、履歴のSHAが現在のファイルと同じとは限りません。

## これまでの段階

新しい段階から並べています。本文の数値・成功/未達の判断は当時のものです。

| 日付 | 記録 | 当時の結果・目的 |
| --- | --- | --- |
| 2026-10-27 | [query-fusion](query-fusion/RESULTS.md) | 182→161 queries、転送約77 MB削減、長文命令数7.35%減 |
| 2026-10-27 | [full-context](full-context/RESULTS.md) | QKV・Attention出力のStrassen、kernelのコード量削減、5,000 tokens / 182 queries |
| 2026-10-27 | [cleanup](cleanup/RESULTS.md) | 旧runner・不採用feature・再帰Strassenを削除。短文・長文のbit一致を検証 |
| 2026-10-27 | [frontier](frontier/RESULTS.md) | WoのStrassen、長文AVのi64合算。300 / 1 query、1,365 / 7 query |
| 2026-10-27 | [default-adoption](default-adoption/RESULTS.md) | 命令数優先版を既定へ採用。256 tokensは約42.31億命令 |
| 2026-10-26 | [tile-optimization](tile-optimization/RESULTS.md) | タイル拡大と早期書き出しを比較し、命令数優先の選択肢を追加 |
| 2026-10-26 | [exhaustive-optimization](exhaustive-optimization/RESULTS.md) | 288 / 1 query、512 / 2 queryへ拡張。全511長を走査 |
| 2026-10-26 | [further-optimization](further-optimization/RESULTS.md) | 256 tokensを約44.93億命令へ削減 |
| 2026-10-26 | [goal-run](goal-run/RESULTS.md) | 256 tokens / 1 queryを達成 |
| 2026-10-26 | [single-query](single-query/RESULTS.md) | 当時は256 / 1 queryが未達。上限と精度を比較 |
| 2026-10-26 | [efficiency](efficiency/RESULTS.md) | 末尾の計算削減と圧縮。256 / 2 query、512 / 5 query |
| 2026-10-26 | [optimization](optimization/RESULTS.md) | 512 tokens / 6 queryのAPIを検証 |
| 2026-10-26 | [初版](RESULTS.md) | 低水準の演算分割で5,000 tokensを687 queryで実行。現在の公開APIとは別方式 |

[FEASIBILITY.md](../FEASIBILITY.md)は実装前の設計調査です。
ビルド・参照生成・再測定は[開発手順](../docs/DEVELOPMENT.md)を参照してください。
