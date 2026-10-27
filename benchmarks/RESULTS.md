# 5,000 tokens INT8 query inference — ローカル実測

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](frontier/RESULTS.md)、経緯は[計測一覧](README.md)を参照してください。

2026-10-26。Granite Embedding Small English R2、12層、384次元CLS embedding。

**5,000 tokensの全文推論は、INT8版を複数queryに分割することで完了した。**
**同じ実装を単発queryで実行すると、50億命令上限で停止した。**

モデルが小さいので重みと作業メモリは収まる。しかしencoderは全文の全tokenを各層で処理し、
さらに4層のglobal attentionで全文同士を比較する。重みのサイズだけではqueryの可否は判断できない。

## 実測値

環境: Apple M6 / RAM 32 GiB / arm64 macOS、Rust 1.97.1、PocketIC server 11.0.0。
Wasm SIMD128を使用。Application subnetのProduction設定で、命令上限を緩めていない。
モデルロードを済ませてからの計測であり、モデルダウンロード・ビルド・初期ロード時間は含めない。

| 指標 | 結果 |
| --- | ---: |
| 入力 | 5,000 tokens、CLS/SEP込み |
| 実行した層 | 12 / 12 |
| 出力 | 384次元、CLS + L2正規化 |
| 推論query数 | 687 |
| 全体経過時間 | 74.33秒 |
| query呼び出しの経過時間合計 | 73.94秒 |
| 総命令数（各query内の計測値合計） | 1,123,612,208,589 |
| 最大query命令数 | 2,889,127,550（上限5,000,000,000） |
| request総量 | 879,970,840 bytes |
| response総量 | 468,513,100 bytes |
| 往復合計 | 約1.35 GB（約1.26 GiB） |
| 最大request | 2,067,500 bytes（2 MiB未満） |
| 最大response | 1,769,518 bytes（3 MiB未満） |
| F32参照とのコサイン類似度 | 0.9996813864 |
| 独立したINT8参照とのコサイン類似度 | 0.9997755559 |

初回の完走は78.73秒、最終計測は74.33秒。両方で出力vector・命令数・query回数・転送量が一致した。
ネットワーク越しのmainnet速度は測っていない。localhostの往復時間を、公開サービスのレイテンシとして扱わない。

単発queryは実際に次のエラーで停止した。シミュレーション上限を増やして完走させる方法は使っていない。

```text
Canister exceeded the limit of 5000000000 instructions for single message execution.
CanisterInstructionLimitExceeded
```

これは今回の実装についての結果であり、あらゆる最適化に対して不可能だという証明ではない。

## queryの分け方

token方向のtileを384に設定。
QKV projectionとoutput projection/MLPをtoken tileごとに計算する。
global attentionはheadを3つずつに分け、各queryが5,000 tokens全体のK/Vを受け取る。
local attentionでは必要なhaloを含めてK/Vを渡す。
クライアントは各層の全文hidden stateを保持し、12層分を順に実行する。

文書を独立した短文に分けたり、短文embeddingの平均で代用したりしていない。
global attentionが見るtoken数を削減していない。
中間状態を保存するupdateも、推論中のupdateも使っていない。

| 処理 | query数 | 合計命令数 | 最大query命令数 |
| --- | ---: | ---: | ---: |
| Embedding lookup / norm | 14 | 74,975,318 | 5,757,929 |
| QKV projection / RoPE | 168 | 90,683,316,028 | 581,023,050 |
| Attention | 336 | 634,282,634,772 | 2,889,127,550 |
| Output projection / MLP / residual | 168 | 398,571,247,690 | 2,579,200,494 |
| CLS final norm / L2 | 1 | 34,781 | 34,781 |

命令数は `step` 内の `performance_counter(0)` の差分。Candidの入出力処理の全量は含まない。
実際のquery自体にもPocketICの50億命令制限が適用され、全queryが完了している。
転送量はハーネスで扱ったCandid request/response bytesの合計であり、HTTPヘッダー・署名等を含まない。

## 数値の検証

Linear重みは出力channelごとのINT8、活性値はtokenごとの動的INT8。
重みのINT8値をi16へ準備し、SIMDのdot productでi32へ累積する。
Embedding tableとLayerNorm等はF32。K/Vのquery間転送はBF16に丸める。

元の配布重みをF32に展開したHugging Face `AutoModel` を参照にした。
さらに、同じexport済みINT8重み・動的活性量子化・BF16 K/V丸めを使うPyTorch実装とも比較した。
モデルrevisionと重みhashは固定し、参照計算ではdropoutを無効にしている。

32 tokensでは単発queryが3,214,683,332命令で完了。
これを8 tokensずつに分割した197 queryの結果は、単発queryと**全384要素で完全一致**した。
F32参照とのコサイン類似度は0.9995484591。
Wasmとnative間、WasmとPyTorch間のbit一致はない。浮動小数点差と量子化の丸め境界の影響がある。

サンプルは英語の技術説明文を反復して作った、正確に5,000 tokensの入力1件。
入力上限・数値差・完走の検証であり、多様な長文や検索タスクでの品質評価ではない。
コサイン類似度を、そのまま検索精度の維持率と解釈しない。

追加の3つのunit testも通過した。

- 最大INT8値を含むdot productのi32累積。
- BF16のties-to-even丸め。
- global attentionの全文参照と、local attentionのinclusiveな前後64 tokens境界。

## 実用化に向けた判断

5,000 tokensをquery経由で処理できることは確認できた。
現在の687 query・約1.35 GBの転送は、ブラウザ向けサービスとしては負担が大きい。
次の改善対象はattentionの命令数（全体の約56%）と、K/Vの繰り返し転送。
K/VのINT8転送、wire formatとhead/token tileの再設計、kernel最適化を測定しながら比較する価値がある。
量子化やattention計算法を変更する場合は、その都度数値とretrieval品質を確認する。

単一queryで5,000 tokensを返すことを最初の到達点にするより、
まず全文推論のままquery回数・転送量・時間を減らす方針が、この実測では妥当。

## 生データと再現情報

- [5,000 tokensの測定JSON](pocketic-5000.json)
- [32 tokensの単発/分割比較](pocketic-32.json)
- [参照結果、model revision、重み/pack hash](references.json)
- [環境、Wasm hash、ソースhash](environment.json)
- [再実行手順](../README.md)

モデルweightsはこのディレクトリに複製していない。
