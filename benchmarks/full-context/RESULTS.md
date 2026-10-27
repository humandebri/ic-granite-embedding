# 全射影の最適化と5,000 tokensの分割query

2026-10-27。通常APIの1 query上限を300から304 tokensへ拡張しました。実測では309まで成功し、310〜320は実際の50億命令制限で拒否されました。
1,366〜5,000 tokensには、クライアントでstateを保持する分割query経路を追加しました。5,000 tokensは182 queriesです。

## 採用した変更

- QKV・Attention出力射影にも1段の整数Strassenを適用。Wi/Woと同様に整数で復元してから元のF32 scaleを適用する。
- SIMD kernelの出力channel方向を16-channelの生成bodyにまとめ、入力ロードの共有を維持してWasmのコード量を削減する。
- キー長512以下の全域Attentionで4つのQ行にKのロードを共有する。長文のコンパイル結果を維持するため、同じ行kernelのソースを通常経路とbatch経路で別々にコンパイルする。
- K/Vを既存のBF16形式で返し、context INT8・局所AttentionのQ INT16への既存の量子化をクライアントへ移す。
- 5,000 tokensの全域Attentionは1 head/query、局所Attentionは2 heads/queryに分ける。V scaleを維持するため、各head群に全入力のK/Vを渡す。
- 最後の3層ではCLSに必要な129→65→1行へ絞る。state全体を2 MiBのrequestへ再梱包せず、1,000行のQKV・1,088行のfinish単位で送る。

数値精度を下げる追加近似は導入していません。Candidメソッド型も維持しました。
通常APIは1,365 tokensまで、追加の分割クライアントは5,000までです。[プロトコルと実行方法](../../docs/SHARDED.md)を参照してください。

## 通常版の前後比較

直前の整理版との比較です。命令数は`startEmbedding`による256 tokensの公開API、時間は通常版の`step`で全12層を1 queryで実行した値です。

| 指標 | 直前 | 採用版 |
| --- | ---: | ---: |
| 256 tokens命令数 | 4,166,069,524 | 4,064,727,088（−2.43%） |
| 1 queryの公開上限 | 300 | 304 |
| 1 queryの実測最大成功 | 302 | 309 |
| Wasm bytes | 6,152,406 | 3,602,937（−41.44%） |
| 256 tokens中央値 | 0.593秒 | 0.490秒（約17.4%短縮を観測） |
| activate直後のWasm線形メモリ | 326,631,424 bytes | 351,272,960 bytes |

新しいStrassen重みを準備するため、線形メモリは約23.5 MiB増えます。上記は線形メモリの確保量で、使用中のmodel allocationだけを示す値ではありません。

1 queryの境界は289〜320の各長×3入力で測定しました。公開範囲2〜304の全長走査では303 tokensの4,902,457,898命令が最大で、50億に対し約1.95%の余裕があります。
309の成功は余裕が小さいため公開上限にはしていません。[境界データ](../validation/capacity.json)と[全511長](../validation/regression/sweep.json)を確認できます。

時間比較は全ビルド・他のsuite終了後に4文書×5回を順次実行し、未使用fieldのnonceでquery cacheを避けました。
[比較条件と20件ずつの記録](latency-comparison.json)を保存しています。OS・背景プロセス・熱状態は未制御で、mainnetのレイテンシ保証ではありません。

## 5,000 tokens

| case | queries | 合計命令 | 最大/query | cosine対FP32 |
| --- | ---: | ---: | ---: | ---: |
| 1365 | 67 | 28,016,999,689 | 1,352,995,314 | 0.999610393 |
| 1366 | 70 | 28,060,337,444 | 1,171,880,309 | 0.999626836 |
| 2048 | 90 | 47,646,038,552 | 1,497,365,560 | 0.999556149 |
| 4096 | 173 | 130,193,977,969 | 1,974,470,640 | 0.999641624 |
| 5000 | 182 | 177,813,740,538 | 2,936,313,152 | 0.999669377 |
| space-5000 | 182 | 177,813,740,538 | 2,936,313,152 | 0.999642830 |
| cooking-5000 | 182 | 177,813,740,538 | 2,936,313,152 | 0.999508387 |
| weather-5000 | 182 | 177,813,740,538 | 2,936,313,152 | 0.999698703 |

1,365の分割クライアント実行は通常APIとのbit一致を調べる比較用です。1,365以下では7 query以下の通常APIを使います。

5,000 tokensの内訳はembedding 3、QKV 52、Attention 78、finish 48、最終正規化1で、合計182です。
最大requestは2,093,356 bytes、最大replyは3,072,047 bytesで、2 MiB / 3 MiBの上限内です。
モデルupload・activateと不正入力検査はinference query数に含めません。

初版の[5,000 tokens記録](initial-5000-metrics.json)との累積比較は次のとおりです。初版以降の整数Attention・CLS依存範囲削減なども含む比較で、今回の差分だけの効果ではありません。

| 指標 | 初版 | 現在 |
| --- | ---: | ---: |
| queries | 687 | 182（約73.5%削減） |
| 合計命令 | 1,123,612,208,589 | 177,813,740,538（約84.2%削減） |
| Candid request/reply合計bytes | 1,348,483,940 | 589,677,652（約56.3%削減） |

転送量は約590 MB残ります。HTTP/TLSやモデルuploadを含まない値です。
長文suiteの時間は並列検証中の記録であり、上の256 tokensの順次測定との速度比較には使いません。

## 検証

- Rust unit 12ケース。
- 独立したnativeの密行列計算と線形演算112ケースでhash一致。
- QK 44ケースで旧SIMD経路・scalar i64とbit一致。±8191の極値とkey長の余りを含む。
- AV 44ケースでi64参照と一致。512/513境界・5,000 keys・±2047を含む。
- 2〜512の全511長、追加57 stress、公開API 16ケースで既存embeddingとbit一致。
- 513〜1,365の32長×3入力＝96ケース、FP32参照11ケースで既存embeddingとbit一致。
- 分割クライアント8ケースで有限・384次元・L2 norm・FP32 cosine > 0.999・サイズと命令上限を確認。
- 5,000のパック経路は現行演算を使ったパック前275-query経路とbit一致。初版687-query経路とのbit一致を意味しない。
- 1,365の分割経路と既存の通常APIはbit一致。不正request 9種類の拒否も8ケースそれぞれで検証。
- 通常Candid維持、通常Wasmに`profileStep`がないことを確認。

[source・通常Wasmの検証](../validation/environment.json)、[分割経路のfixture SHAと検証](sharded/environment.json)、[新しい参照生成のmodel revision・pack SHA](sharded/references.json)を保存しました。
[QK](qk.json)・[AV](av.json)・[線形演算](linear.json)・[12層のprofile](profile-1365.json)は、別途作った同一profiling Wasmによる測定です。
これらのprofiling値を通常Wasmの削減量に加算していません。retrievalのRecall@kやmainnet運用は未検証です。

## 採否とボトルネック

候補ごとのSHA・命令数は[experiments.json](experiments.json)にあります。
4-channel生成bodyは完全展開より命令が増えたため、より良い最終構成を選びました。
QKを全キー長に適用する案・短文だけを同じ関数群で処理する案・明示的inlineを増やす案も比較し、長文で増えたものは不採用としました。

短文batchを追加した際、profiling版の長文phaseはほぼ同じでも通常版の長文命令数が増えました。
共有関数のコンパイル・inline展開の影響と考えられますが、IRレベルの因果特定はしていません。
行kernelのソースを共有しつつコンパイルする関数群を分けると、5,000の差は177,813,739,982→177,813,740,538（+556命令）になり、短文の改善を残せました。
不採用の実装・切替用環境変数は製品ソースに残していません。

長文の大きなコストは全域Attentionです。5,000 keys・1 headのprofilingではQK、AV、softmaxが主要部分を占めます。
[通常版の層別・QKV/Attention/finish内訳](sharded/5000.json)と[Attention単体のphase](attention-profile.json)を分けて記録しています。
転送量の削減と、query間で再送する中間値の削減が残る課題です。

## 成果物

- 通常Wasm: `6c56483bfd1b669971b15a76f2c2a58491593fed28988759822ad3eec84679b3`
- profiling Wasm: `aa94460718a441833b3cba1ec1d47c8ced792df9f69e9d0f0100976a2a08022c`
- [artifactのパス・bytes・SHA](artifacts.json)
- [README](../../README.md) / [開発・再検証](../../docs/DEVELOPMENT.md) / [分割プロトコル](../../docs/SHARDED.md)
