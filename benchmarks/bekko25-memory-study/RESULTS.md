# Bekko a25m 容量と精度の調査

2026-10-30。対象は commit `148878f2b94d18dceb7abb73fb2075e6a6a4189f` の Bekko a25m。推論コードや canister は変更していない。モデル pack SHA-256 は `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`。

追記: この調査後、分割読み込みを実装し、旧版からのupgradeとメモリ・出力・命令数を[別途測定した](../bekko25-stream/RESULTS.md)。この文書内の「未測定」は当初調査時点を示す。

## 容量

pack は 418,404,508 bytes。このうち 256,000 × 384 個の F32 語彙表が 393,216,000 bytes（94.0%）。その他は 25,188,508 bytes。既存のローカル検証では stable 418,447,360 bytes、Wasm heap 985,595,904 bytes を測定した。[元の運用検証](../bekko25-operations/RESULTS.md)

| 方式 | pack概算 | 既存pack比 | 数値 |
|---|---:|---:|---|
| 現行 F32 語彙表 | 418.4 MB | — | 基準 |
| 語彙表の byte shuffle + zlib level 6 | 354.7 MB | -15.2% | **全393,216,000 bytes の復元一致を確認**。ブロック管理情報を除く |
| FP16 または BF16 語彙表 | 221.8 MB | -47.0% | 丸めにより変化。値を F32 に展開してから推論する想定 |
| 行単位 INT8 + F32 scale | 124.5 MB | -70.2% | 丸めにより変化。値を F32 に展開してから推論する想定 |

圧縮は語彙表を4 MiBごとに分け、各ブロックの4バイトを byte shuffle して zlib level 6 で圧縮し、全94ブロックの伸長・逆変換を元バイトと比較した。語彙表の圧縮後は329,496,221 bytes。これはPythonでの保存形式実験であり、Wasmでの解凍命令数・安定メモリ形式・起動時間は未検証。[測定結果](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/storage.json)

語彙表の全98,304,000値のうち、FP16変換後にF32へ戻して完全一致したのは12,120値、BF16で下位16 bitがゼロだったのは2,577値。FP16の表内の最大絶対誤差は `0.00019073486328125`。これらは出力 embedding や検索品質への誤差とは異なる。[測定結果](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/storage.json)

現在は `activate_model` がpack全体をheapへ読み、その参照から最終のF32語彙表等を別に確保する。stableからテンソル単位で最終配列へ読み込めば、同じF32値と推論コードを使いながら、この全packの一時コピーを避けられる可能性がある。**最大418 MBは削減上限の見積もりであり、実際のheap削減量・有効化の命令数・出力一致・推論性能は未測定**。Wasmメモリは確保後に通常縮まないため、単に既存の一時バッファを解放するだけでは効果を見込めない。

## 出力と検索への影響

既存packのうち、評価入力で使用するtoken IDの語彙行だけをFP16/BF16に丸め、または行ごとの最大絶対値をscaleにした対称INT8に丸めてからF32へ展開した。語彙表以外と `Encoder::full` は同じ。全表を量子化した場合も、今回の入力については参照する行が同じになる。**実際の圧縮packやWasmへの実装は行っていない。**

NanoSciFactの英語・アラビア語は既存評価と同じ各50問。日本語はJaQuADから50問・50文書、同一記事5段落ずつ×10記事を選び、質問と文書を256 tokens以内で切り詰めずに収めた。各質問のラベルは元段落1件のみで、ほかの関連段落は判定していない。日本語の別セットは50記事から1段落ずつ。英語・アラビア語の現行native Recall@5は保存済みWasm実測の0.94 / 0.86と一致した。

| 方式 | 英語 Recall@5 | アラビア語 Recall@5 | 日本語・同記事候補 Recall@5 | 同記事日本語のtop-5集合変更 |
|---|---:|---:|---:|---:|
| 現行 | 94% | 86% | 100% | 基準 |
| FP16 | 94% | 86% | 100% | 4 / 50問 |
| BF16 | 94% | 86% | 100% | 5 / 50問 |
| INT8 | 94% | 86% | 100% | 3 / 50問 |

同記事日本語の正解top-1件数は現行41、FP16 42、BF16 43、INT8 42。改善を主張できる標本数ではない。日本語Recall@5は全方式で飽和しており差の検出力が低い。英語のtop-5集合はFP16 14問、BF16 11問、INT8 17問で変化。アラビア語はそれぞれ13、10、16問。最小cosine（現行出力との比較）は同記事日本語でFP16 0.996857、BF16 0.996209、INT8 0.995986。数値の完全一致や、未知のコーパスでの精度維持は保証できない。[英語・アラビア語](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/retrieval-summary.json)・[日本語同記事](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/retrieval-ja-hard-summary.json)・[日本語別記事](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/retrieval-ja-summary.json)

現行ベクトルの索引に量子化後の質問だけを混在させた比較でも、各50問のRecall@5は変わらなかったが、top-5の集合は英語・アラビア語で7〜14問、日本語同記事で2〜5問変わった。既存の `embedding_space` はモデルの数値仕様を含むため、量子化を採用するなら別IDとして扱い、索引を再構築する。[混在比較](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/cross-index.json)・[日本語混在比較](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-memory-study/cross-index-ja.json)

CPU実行時間は並列処理・無制御環境のため性能比較に使えない。実機Wasmのquery命令数・全255入力長・実利用文書の検索・mainnetの性能とcycles・アップグレード復元は未検証。現行Wasmの最大観測は約4.844B / 5B instructionsで余裕は約3.1%。[既存の上限検証](../bekko25-operations/RESULTS.md)

## 次の候補

1. 推論結果の完全一致を優先するなら、stableからテンソルごとに最終heapへ読む試作を作り、実Wasmでheap、全入力長の命令数、既存golden vector、uploadとupgrade復元を検証する。
2. さらにstableを減らすなら、上記可逆圧縮を段階的に展開する試作を検証する。起動処理に追加コストがある。
3. より大きく減らすならINT8/FP16/BF16表を別embedding spaceとして評価する。実Wasmでの命令数と、より大きく難しい日本語を含む実利用データでの検索品質が必要。

再現手順と検証プログラムは [experiments/bekko25-memory-study](../../experiments/bekko25-memory-study/README.md)。計測用vectorは `.cache/bekko-memory-study/` に置き、集計だけをこのディレクトリに保存した。既存のmainnet/local canisterは変更していない。
