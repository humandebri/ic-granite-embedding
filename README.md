# Granite Small English R2 query canister

[Granite Embedding Small English R2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)をInternet Computerのqueryで実行する、固定モデル向けのRust/Wasm実装です。
INT8重みと動的INT8活性値（W8A8）を使い、CLS pooling・L2正規化済みの384次元embeddingを返します。
入力全体を参照して全12層を実行し、独立した短文embeddingの平均で全文推論を代用しません。

**通常APIは304 tokensまで1 query、1,365 tokensを7 queryで処理します。追加の分割クライアントは5,000 tokensを161 queryで処理します。**
長さには先頭CLSと末尾SEPを含みます。通常の推論はすべてqueryで、モデルのロードだけcontroller用updateを使います。

- [API・モデルのロード・クライアント例](docs/API.md)
- [ビルド・参照データ生成・検証・profiling](docs/DEVELOPMENT.md)
- [5,000 tokensの分割クライアント](docs/SHARDED.md)
- [最新の性能・検証](benchmarks/query-fusion/RESULTS.md) / [計測の履歴](benchmarks/README.md)

多言語の **Bekko a8mを256 tokens・1 queryで使う専用canister** も追加しています。[Bekkoの起動・API・検証結果](docs/BEKKO.md)を参照してください。

## 対応する入力長

| 入力tokens（CLS/SEP込み） | query数 | 実行する層数の分割 |
| --- | ---: | --- |
| 2〜304 | 1 | `[12]` |
| 305〜512 | 2 | `[5, 7]` |
| 513〜768 | 4 | `[3, 3, 3, 3]` |
| 769〜1,024 | 5 | `[2, 2, 2, 2, 4]` |
| 1,025〜1,365 | 7 | `[1, 2, 1, 2, 1, 2, 3]` |

各planは合計12層です。複数queryでは、返された中間stateをクライアントが次のqueryへ渡します。
1 queryの境界を複数入力で測定し、公開上限は304にしています。境界での成功・失敗は[最新の性能レポート](benchmarks/query-fusion/RESULTS.md)を参照してください。
`startEmbedding`は1,365を超える入力を拒否し、暗黙に切り捨てません。1,366〜5,000 tokensは[分割query経路](docs/SHARDED.md)を使います。

1,365 tokensの最大requestは2,096,674 bytesでした。
1,366 tokensでは中間stateのF32値だけで2,098,176 bytesとなり、2 MiBを超えます。
この上限は現在のstate転送形式に対するもので、モデル自体の最大長を示す値ではありません。

## ローカル実測と検証

2026-10-27、PocketIC server 11.0.0、Application subnet / Productionで測定しました。
実際の50億命令/queryの上限を適用し、Benchmarking設定による緩和はしていません。

| tokens | query数 | 合計命令 | 最大/query |
| --- | ---: | ---: | ---: |
| 256 | 1 | 4,064,748,620 | 4,064,748,620 |
| 304 | 1 | 4,884,221,016 | 4,884,221,016 |
| 512 | 2 | 8,351,268,853 | 4,431,648,644 |
| 768 | 4 | 13,293,671,842 | 4,222,022,185 |
| 1,024 | 5 | 18,640,233,102 | 4,353,733,502 |
| 1,365 | 7 | 26,670,559,915 | 4,322,917,417 |
| 5,000（分割client） | 161 | 164,741,174,056 | 2,628,934,924 |

304 tokensは固定seedの走査入力、ほかはtokenizerで生成した文書です。
5,000 tokensの順次1回比較では、前版19.05秒・今回18.85秒でした。
速度差の再現性は未検証で、OS・熱状態も未制御です。mainnetのレイテンシを示す値ではありません。
5,000 tokensの転送はCandid request/reply合計約513 MBです。

旧入力域2〜512の全511長は前版と出力がbit単位で一致しました。分割clientの境界54ケースと、680行replyの上限2ケースも照合しています。
長文は32長×3入力の96ケースとFP32参照11ケースを検証し、長文参照との最小cosineは0.999592611でした。
これは検索のRecall@kなどを測ったretrieval benchmarkではありません。
5,000 tokensは4文書を検証し、FP32との最小cosineは0.999508387でした。最新のソース・Wasm・分割方式は[性能レポート](benchmarks/query-fusion/RESULTS.md)にまとめています。

## ビルドして使う

Rustは[rust-toolchain.toml](rust-toolchain.toml)で1.97.1に固定しています。リポジトリのルートで実行します。

```sh
cargo build --release --locked -p granite-canister --target wasm32-unknown-unknown
```

出力は`target/wasm32-unknown-unknown/release/granite_canister.wasm`です。
Wasmに重みは含めていません。利用するには次の順で準備します。

1. [参照データの準備手順](docs/DEVELOPMENT.md#モデルと参照データを準備する)で`.cache/model.pack`を生成する。
2. 利用先にWasmをインストールし、controllerとして`upload`と`activate`を呼ぶ。
3. 固定revisionのtokenizerでID列を作り、`startEmbedding`から開始する。

モデルのロードと推論の呼び出し例は[APIドキュメント](docs/API.md)を参照してください。
デプロイ先の作成・identity設定は利用側で用意します。付属の検証ハーネスは専用のPocketIC環境を作成し、ロードからqueryまで実行します。

## 実装の構成

| パス | 役割 |
| --- | --- |
| [crates/core](crates/core) | 固定モデルのloader、W8A8計算、Attention、query plan、state codec |
| [crates/core/build.rs](crates/core/build.rs) | 固定形状のWasm SIMD kernel生成 |
| [crates/canister](crates/canister) | controller用ロードAPIとquery API |
| [crates/canister/granite.did](crates/canister/granite.did) | 通常ビルドのCandid |
| [crates/bench](crates/bench) | native参照・PocketIC検証・計測ハーネス |
| [scripts](scripts) | 重みexport、FP32/INT8参照生成、検証、profile集計 |
| [benchmarks](benchmarks/README.md) | 段階別の測定記録と検証データ |

重みのINT8化はオフラインで行い、i16展開・Strassen用の重み和差・RoPEのsin/cosは`activate`で準備します。
入力に依存する活性値の量子化はquery中に行います。
QKV・Attention出力・Wi・Woには整数で復元する1段Strassenを使い、長文AttentionのAVは512-keyブロックをi32で計算してi64で合算します。
GELU近似やK/VのBF16丸めを含む数値仕様・feature構成は[開発ドキュメント](docs/DEVELOPMENT.md#数値仕様とfeature)に記載しています。

## 現在の範囲

ローカルPoCで、mainnetにはデプロイしていません。
重みはheapに保持するため、upgrade後の復元手順・永続化は未実装です。ブラウザSDKやcanister内tokenizerも含みません。
query結果とクライアントが返すstateに、replicated consensusや改変防止の保証はありません。
stateは推論ごとに分け、モデルやmoduleの更新をまたいで再利用しないでください。

`startEmbedding` / `continueEmbedding`の対応上限は1,365です。追加の[分割クライアント](docs/SHARDED.md)は`step`を使い、5,000 tokensを処理します。
[初版の687 query](benchmarks/RESULTS.md)から、現在は161 queryへ削減しました。

## 参照

- [固定model revision](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/tree/2ab6fa8ea2d674564defd37171ae19079b864b33)
- [参考にしたic-imajev](https://github.com/humandebri/ic-imajev)
- [初回の設計調査](FEASIBILITY.md)（履歴）
