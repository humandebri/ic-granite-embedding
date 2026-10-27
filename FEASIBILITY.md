# Granite Embedding Small English R2 の query canister 化

> **履歴: 実装前の設計調査です。** 現在の仕様は[README](README.md)・[API](docs/API.md)、実測は[最新レポート](benchmarks/frontier/RESULTS.md)を参照してください。
> このページの候補APIや推奨手順は当時の案です。後続で5,000 tokensを低水準に分割した実験と、現在の300 tokens / 1 query・1,365 tokens / 7 queryの公開APIをそれぞれ検証しました。

調査日: 2026-10-26。対象は `ibm-granite/granite-embedding-small-english-r2`。
ユーザーの希望に合わせ、通常の推論は query で提供する前提。

## 結論

単一 canister に重みを配置し、Rust/Wasm で推論する構成は実装候補として有望。
容量面では十分収まる。短文を1回の query で処理できる最大長は、実機相当の Wasm 計測が必要。
初期の計測対象を32・64・128 tokens、次に256・512 tokensとし、特殊トークンも長さに含める。
これらは測定条件であり、対応を保証する入力長ではない。

推奨は短文向けの単発 query を第一目標にし、必要に応じて層・演算を分割した複数 query を追加する方式。
モデル本来の8,192 tokensを、最初から単発 query の対応範囲にはしない。
初回調査時点では canister 実装・モデル実行・デプロイ・性能測定は行っていなかった。

## 確認したモデル仕様

Hugging Face revision: `2ab6fa8ea2d674564defd37171ae19079b864b33`。

| 項目 | 確認結果 |
| --- | --- |
| モデル | ModernBERT encoder、約47M parameters |
| 層 / hidden / heads | 12 / 384 / 12（head dimension 32） |
| MLP | intermediate 1,536、GeGLU |
| Attention | 双方向。0始まりの層0・3・6・9がglobal、それ以外はlocal |
| Local / RoPE | local_attention=128、global theta=80,000、local theta=10,000 |
| 最大長 / 語彙 | 8,192 / 50,368 |
| 出力 | 384次元、最終hidden stateのCLS pooling |
| 正規化 | SentenceTransformerの既定出力は未正規化。L2正規化をAPIで明示する |
| 配布重み | BF16 safetensors、95,332,048 bytes（約90.9 MiB） |
| ライセンス | Apache-2.0 |

`config.json` の `classifier_pooling=mean` を embedding pooling と解釈しない。
SentenceTransformer の `modules.json` と `1_Pooling/config.json` は CLS pooling を指定している。
embedding LayerNorm、先頭層のattention norm省略、各層のLayerNorm、最終LayerNorm、GeGLUの分割順序も参照実装に合わせる。

## ic-imajev から参考にできるもの

- `inference-core`: モデル非依存のF32行列計算、Wasm SIMD128、BF16/block256 codec。
- `weight_cache.rs`: stable memory上の重みを、管理操作で不変のheap cacheに準備する設計。
- query API: queryごとに中間状態を返し、クライアントが次のqueryへ渡す方式。
- 各ステップのinstruction counter、モデル・重み・入力・進捗を識別するプロトコル。

そのまま動く汎用モデルローダーではない。`imajev-runtime` は Qwen3.5系のINT8・LoRA・DeltaNet等に依存し、
GraniteのModernBERT graph、tensor loader、attention mask、RoPE、LayerNorm、GeGLU、poolingは別途実装する。
既存INT8 kernelにも形状・配置の前提があるため、384幅への適合と性能を確認する。
既存の32～63 queryという回数やレイテンシをGraniteに流用できない。

双方向encoderなので、decoderのように共通prefixの最終状態やKV cacheを後続入力にそのまま再利用する設計は採らない。

## query の制約と計算量

調査時に参照したICPの制限は、queryあたり50億Wasm命令、wasm32 heap 4 GiB。
通常のingress requestは2 MiB、非replicated query responseは3 MiB。
queryで変更した状態は次の呼び出しへ永続化されない。
モデルのアップロード・有効化・永続heap cacheの準備は管理用update等で行い、利用者の推論をqueryにする。

重みをF32に展開しても約191 MB程度であり、作業バッファを含めても容量には余地がある。
まずBF16由来の重みをF32で計算し、参照結果との一致を確認する。
必要に応じてINT8重み等を比較する。INT8化による検索品質の変化と、Wasm命令数の削減効果は別々に検証する。

通常の全token・全層計算で、attention本体を除いたdense projectionのMAC概算は次のとおり。
`12 × L × (4 × 384² + 3 × 384 × 1536)`。

| 入力tokens L | Dense MAC | 層境界のhidden state（F32） |
| --- | ---: | ---: |
| 32 | 約9.06億 | 48 KiB |
| 64 | 約18.12億 | 96 KiB |
| 128 | 約36.24億 | 192 KiB |
| 256 | 約72.48億 | 384 KiB |
| 512 | 約144.96億 | 768 KiB |

MAC数はWasm命令数ではない。SIMD、ロード、ループ、softmax、正規化、codec、tokenizerの影響があるため、
この表だけで単発queryの可否や秒数を断定しない。量子化してもMAC数自体は減らない。

## 推奨APIと分割方式

まず `embed_tokens(token_ids, normalize) -> embedding` のqueryを実装し、同一token列で精度・性能を測る。
利用側には `embed(text)` を提供するSDKを用意できる。
canister自体がtextを受け取るAPIも候補だが、tokenizerのWasm対応と追加命令数を確認してから追加する。
特殊トークン、空入力、範囲外ID、最大長、padding mask、切り詰めの扱いを仕様化する。
上限超過は明示エラーを基本とし、暗黙の切り詰めは避ける。

単発で収まらない場合は `begin_embed` / `step_embed` のqueryで中間状態を返す。
最初は層境界で区切り、必要ならattention・MLPをさらに分割する。
例えば1層ずつ処理する計画が成立すれば概ね12段階だが、各層が上限内かは測定で判断する。
SDKは複数queryを1つの `embed(text)` 呼び出しにまとめる。

512 tokensの層境界carryはF32で768 KiBなので、通常のquery requestに余裕を持って収まる。
途中演算のcarryは残差やQ/K/V等で増えるため、別途予算化する。
2,048 tokensのhiddenだけでも3 MiBとなり、通常の2 MiB ingressでは渡せない。
長文の完全対応には転送の分割や圧縮、演算のtile化が必要。F32からBF16への丸めはlossless codecではない。
全長attention matrixを保持しない実装も必要になる。
独立チャンクのembeddingを平均する方法は、元の全文推論とは異なるため別機能として扱う。

carryにはmodel revision・重みhash・入力識別子・層/phase・shape・format versionを持たせる。
チェックサムは破損検出用であり、第三者が中間状態を改変していない証明にはならない。
結果をブラウザ自身が使う場合と、第三者に検証可能な結果として渡す場合は信頼モデルが異なる。
通常のqueryはreplicated consensusによる結果ではない。
他canisterからの通常のinter-canister callは、ブラウザの非replicated queryと同じ実行条件にはならない。

## 実装前に決着させる検証

1. revision・重みhash・tokenizerを固定し、Python参照のtoken IDsと未正規化/正規化embeddingを保存する。
2. Rustのnative実装で層ごとの差分を確認する。64 tokens前後のlocal mask境界、global層、CLSを重点確認する。
3. 同じkernelをWasmで実行し、32/64/128/256/512 tokensの総命令数、演算別命令数、heap、応答サイズを測る。
4. 余裕を持って50億命令内に収まる単発上限を決める。超過範囲では分割queryの回数と転送量を測る。
5. INT8候補はF32参照との誤差だけでなく、検索順位・retrieval品質も比較する。
6. localの結果と区別して、必要に応じてmainnetでquery latency・並行要求・リソース制限を確認する。

## 一次資料

- [モデルカード](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- [固定revisionのconfig](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/blob/2ab6fa8ea2d674564defd37171ae19079b864b33/config.json)
- [CLS pooling設定](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/blob/2ab6fa8ea2d674564defd37171ae19079b864b33/1_Pooling/config.json)
- [ModernBERT参照実装 v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/modernbert/modeling_modernbert.py)
- [ic-imajev構成・数値契約](https://github.com/humandebri/ic-imajev/blob/main/docs/runtime/ARCHITECTURE.md)
- [ic-imajevのモデル非依存プリミティブ](https://github.com/humandebri/ic-imajev/blob/main/crates/inference-core/README.md)
- [ic-imajev query API](https://github.com/humandebri/ic-imajev/blob/main/docs/PUBLIC_QUERY_API.md)
- [ICP resource limits](https://docs.internetcomputer.org/references/resource-limits/)
