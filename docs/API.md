# モデルのロードとquery API

このページは現在のdefault feature構成を対象とします。
インターフェースの型定義は[granite.did](../crates/canister/granite.did)を正とします。

## モデルを準備する

Wasmに重みは含まれません。[開発手順](DEVELOPMENT.md#モデルと参照データを準備する)で生成した`.cache/model.pack`を、controllerからロードします。

| メソッド | 種別 | 引数と結果 |
| --- | --- | --- |
| `upload` | update、controllerのみ | `(offset: nat64, bytes: blob)`。追記位置は現在の受信済み長と一致させる |
| `activate` | update、controllerのみ | 引数なし。`variant { Ok; Err: text }` |

`upload`は1回1,900,000 bytes以下、PACK全体は160,000,000 bytes以下です。
`activate`は固定モデルの構造・INT8範囲などを検査し、重みの展開・Strassen用データ・RoPEをheapに準備します。
成功後は受信PACKを解放します。実行済みのロード処理は[検証ハーネス](../crates/bench/src/main.rs)にもあります。

以下は、Candidから構築した`actor`と`Uint8Array`の`modelPack`がある場合の初回ロード例です。
このリポジトリにactor接続処理やSDKは含めていません。

```js
function unwrap(result) {
  if ("Err" in result) throw new Error(result.Err);
  return result.Ok;
}

async function loadModel(actor, modelPack) {
  const chunkBytes = 1_800_000;
  for (let offset = 0; offset < modelPack.byteLength; offset += chunkBytes) {
    await actor.upload(
      BigInt(offset),
      modelPack.subarray(offset, offset + chunkBytes),
    );
  }
  unwrap(await actor.activate());
}
```

初回ロード用の例で、自動retry・途中アップロードの再開・運用上のモデル切替は扱いません。
モデル有効化前の推論は`Err`になります。

## 入力

モデルは`ibm-granite/granite-embedding-small-english-r2`、revisionは`2ab6fa8ea2d674564defd37171ae19079b864b33`に固定しています。
同じrevisionのtokenizerを使い、CLS/SEP込みで2〜1,365個のtoken IDを渡します。

| 項目 | 条件 |
| --- | --- |
| 先頭 | CLS `50281` |
| 末尾 | SEP `50282` |
| padding | PAD `50283`を含めない |
| 語彙範囲 | `0..50368`（上端を含まない） |
| textの受付 | 未実装。tokenizerはクライアント側で実行する |

上限超過や不正なtoken列は`Err`となり、暗黙の切り詰めは行いません。

## 推論を開始して継続する

```candid
startEmbedding : (vec nat32) -> (variant { Ok : EmbeddingStep; Err : text }) query;
continueEmbedding : (nat32, nat32, blob) -> (variant { Ok : EmbeddingStep; Err : text }) query;
```

| `EmbeddingStep`のfield | 意味 |
| --- | --- |
| `tokens: nat32` | 最初に渡したtoken数。継続時も同じ値を渡す |
| `next_layer: nat32` | 次に処理する層。`12`なら完了 |
| `state: blob` | 継続用のopaqueなstate。完了時は空 |
| `embedding: vec float32` | 完了時の384要素。途中では空 |
| `instructions: nat64` | メソッド内で測った命令数。JavaScriptのCandid bindingでは通常`bigint` |

304 tokens以下は`startEmbedding`だけで完了します。
それ以上は返された`tokens`・`next_layer`・`state`をそのまま次の呼び出しに渡します。
`next_layer`を1ずつ増やす実装にはせず、返された進捗を使ってください。

```js
async function embedTokens(actor, tokenIds) {
  let result = unwrap(await actor.startEmbedding(tokenIds));
  while (result.next_layer < 12) {
    result = unwrap(await actor.continueEmbedding(
      result.tokens,
      result.next_layer,
      result.state,
    ));
  }
  return result.embedding;
}
```

結果は最終CLSのLayerNorm後、L2正規化した384次元embeddingです。
入力全体が処理対象で、別々の短文embeddingを合成しているわけではありません。
長さ別のquery数は[READMEの表](../README.md#対応する入力長)を参照してください。

## stateと測定値の扱い

stateはGRC1形式ですが、クライアントは中身を解釈せずblobのまま扱います。
defaultの転送形式は可逆なraw F32です。decoderは以前の圧縮形式にも対応していますが、「常に圧縮済み」とは限りません。
推論ごとにstateを分け、別入力・別モデル・別moduleのstateを混ぜないでください。
状態の永続保存や、改変されていないことの認証は行っていません。

`instructions`はメソッド内の`performance_counter(0)`の差分です。
Candidの入出力処理全体を含む値ではありません。検証では値の比較に加えて、PocketICの実際のquery上限内で呼び出しが成功することも確認しています。
転送サイズの測定はCandid payloadのbytesで、HTTPヘッダーや署名などは含みません。

## 診断用メソッド

通常ビルドにも`step(blob)`があります。token/層/演算を指定する低水準のバイナリプロトコルで、付属ハーネスの診断用です。
1,365 tokensまでは`startEmbedding` / `continueEmbedding`を使います。1,366〜5,000 tokensは[分割queryクライアント](SHARDED.md)が`step`を呼び出します。

`profiling` featureでビルドすると`profileStep`が追加されます。
通常の[granite.did](../crates/canister/granite.did)には含まれません。計測方法は[開発手順](DEVELOPMENT.md#計測版を再実行する)を参照してください。
