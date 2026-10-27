# 5,000 tokensの分割query

`startEmbedding` / `continueEmbedding`は1,365 tokensまでです。それ以上はクライアントがhidden stateとQKVを保持し、通常の`step` queryを呼びます。全12層を実行し、全域Attentionは入力全体を参照します。

[参照クライアント](../crates/bench/src/sharded.rs)はPocketICによるロード・測定を含むハーネスです。Candidメソッド型は変えていません。

## 実行

[開発手順](DEVELOPMENT.md)に従って依存・`POCKET_IC_BIN`を準備します。

```sh
cargo build --release --locked -p granite-canister --target wasm32-unknown-unknown
cargo build --release --locked -p granite-bench
.cache/venv/bin/python scripts/prepare.py --lengths=256,5000
target/release/granite-bench 5000 --sharded
```

通常版を指定すると、専用PocketICへのインストール・モデルロード・推論まで実行します。`GRANITE_WASM`でWasm、`CARGO_TARGET_DIR`でビルド先を指定できます。

8ケースを検証する場合は、通常の[参照生成](DEVELOPMENT.md#モデルと参照データを準備する)に加え、次のfixtureを用意します。

```sh
mkdir -p .cache/next/references
.cache/venv/bin/python scripts/prepare.py --output .cache/next/references \
  --lengths=1366,2048,4096,5000 --stress --stress-lengths=5000
# tokens-*.json / reference-*.f32 / reference-int8-*.f32を.cache/にも配置。
# 既存ファイルは内容の一致を確認する。1,365の既存fixtureも必要。
python3 scripts/validate_sharded.py
```

`--summarize-only`は保存済みの同一SHAの結果を再検査します。結果は`benchmarks/query-fusion/sharded/`に保存します。

## 現在の実行手順

1. token IDを640行ずつ送信し、embeddingと初層QKVを同じqueryで計算する。
2. hidden stateはF32のバイト列、QKVはQ:F32・K/V:BF16の形式でclientに保持する。
3. Attentionをhead群に分割する。5,000 tokensの全域Attentionは1 head/query、局所Attentionは2 heads/query。各head群に全キーを渡してV scaleを維持する。
4. contextを既存のper-token INT8へ量子化し、residual F32とともに640行ずつ送る。同じquery内で出力射影・MLP・次層QKVまで計算する。
5. 第9層の後は129行、第10層の後は65行へ絞る。第11層のCLS計算と最終正規化も同じqueryで完了する。

5,000 tokensは161 inference queriesです。層境界を返送して再送する前版の182 queriesから21回減らしました。転送は約590 MBから約513 MBです。[最新の測定結果](../benchmarks/query-fusion/RESULTS.md)を参照してください。

K/VのBF16と、context・局所Qの量子化精度は従来のままです。`PackedQkv`はBF16をF32へ展開せずに転送用データへ並べ替えます。RoPEは元の絶対位置を使います。

V scaleは各channelのキー全体から求めます。局所Attentionも全キーをhead方向に分割することで、この値を変えずに実行します。

## バイナリ形式

requestは8個のlittle-endian `u32`とpayloadです。

```text
[op, layer, start, total, head, heads, kstart, rows]
```

`total`は1〜5,000、`layer`は0〜11、`rows`は正で`start + rows <= total`です。request本体は`2 MiB - 128 bytes`以下です。返り値はCandidの`Result<{ data: blob; instructions: nat64 }, text>`です。

| op | payload | data |
| --- | --- | --- |
| 15 | `rows`個のtoken ID。layer 0、最大680行 | hidden F32全行、続いてpacked QKV全行 |
| 2 | Q F32、K BF16、V BF16 | `rows × heads × 32` context F32 |
| 13 | 局所層のみ。Q INT16、Q scale F32、K BF16、V BF16 | context F32 |
| 16 | residual F32全行、context scale F32全行、context INT8全行。最大680行 | 第0〜10層ではhidden F32と次層packed QKV。第11層では最終384 F32 embedding |

op 15/16の通常replyは`rows × 384` F32 hiddenを先に連結し、その後にpacked QKVを置きます。packed QKVは行ごとにQ:384 F32、K:384 BF16、V:384 BF16です。計4,608 bytes/行で、680行でも3 MiB以内です。
op 16の第11層だけは`start == 0 && rows == 1`が必要で、hidden/QKVの連結形式ではなく最終embeddingを返します。

op 2/13はtoken→head→32 channelsの順です。K/Vはkey→head→32 channelsの順で、`key_range(layer,start,rows,total)`の範囲を含みます。Q、K、Vを順に連結し、op 13ではQの直後に`rows × heads`個のscaleを置きます。

context scaleは`max(max(abs(context[token])) / 127, 1e-12)`、符号付きINT8はround-to-nearest-ties-to-even後に±127へclampします。`-128`は拒否します。
局所Qは32 channelsごとに`max(max(abs(Q)) / 8191, 1e-12)`でscaleを求め、同じ丸めで±8191のINT16にします。scaleはF32です。

[codec実装](../crates/core/src/packed.rs)と[演算・request検査](../crates/core/src/lib.rs)が実際の仕様です。検証ハーネスは不正request 15種類の拒否も確認します。

## 比較用の経路

`--sharded-split`は融合前のpacked経路、`--sharded-raw`はF32 wireによる比較経路です。通常は`--sharded`を使います。既存の低水準opも維持しています。

| op | 内容 |
| --- | --- |
| 0 | ID列からembedding norm後のhiddenを返す。`rows == total` |
| 1 | hidden F32からRoPE後のQKV F32を返す |
| 3 | residual F32とcontext F32から次層hiddenを返す |
| 4 | CLSの384 F32から最終embeddingを返す |
| 11 | op 1と同じ入力からpacked QKVを返す。最大1,023行 |
| 12 | op 16と同じpayloadからhiddenだけを返す。最大1,088行 |
