# 5,000 tokensの分割query

`startEmbedding` / `continueEmbedding`は1,365 tokensまでです。それ以上の入力には、クライアントがhidden stateとQKVを分割保持し、通常の`step` queryを呼ぶ経路を追加しました。全文を12層に通し、全域Attentionでは各headが入力全体のK/Vを参照します。

参照クライアントは[crates/bench/src/sharded.rs](../crates/bench/src/sharded.rs)です。PocketICによるロード・測定を含むハーネスで、ブラウザSDKではありません。Candidのメソッド型は変えていません。

## 実行する

通常版Wasmとnativeハーネスをビルドし、参照データを準備します。`POCKET_IC_BIN`の指定は[開発手順](DEVELOPMENT.md)を参照してください。

```sh
cargo build --release --locked -p granite-canister --target wasm32-unknown-unknown
cargo build --release --locked -p granite-bench
.cache/venv/bin/python scripts/prepare.py --lengths=256,5000
# 専用PocketICへインストール・ロードして、全文の分割queryを実行
target/release/granite-bench 5000 --sharded
```

`--case=space`などは対応する`.cache/tokens-space-5000.json`と参照ファイルが必要です。8ケースの検証に必要な生成・照合手順は次のとおりです。

```sh
mkdir -p .cache/next/references
.cache/venv/bin/python scripts/prepare.py --output .cache/next/references \
  --lengths=1366,2048,4096,5000 --stress --stress-lengths=5000
# 生成したtokens-*.json / reference-*.f32 / reference-int8-*.f32を.cache/にも配置。
# 既存ファイルがある場合は同一内容を確認する。
python3 scripts/validate_sharded.py
```

この8ケースには既存の1,365 tokensのfixtureも必要です。通常の[参照生成手順](DEVELOPMENT.md#モデルと参照データを準備する)で準備します。
`GRANITE_WASM`で通常版のパス、`CARGO_TARGET_DIR`でハーネスのビルド先を指定できます。`--summarize-only`は保存済みの同一SHAの結果を再検査します。

## 分割と数値の維持

1. 1,984 tokensずつembeddingを取得する。
2. 1,000行ずつQKVを計算し、QをF32、K/VをBF16で受け取る。
3. Attentionをhead群に分割する。5,000 tokensの全域Attentionは1 head/query、局所Attentionは2 heads/query。どちらも全入力のK/Vを渡す。
4. contextを既存の出力射影と同じper-token INT8へ量子化し、residual F32とともに1,088行ずつ送る。
5. 次の層へ進む。第9層の後は129行、第10層の後は65行、第11層の後はCLSの1行に絞り、最後に正規化する。

K/VのBF16は従来からAttention境界で使っていた形式です。contextと局所AttentionのQは、従来query内で行っていた量子化を送信前に行います。新たな量子化精度は導入しません。

Vのscaleはheadごとの各channelについてキー全体から決まります。局所Attentionも全入力のK/Vを送ることで、haloだけに切り出した場合のscale変化を避けています。RoPEは元の絶対位置を使います。

5,000 tokensは182 inference queriesです。初期の687 queriesとの比較では、query数を約73.5%削減しました。転送量・命令数・精度の最新値は[測定結果](../benchmarks/full-context/RESULTS.md)を参照してください。モデルアップロードはこのquery数に含みません。

## バイナリ形式

requestは8個のlittle-endian `u32`、続いてpayloadです。

```text
[op, layer, start, total, head, heads, kstart, rows]
```

`total`は1〜5,000、`layer`は0〜11、`rows`は正、`start + rows <= total`です。request本体は`2 MiB - 128 bytes`以下です。queryの返り値はCandidの`Result<{ data: blob; instructions: nat64 }, text>`です。

| op | payload | data |
| --- | --- | --- |
| 0 | `total`個のtoken ID。`rows == total` | embedding norm後の`rows × 384` F32 |
| 1 | `rows × 384` F32 | RoPE後の`rows × 1152` QKV F32 |
| 2 | Q F32、K BF16、V BF16 | `rows × heads × 32` context F32 |
| 3 | residual F32、context F32 | 次層hiddenの`rows × 384` F32 |
| 4 | CLSの384 F32 | 最終384 F32 embedding |
| 11 | op 1と同じ。最大1,023行 | 行ごとにQ:384 F32、K:384 BF16、V:384 BF16 |
| 12 | residual F32全行、scale F32全行、context INT8全行。最大1,088行 | 次層hidden F32 |
| 13 | 局所層のみ。Q INT16全行、Q scale F32全行、K BF16、V BF16 | context F32 |

op 2/13のQはtoken→head→32 channelsの順、K/Vもkey→head→32 channelsの順です。K/Vは`key_range(layer,start,rows,total)`で決まるキー範囲を含みます。各配列はQ、K、Vの順で連結します。op 13だけQの直後に`rows × heads`個のscaleがあります。

op 12のscaleは`max(max(abs(context[token])) / 127, 1e-12)`、符号付きINT8はround-to-nearest-ties-to-even後に±127へclampします。`-128`は拒否します。
op 13のQ scaleはheadの32 channelsごとに`max(max(abs(Q)) / 8191, 1e-12)`、INT16は同じ丸めで±8191です。scaleはF32で計算・転送します。

実際のcodecは[coreのpack関数](../crates/core/src/lib.rs)を参照してください。検証ハーネスは上限超過・不正なscale・整数範囲外・非有限値など9件の拒否も確認します。

`--sharded-raw`はパック前のop 0〜4による比較経路です。新規クライアントは`--sharded`のパック形式を参照してください。
