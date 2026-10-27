# query境界の融合・コピー削減・長文Attention batch

5,000 tokensの分割経路を182から161 queriesへ減らしました。転送を約590から513 MBへ削減し、合計命令数も約7.35%減っています。全12層の最終embeddingは前版とbit単位で一致しました。

通常APIの上限は304 tokens / 1 query、1,365 tokens / 7 queriesを維持しています。5,000 tokensは追加の分割クライアントを使います。

## 通常版の比較

前版はmainの`9d43e82`時点、Wasm `6c56483bfd1b669971b15a76f2c2a58491593fed28988759822ad3eec84679b3`です。今回の測定はApplication subnet / Production、50億命令/queryで行いました。

| 5,000 tokens | 前版 | 今回 |
| --- | ---: | ---: |
| queries | 182 | 161（−11.54%） |
| 合計命令 | 177,813,740,538 | 164,741,174,056（−7.35%） |
| 最大/query | 2,936,313,152 | 2,628,934,924 |
| Candid request bytes | 288,952,687 | 211,852,239 |
| Candid reply bytes | 300,724,965 | 300,722,492 |
| request/reply合計 | 589,677,652 | 512,574,731（−13.08%） |
| 最大request | 2,093,356 | 1,969,580 |
| 最大reply | 3,072,047 | 2,949,167 |

Candid bytesはHTTP/TLS・モデルuploadを含みません。query数もinferenceだけの値です。
256 tokensの通常APIは4,064,727,088→4,064,748,620命令で、+21,532命令（約0.00053%）の微差でした。公開上限304の全長走査と境界試験は通っています。

## 変更した処理

**初回embeddingとQKVを融合。** op 15で640行ずつ処理し、hidden F32とpacked QKVを返します。

**finishと次層QKVを融合。** op 16で出力射影・residual・MLPを実行し、そのまま次層のnorm・QKV・RoPEへ進みます。第11層では最終正規化も同じqueryで終えます。

以前の処理には、queryが返したhiddenを次のQKV queryへ送り直すだけの区間がありました。再送payloadは初回分も含め77,097,984 bytesで、融合により取り除いています。header削減も含む実測の転送削減量は77,102,921 bytesです。

**clientはwire表現を保持。** hiddenをF32のバイト列、QKVをQ:F32・K/V:BF16のまま持ちます。`PackedQkv`はheadを選んで元のBF16 bytesを直接転送し、BF16→F32→BF16の往復を除きます。bufferも必要行数ぶん事前確保します。

**送信payloadのcloneを除去。** 長さを記録してからVecをAPIへmoveします。前版の1推論ぶんでは約289 MBのclone対象がありました。これはclient内のコピー削減で、通信量の削減とは別です。

**canisterのcodecを改善。** tokenごとのQ用一時Vecを除き、出力bufferへ直接追記します。BF16変換は8要素のSIMD処理とscalar tailにし、元の丸めを維持します。

**全域Attentionのbatchを5,000 keysまで適用。** 関数を分けた構成で再測定したところ、長文でも4つのQ行でKを共有する効果がありました。局所Attentionは行ごとの既存経路を使います。量子化幅や近似式は変更していません。

新規opのreplyは1行4,608 bytesです。640行tileで161 queriesとなり、公開上限680行でもreplyが3 MiB以内に入ることを確認しています。[プロトコル](../../docs/SHARDED.md)に形式を記載しました。

## 採用の切り分け

| 構成 | 5,000 tokens合計命令 | queries |
| --- | ---: | ---: |
| 前版 | 177,813,740,538 | 182 |
| 融合・codec改善、batchは512 keys以下 | 175,900,617,388 | 161 |
| 上記＋長文の全域Attention batch | 164,741,174,056 | 161 |

候補の記録は[short-5000.json](short-5000.json)と[long-5000.json](long-5000.json)です。通常Wasmの命令数で採否を決め、profiling値の差は通常版へ足し合わせていません。

## 出力と境界の検証

| case | queries | 合計命令 | 最大/query | cosine対FP32 |
| --- | ---: | ---: | ---: | ---: |
| 1365 | 55 | 26,634,458,682 | 1,214,886,899 | 0.999610393 |
| 1366 | 58 | 26,675,252,059 | 852,570,022 | 0.999626836 |
| 2048 | 77 | 44,947,815,189 | 1,342,156,866 | 0.999556149 |
| 4096 | 151 | 121,102,609,131 | 1,767,794,170 | 0.999641624 |
| 5000 | 161 | 164,741,174,056 | 2,628,934,924 | 0.999669377 |
| space-5000 | 161 | 164,741,174,056 | 2,628,934,924 | 0.999642830 |
| cooking-5000 | 161 | 164,741,174,056 | 2,628,934,924 | 0.999508387 |
| weather-5000 | 161 | 164,741,174,056 | 2,628,934,924 | 0.999698703 |

1,365の分割実行は通常APIとの比較用です。1,365以下には通常APIを使います。

- Rust unit 16件。
- 線形演算112件を独立したnative密行列計算と照合。
- QK 44件を旧経路・i64、AV 44件をi64と照合。
- BF16 SIMD codec 30件をscalarとbit比較。NaN・符号・丸め境界・8要素の余りを含む。
- 通常APIの2〜512全511長、追加stress 57件、実文書16件を旧出力とbit比較。
- 通常APIの513〜1,365の32長×3入力＝96件、実文書11件を旧出力とbit比較。
- 融合経路の実文書8件を前版の分割経路とbit比較。最小cosine対FP32は0.999508387。
- 2/3、31/32、63/64/65、127/128/129、255/256/257、639/640/641など54件を旧出力とbit比較。
- op 15とop 16の680行をそれぞれ分離演算とbit比較。実際の最大replyは3,133,487 bytes、3 MiB以下。
- 上限超過・非有限値・不正なscale・head/startの範囲など15種類の不正requestを拒否。
- 通常Candidを維持し、通常Wasmに`profileStep`がないことを確認。

[8文書の結果](sharded/summary.json)、[54件と680行境界](sharded/stress.json)、[fixtureのSHAと検証範囲](sharded/environment.json)、[通常版のsource・Wasm](../validation/environment.json)を保存しています。

## 時間の参考計測

全ビルドと他の検証終了後、旧client＋旧Wasm、今回のclient＋今回のWasmを新しいPocketIC環境で1回ずつ順に実行しました。
モデルのロードと不正入力検査は計測区間外です。計測区間はclientのwire準備・query・返り値処理を含みます。

| 5,000 tokens | 前版 | 今回 |
| --- | ---: | ---: |
| inference wall | 19.051秒 | 18.851秒 |

[前](latency-before.json)・[後](latency-after.json)。単発比較では時間差は小さく、速度差の再現性は未検証です。OS・熱・背景処理を制御しておらず、mainnetのレイテンシを示す値ではありません。
命令数と転送量の改善は上の通常版計測で確認しています。

## 成果物と再実行

- 通常Wasm: `4591daf010e1722026cb334120b8d4b7ffa2f723431d2387a95ad6c8033f182b`、3,605,841 bytes。
- profiling Wasm: `2e18be3fdc3805a2fb5acac28e7c0ab6e3677edeaf8b5efc37ec5953ef72f249`。
- [artifact一覧](artifacts.json)、[比較metrics](metrics.json)。
- [線形演算](linear.json)、[QK](qk.json)、[AV](av.json)、[BF16 codec](codec.json)、[層別profile](profile-1365.json)、[Attention単体](attention-profile.json)。profiling結果は同じprofiling Wasmから取得。

[開発手順](../../docs/DEVELOPMENT.md)と[分割queryの手順](../../docs/SHARDED.md)を参照してください。通常APIの対応上限は1,365、分割経路は5,000で、入力を暗黙に切り捨てません。mainnet運用・retrievalのRecall@kは今回の検証対象外です。
