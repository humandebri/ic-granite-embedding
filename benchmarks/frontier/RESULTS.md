> 過去段階の記録です。現在の通常版・5,000 tokens対応は[最新レポート](../full-context/RESULTS.md)を参照してください。

# 300 tokensを1 query、長文は1,365 tokensまで

> この計測後に旧実装を整理した。[現在の実装の検証](../cleanup/RESULTS.md)。このページは最適化・長文対応を測った時点の記録。

2026-10-27。PocketIC server 11.0.0、Application subnet / Production、各queryの上限50億命令。

出力側MLPへ1段Strassenを追加し、長文Attentionの整数和を安全に拡張した。
**256 tokensは4,230,646,590 → 4,166,097,469命令（約1.53%削減）**。
公開上限を **300 tokens / 1 query、1,365 tokens / 7 query** に拡張した。
INT8、CLS/SEPを含む入力全体、全12層、384次元CLS embeddingを維持する。

## 通常公開APIの実測

| 入力tokens | query数 | 合計命令 | 最大/query |
| --- | ---: | ---: | ---: |
| 256 | 1 | 4,166,097,469 | 4,166,097,469 |
| 300 | 1 | 4,940,779,373 | 4,940,779,373 |
| 512 | 2 | 8,626,850,792 | 4,589,044,787 |
| 640 | 4 | 11,141,612,099 | 3,515,510,548 |
| 768 | 4 | 13,726,148,050 | 4,356,520,027 |
| 1024 | 5 | 19,330,564,494 | 4,550,848,763 |
| 1280 | 7 | 25,554,271,132 | 4,120,600,289 |
| 1365 | 7 | 27,788,178,100 | 4,570,613,345 |

300 tokensは固定seed入力の走査、ほかは既存または追加したtokenized文書。
公開planは2〜300 `[12]`、301〜512 `[5,7]`、513〜768 `[3,3,3,3]`、769〜1024 `[2,2,2,2,4]`、1025〜1365 `[1,2,1,2,1,2,3]`。
各planの合計は12層で、入力を独立チャンクへ分割して意味を変える処理は使わない。

[旧入力域の結果](final/regression/summary.json)、[長文11ケース](final/summary.json)。

## 1 queryの実測上限

採用済み旧版は280〜336 tokensを1 token刻みで調べ、固定seed入力では298まで成功した。
改善後は289〜320を、固定seed・同一token 0・同一token 50367の3入力、計96ケースで調べた。
この走査での最大成功長は **302 tokens**。302の診断queryは4,992,965,068命令、余裕は約0.14%。
303〜320は各入力で実際の命令上限に達した。それ以外のエラーを上限超過として扱わないよう検証した。

公開上限は余裕を残して300にした。2〜512 tokensの全511長を公開APIで再検証し、
その最大/queryは299 tokensの4,943,852,319命令、余裕は約1.12%。
長文の追加96ケースでは1023 tokensの4,575,188,043命令が最大だった。
これは検査した入力での実測であり、あらゆる入力の最悪値の証明ではない。

[旧版の境界走査](capacity-baseline.json)、[改善後の96ケース](final/capacity.json)、[旧入力域の全511長](final/regression/sweep.json)。

## 長文の上限と転送量

中間stateは従来のGRC1 raw F32で、1 tokenあたり384×4=1,536 bytes。
1,365 tokensの最大requestは **2,096,674 bytes** で2 MiB以内、最大responseは2,096,727 bytesだった。
全7 queryの往復転送量は **25,167,487 bytes**。
1,366 tokensではstateのF32値だけで2,098,176 bytesになり、request上限を超える。
したがって1,365は現在の「全stateを1 messageで次へ渡す」形式での上限である。

内部診断APIには以前2,090,000 bytesの制限があり、最初の1,365-token層別計測がそこで止まった。
2 MiBからCandid用に128 bytesを残す制限へ揃え、全12層の計測と公開APIの実行を完了した。
通常の公開Candidの型・メソッド名は変更していない。

1,365 tokensは全層を1層ずつ実行した計測版と、7 queryで実行した通常版の出力がbit単位で一致した。
さらに長い入力、とくに5,000 tokensは未対応。F32 stateだけで7.68MBになるため、state分割と層内計算の分割を伴う別のプロトコルが検討対象となる。
モデル自体の最大長と、このAPIで実行できる長さは区別する。

[1,365 tokens公開API](final/api-1365.json)、[各層の計測](profile-1365.json)。

## 採用した演算の変更

**出力側MLP（Wo）にも1段Strassenを適用。**
384出力×1536入力を、192出力×768入力の7積へ変換する。
leafは32 tokens×64出力、Kを192ずつ4ブロックで計算し、大きな単一関数を避ける。
重みの和差は有効化時に準備する。整数で復元した後、元のtoken scaleと出力channel scaleを元と同じ順序で適用する。
この恒等変形による追加の近似はない。既定featureへ`strassen-wo`を追加した。

**512 keysを超えるAVを、512-key i32ブロック＋i64合算へ拡張。**
1ブロックの絶対上限は512×2047²=2,145,387,008でi32以内。
ブロック間をi64で加算し、最後に一度だけF32へ変換してscaleを適用する。
5,000要素でも絶対上限は20,951,045,000でi64以内。
512 keys以下は従来のSIMD経路を使い、既存入力の結果を変えない。
QKの幅はD=32なので、入力長を伸ばしても従来のi32範囲保証を維持する。

[生成kernel](../../crates/core/build.rs)、[Strassen](../../crates/core/src/strassen.rs)、[Attention](../../crates/core/src/integer_attention.rs)。

## 命令数の切り分け

2段Strassenは新しいleafでも計測用通常stepが9,139,642命令増えたため不採用。
Woへの適用だけでは計測用stepが65,029,104命令減り、4文書の出力が完全一致した。
各候補のGEMM 112ケースも、独立したnative整数参照と一致した。
[候補の比較](experiments.json)。

12層を別queryで測ったprofile集計では、GEMM・水平加算・scaleが3,122,629,267 → 3,058,921,077命令、
AVが215,721,156 → 183,927,132命令になり、QKは同じだった。
AVの分岐追加後に計測用ビルドの生成コードも変化した。
profilingの有無で最適化結果が異なるため、この差分を通常版へ合算していない。
採否は上の通常公開APIの実測値で判断した。
[phase比較](instruction-phase-comparison.json)、[256 tokensのprofile](profile-256-summary.json)。

## CPU時間

サンプリングでは、実行threadがWasmtimeからJITコードへ入るstackを観測した。
JIT内の関数名は解決できず、キャッシュmissやregister spillを原因として断定する測定にはなっていない。
切り分け用の行列積別時間では、以前のタイル拡大でWi/QKVが遅くなる一方、ブロック処理するWoは改善していた。
これらの補助計測はビルド等と並行したため、採否のレイテンシ値としては下の最終比較を用いた。
[CPU sample](cpu-baseline.sample.txt)、[小タイル](micro-small.json)、[採用済みタイル](micro-adopted.json)、[新構成](micro-final.json)。

最終比較は全ビルド・検証後、前版→新版の順に実行した。
各構成は4文書×5回。未使用fieldのnonceを毎回変えてqueryキャッシュを避け、出力一致も検査した。

| 構成 | 256 tokens中央値 | 最小〜最大 | 初回install |
| --- | ---: | ---: | ---: |
| 前版 | 0.579224秒 | 0.568880〜0.870056秒 | 41.41秒 |
| 新版 | 0.560764秒 | 0.552402〜0.582391秒 | 43.11秒 |

今回の順次測定では約3.19%の短縮を観測した。OS負荷・温度は管理しておらず、mainnetの時間保証ではない。
[最終時間比較](latency-comparison.json)、[前版20試行](latency-before-repeat.json)、[新版20試行](latency-after-repeat.json)。

Wo用の準備済み重みは論理的に約24.8MB増える。有効化直後のWasm heapは312,868,864 → 326,631,424 bytesだった。
空き領域の再利用を含む値で、query中のpeak heapを保証する測定ではない。
Wasmコードは5,434,900 → 6,154,325 bytesに増えた。
[前版memory](latency-before-memory.json)、[新版memory](latency-after-memory.json)。

## 検証の範囲

- native unit test 10件。Strassenの384/1536列・1〜3段をINT8極値とi64直接計算で比較した。
- 旧入力域2〜512の全511長、追加57ケース、公開API 16ケースで、前回版とF32出力がbit単位で一致。
- 新しい長文域は32長×3入力の96ケースでquery上限・有限値・384次元・L2 normを検査。
- 長文11ケースをFP32参照と比較し、最小cosineは0.999592611。
- AVは512/513境界、最大値・最小値・ブロック間相殺・変動する値を含む44ケースでi64参照と完全一致。最大5,000要素を検査。
- 最終計測版のGEMM 112ケースはnative整数参照と一致。
- 1,365 tokensの12層別実行と7 query実行がbit単位で一致。
- 1,366 tokens、不正な継続位置、NaN stateを拒否。
- 通常WasmにprofileStepがなく、Candidとpayload上限を確認。

長文域の全853長を走査したものではない。retrievalのRecall@kも評価していない。
[検証環境・source SHA](final/environment.json)、[長文stress](final/stress-long.json)、[FP32参照の由来](final/references-provenance.json)。

## 再現

環境・参照manifest・計測用JSONの準備を含む手順は[開発ドキュメント](../../docs/DEVELOPMENT.md)を参照してください。

READMEのfixture準備後に`python3 scripts/validate_frontier.py`。
旧入力域・長文域・1 query境界を並行して検査するため、検証中の時間は性能比較に使わない。
`--summarize-only`は実行済み結果の再照合用で、推論を再実行しない。
初回のツール早期終了を修正した後、完了していた計算量検査を保持し、長文精度比較を追加実行して最終結果を再照合した。

計測版を再作成する場合はdefault feature＋`profiling`でbuildし、
`granite-bench 1365 --micro-av --micro --micro-extended --probe`を実行する。
`.cache/micro-av.json`、`micro-linear.json`、`probe-profile.json`をこのdirectoryの対応する計測JSONとして保存する。
`--capacity-range=289:320 --capacity-patterns=3`で1 query境界を再測定できる。
`CARGO_TARGET_DIR`と`GRANITE_WASM`でビルド先・測定moduleを明示できる。

検証済み通常版は`target/wasm32-unknown-unknown/release/granite_canister.wasm`。
SHA256: `fb922e5ce45bc5128fcb9dffbbe19a486b9acec6ad0d3b39d56459e04f473c00`、6,154,325 bytes。
[artifact一覧](artifacts.json)。mainnetにはdeployしていない。
公開上限の資源検証はdefault feature構成に対するもの。
