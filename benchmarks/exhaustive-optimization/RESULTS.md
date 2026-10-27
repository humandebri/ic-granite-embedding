# 追加最適化の検証結果: 288 tokensを1 query、512 tokensを2 queryへ

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26。PocketIC server 11.0.0、Application subnet / Production、query上限50億命令。
**INT8、入力全体、全12層、CLS poolingを維持した。今回の変更前の出力とbit単位で一致する。**
公開Candidの型・メソッドは変更せず、queryの計画を2〜288 tokensは `[12]`、289〜512 tokensは `[5,7]` にした。
2〜512 tokensの全511長を、公開APIで実行して確認した。

| tokens | query数 | 合計命令 | 最大/query | ローカル単発時間 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1 | 554,156,463 | 554,156,463 | 0.087秒 |
| 64 | 1 | 1,104,134,170 | 1,104,134,170 | 0.145秒 |
| 128 | 1 | 2,199,360,247 | 2,199,360,247 | 0.264秒 |
| 160 | 1 | 2,720,196,434 | 2,720,196,434 | 0.321秒 |
| 176 | 1 | 2,999,779,780 | 2,999,779,780 | 0.353秒 |
| 256 | 1 | 4,274,785,648 | 4,274,785,648 | 0.477秒 |
| 288 | 1 | 4,826,633,982 | 4,826,633,982 | — |
| 512 | 2 | 8,836,065,163 | 4,699,486,897 | 0.937秒 |

256 tokensは4文書とも4,274,785,648命令。直前の4,493,299,857から218,514,209命令（約4.86%）削減した。
50億命令への余裕は約14.50%。1 query達成時の47.38億命令からは約9.77%削減した。
288の値は長さ走査の固定seed入力、ほかの表の値は既存のtokenized文書による測定。

全511長のうち最大/queryは **287 tokensの4,871,607,769命令**で、余裕は約2.57%。
これは固定seedの各長1入力に対する実測で、あらゆる入力に対する最悪値の証明ではない。
511/512 tokensは反復入力も検査し、512 tokensの既存4文書も通過した。
[全長走査](sweep.json)、[追加57ケース](stress.json)、[公開API 16ケース](summary.json)。

## 採用した変更

- 8/4/1 tokensの端数にも固定形状GEMM kernelを生成し、16出力channelで入力を共有する。255などの境界長の割高な処理を減らす。
- 直後に全要素を上書きするGEMM出力、量子化出力、Attention出力のゼロ初期化を省く。未初期化領域は`MaybeUninit`または生pointer経由で全要素を書き、最後にVecの長さを設定する。
- LayerNormの差分・平方・重み適用をSIMD化する。平均・分散の加算順序と、浮動小数点の積の順序を維持する。
- Wiに1段のStrassen法を使う。重みの和差はモデル有効化時に準備し、queryでは64 tokens単位のprefixに適用する。残りは通常kernelで処理する。
- Strassenのleafは1出力あたりi32を1個だけ保持する。以前の4個の部分和を保持する版より中間バッファを4分の1にし、整数で復元してから元のF32 scaleを適用する。
- AVでは各確率ロードを32出力channelで共有する。named accumulatorをbuild.rsで生成し、動的な配列index処理を避ける。
- 1536列×32 tokensのkernelは、Kを384列ずつ4ブロックで処理する。整数和は正確に保ち、大きな単一関数のコード量・live local数を減らす。
- 512 tokensの中間stateは既存GRC1のraw F32形式を既定にした。DEFLATE処理を省き、F32のバイト変換もlittle-endian上で一括copyする。従来の圧縮形式もdecode可能。

重みの量子化・RoPEの準備は引き続きquery前に行う。層数・token数・INT8の範囲は減らしていない。
Strassenは整数演算の恒等変形で、追加の近似を含まない。1〜3段の整数上限とi64直接計算との一致をunit testでも確認した。

計測用ビルドの段階別合算。単一の通常queryとは区別する。

| 段階 | 直前版 | 今回 | 削減率 |
| --- | ---: | ---: | ---: |
| GEMM・水平加算・scale | 3,288,235,776 | 3,166,768,421 | 3.69% |
| 活性値量子化・バッファ準備 | 160,676,401 | 95,959,298 | 40.28% |
| AV | 232,449,048 | 215,721,156 | 7.20% |

通常queryは計測コードを含まないビルドで検証した。
[最終内訳](final-profile-summary.json)、[各層の計測値](final-profile.json)、[計測版の環境](profile-environment.json)。
GEMMが最終プロファイルの約73%を占め、引き続き最大の処理である。

## 比較して採用しなかった案

| 案 | 計測用ビルドの256 tokens・通常step | 判断 |
| --- | ---: | --- |
| 1段・leaf 32×64・AV共有 | 4,307,503,039 | 改善を確認。関数分割を追加 |
| 2段Strassen | 4,322,966,603 | 合成・準備が増え、1段より悪化 |
| 3段Strassen | 4,501,176,607 | さらに悪化し、重みメモリも増大 |
| K=1536だけを分割 | 4,302,540,574 | 採用 |
| 384列とStrassen leafまで分割 | 4,408,392,166 | 約1.06億命令増加。標準には採用せず |

最後の小さいkernel版は、反復中央値約0.454秒・install約3.8秒と速かったが、標準は命令数を優先した。
[2段](depth2-summary.json)、[3段](depth3-summary.json)、[採用版の分割実験](chunked-summary.json)、[小さいkernel版](compact-summary.json)。
`strassen2`/`strassen3`/`compact-gemm`は比較用のexperimental featureとして残した。
公開APIの資源上限の検証はdefault feature構成に対するもの。experimental構成は256 tokensの比較用で、拡張した上限の保証対象ではない。

288 tokensは計測用ビルドでも通常step 1回で通った。296/304は計測用ビルドで上限に達した。
通常公開APIは288以下の全長を走査して採用した。[容量の比較](capacity-before-plan.json)。
512の層別計測から5層＋7層を選び、最終的には公開APIと全長走査で検証した。[当初の512層別計測](profile-512-before-plan.json)。

## 圧縮と転送量

512 tokensを2 queryにした時点では、DEFLATEを含む最大queryは4,876,034,766命令だった。
raw F32への変更で4,699,486,897命令になり、合計は9,075,831,753から8,836,065,163へ減った。
511 tokensの最も重いqueryも4,919,072,611から4,742,868,550へ減った。

512の往復転送量は1,353,926から1,576,654 bytesへ約0.22MB増えた。
元の圧縮率は約14%に留まっており、今回は命令数・時間・余裕を優先した。
512でもrequest最大786,466 bytes、response最大786,519 bytesで、実際のmessage上限内だった。
256 tokensは中間stateを転送せず、往復2,645 bytesのまま。
`compressed-carry` featureで従来の圧縮を選べる。unit testはraw・圧縮の両方のbit round-tripと不正入力拒否を検査する。
[圧縮ありの512計測](compressed-api-512.json)、[最終512計測](api-512.json)。

## 精度・境界・時間

- unit test 10件、最終GEMM単体112ケースが通過。GEMMは独立したnative整数参照のhashと完全一致した。
- 既存API 16ケースと追加57ケースで、変更前のF32 embeddingとbit単位で一致した。
- 全長走査511ケースのうち、2〜288の287長と511/512の計289ケースは変更前とbit単位で比較した。ほかの長さは命令上限、query数、384次元・有限値・L2 normを検査した。
- profiling有無の256 tokens・4文書もbit単位で一致した。
- 513 tokens拒否、不正な継続位置拒否、正しい継続位置5でのNaN state拒否を確認した。
- Candid一致、通常WasmにprofileStepがないこと、request/responseの上限、実際のProduction query上限を検査した。

FP32参照との最小cosineは0.999486899で、直前版と同じ。
検索のRecall@k等は評価していない。固定文書・ランダムtoken・反復tokenでの出力一致とresource検証であり、retrieval benchmarkではない。

キャッシュを避けた256 tokensの4文書×5回の中央値は **0.613968秒→0.476782秒**（観測上約22.3%短縮）。
毎回、診断requestの計算に使わないfieldを変えてqueryキャッシュを避け、出力が同じことも検査した。
初回installは約136.7秒→約21.1秒。
OS負荷・温度等を管理した統計的検証ではなく、mainnetのレイテンシ保証ではない。
[比較元の反復](repeat-baseline.json)、[今回の反復](repeat-single.json)。

Strassenの準備済みWi重みは約49.5MBの追加データを持つ。
モデル有効化後のWasm heapは312,868,864 bytesで、今回の測定では比較元と同じだった。
既存heapの空き領域を再利用できた結果であり、追加データが無料になるという意味ではない。
[最終memory計測](model-memory.json)、[比較元memory](memory-139b99f554d4680ba8af918f506ab000821fece11b3481215ecd91f4f3eeb116.json)。

## 再現・artifact

READMEのfixture生成後、`python3 scripts/validate_exhaustive.py`。
`CARGO_TARGET_DIR`でビルド先、`GRANITE_WASM`で検証する通常Wasmを指定できる。
比較元の出力・長さ走査は同じdirectoryに保存している。
計測用はdefault feature＋profilingでbuildし、`granite-bench 256 384 --micro --micro-extended --probe`を実行する。

[通常版の環境・source hash](environment.json)、[通常Wasmのexport一覧](module-exports.json)、
[追加入力](stress.json)、[比較元](stress-baseline.json)、[全長入力](sweep.json)、[2〜288の比較元](sweep-baseline.json)。
通常Wasm SHA256: `cab8d077f6e8980dfb2333e12c18fc08834c6965d62583a015a0fa9d5bc70265`、3,194,652 bytes。
検証済み通常Wasmはworkspaceの`target/wasm32-unknown-unknown/release/granite_canister.wasm`に配置する。

手元の旧candid-extractor（Cranelift 0.117.2）は依然assertionで失敗した。
同じRust宣言からnative exampleでCandidを出力し、既存DIDとの一致を検査した。旧tool固有の問題とPocketICでの実行結果は区別する。

mainnetにはdeployしていない。重み・RoPEはheap保持で、upgrade後の復元は未実装。
512 tokensのAPI上限を、モデル本来の最大長や5,000 tokens全体のembeddingと混同しない。
