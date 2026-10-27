# 追加最適化: 256 tokensを約44.93億命令へ

> 当時の専用runner・実験featureは整理済みです。再実行には[現行の開発手順](../../docs/DEVELOPMENT.md)を使ってください。

> **履歴: このページは当時の測定記録です。** 本文の上限・query数・「既定」は当時の構成を指します。現在の仕様と結果は[最新レポート](../frontier/RESULTS.md)、経緯は[計測一覧](../README.md)を参照してください。

2026-10-26。PocketIC server 11.0.0、Application subnet / Production設定。
256 tokens・全12層の公開queryは、4文書すべて **4,493,299,857命令**。
直前の4,737,545,056命令から **244,245,199命令（5.16%）削減**した。
50億命令の上限に対する余裕は約10.13%。入力・層数・量子化精度・公開Candidは維持した。

| tokens | query数 | 合計命令 | ローカル単発時間 |
| --- | ---: | ---: | ---: |
| 32 | 1 | 568,343,647 | 0.112秒 |
| 64 | 1 | 1,163,290,681 | 0.242秒 |
| 128 | 1 | 2,314,298,974 | 0.354秒 |
| 160 | 1 | 2,848,907,632 | 0.460秒 |
| 176 | 1 | 3,140,504,301 | 0.529秒 |
| 256 | 1 | 4,493,299,857 | 0.631秒 |
| 512 | 5 | 10,271,637,273 | 1.394秒 |

短い入力が常に軽くなるわけではない。タイル境界の255 tokensは4,606,330,871命令で、検査した256 tokens以下の入力では最大だった。
こちらの余裕は約7.87%。2〜256 tokensの全長・全入力に対する最悪値の証明ではない。
512 tokensの5 query経路も4文書で通過した。合計命令数は文書によってcarryの圧縮量などで少し変わる。

## 変更と実測

- 384列のLinearを32 tokens × 32出力channelに変更し、入力ロードをさらに共有する。1536列は既存の32×16を維持する。
- Kとそのscaleをheadごとに連続配置し、QKスコアを4 keysまとめて計算する。整数の水平加算・F32変換・scale適用も4個ずつSIMD化する。
- AVの4出力の水平加算をSIMDでまとめ、整数配列への一時退避と再ロードを省く。
- exp近似から確率の整数化・整数分母の計算までを融合する。使わない浮動小数点の分母と、expの中間配列への書き戻しを省く。
- AVで使うaligned windowの端だけをゼロにする。毎headの全確率バッファ初期化を省く。

いずれも今回の変更前と同じ量子化範囲、exp近似、浮動小数点の積の順序を使う。
整数和の組み替えは、既存のi32上限内で行う。端数の4-lane/scalar処理区分も維持した。

最終計測用ビルドの段階別合算。通常公開queryの命令数とは区別する。

| 段階 | 直前版 | 今回 | 削減率 |
| --- | ---: | ---: | ---: |
| GEMM・水平加算・scale | 3,341,412,073 | 3,288,235,776 | 1.59% |
| QK | 381,244,404 | 233,935,992 | 38.64% |
| Softmax＋AV | 451,575,144 | 407,044,272 | 9.86% |

融合により量子化と分母のコストがAVからSoftmaxへ移ったため、この2段階は合算で比較した。
Kの配置変更ではpack時のコストが約73万命令増えたが、QKで約1.47億命令削減した。
[最終内訳](final-profile-summary.json)、[層別データ](final-profile.json)、[比較元](../goal-profile/final-summary.json)。

64×48 tileはPocketICのコンパイル時に「Code for function is too large」で拒否された。
64×32 tileは成功しGEMMを約2.35%減らせたが、Wasmの増大を抑えるため32×32を採用した。
[64×32の単体112ケース](linear-64x32.json)、[段階別記録](qk-stage-summary.json)、[拒否記録](rejected-large-tile.json)。
最終通常Wasmは2,961,775 bytesで、直前版2,638,097 bytesより約12.3%大きい。
初回コンパイルは依然重く、今回の通常版はinstall約117.9秒だった。コンパイル時間改善の保証はしない。

## 出力と待ち時間

- 既存の公開API 16ケースで、直前版のF32 embeddingとbit単位で一致した。
- ランダムtoken列、token 0の反復、語彙末尾tokenの反復を、2/3/7/8/31/32/63/64/65/127/128/129/255/256/257/512 tokensで試した。追加48ケースすべて直前版とbit単位で一致した。
- 最終GEMMの単体112ケースでnativeの整数参照hashと完全一致。タイル境界・ゼロ入力・非ゼロ入力を含む。
- profiling有無の256 tokens・4文書もbit単位で一致した。
- unit test 9件、入力拒否3種類、384次元・有限値・L2 norm・payloadサイズ・実際のquery命令制限を検査した。

FP32参照との最小cosineは0.999486899で、直前版と同じ。
検索タスクのRecall@kなどは未評価であり、embeddingのcosineを検索品質の維持率とは解釈しない。
追加48ケースは境界・入力変動の検査で、実文書の検索benchmarkではない。

キャッシュを避けた4文書×5回の256 tokens測定では、中央値は直前版 **0.687688秒**、今回 **0.645652秒**。
観測上は約6.1%短いが、測定範囲は重なり、OS負荷・温度等を管理した統計的検証ではない。
診断op=7が使わないkstartフィールドを毎回変え、入力tokensと計算を維持してqueryキャッシュを避けた。
最初の同一requestの反復は約0.0004秒のcache hitになったため棄却し、比較には使っていない。
[直前版の反復](repeat-baseline.json)、[今回の反復](repeat-single.json)、[棄却した測定](rejected-cached-repeat.json)。

## 再現と保存物

READMEのfixture生成後、`python3 scripts/validate_further.py`。
`CARGO_TARGET_DIR`でビルド先を変更でき、`GRANITE_WASM`で検証する通常Wasmを指定できる。
計測用は`profiling` featureでビルドし、`granite-bench 256 384 --micro --micro-extended --probe`を実行する。

[通常版の環境・source hash](environment.json)、[通常Wasmのexport検査](module-exports.json)、[公開API一覧](summary.json)、
[追加48ケース](stress.json)、[比較元48ケース](stress-baseline.json)、[計測版の環境](profile-environment.json)。
通常Wasm SHA256: `139b99f554d4680ba8af918f506ab000821fece11b3481215ecd91f4f3eeb116`。

外付けSSDの容量不足でnativeのリンクが失敗したため、以降のビルドを内蔵SSDへ移した。
このプロジェクトの未使用native中間ファイルを約1.01GB削除し、空きは約1.18GBになった。
ソース、モデル、計測結果、通常Wasmは保持した。[削除対象と前後容量](disk-cleanup.json)。

mainnetへのデプロイは行っていない。重み・RoPEはheapに保持し、upgrade後の復元は未実装。
