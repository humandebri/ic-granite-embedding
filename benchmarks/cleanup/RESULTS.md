> 過去段階の記録です。現在の通常版・5,000 tokens対応は[最新レポート](../full-context/RESULTS.md)を参照してください。

# 旧実装・実験経路の整理

2026-10-27。計算結果と公開上限を維持し、現行の実装・検証経路へ整理した。
公開Candidは変更せず、300 tokens / 1 query、1,365 tokens / 7 queryを維持する。

## 削除・簡素化

- `scripts/validate_single_query.py`
- `scripts/validate_goal.py`
- `scripts/validate_optimized.py`
- `scripts/validate_further.py`
- `scripts/collect_results.py`
- `benchmarks/goal-profile/rejected/strassen.rs`

- 現在も使う`validate_exhaustive.py`は`validate_regression.py`へ改名し、全体検証から呼び出す。
- ハーネスの`--legacy`、`--fused`、旧`--plan`、旧境界走査とtile位置引数を撤去。
  現行のオプションは`granite-bench --help`で表示し、不明なオプションはモデルのロード前に拒否する。
- StrassenはWi/Woに使う1段だけにし、再帰・Split型・96/48列leafを削除。
- `strassen2`、`strassen3`、`compact-gemm`、`compressed-carry`のfeatureを削除。
- 採用済みkernel形状を常用し、`instruction-gemm`の切替と旧小タイル側の経路を削除。
- 圧縮stateの生成はテスト用fixtureに限定し、新規stateはraw F32に統一。
- `granite_canister_instruction.wasm`という旧別名artifactを削除。現在の成果物は標準名の`granite_canister.wasm`。

ハーネスは1,167→949行、Strassen実装は329→254行になった。
過去のJSON・計測レポートは比較用に保持し、旧runnerが残っているように見えるリンクと現行手順を修正した。
[対象一覧](scope.json)。

## 維持した経路

現在のquery API、診断用`step`、profiling、独立したnative参照を維持した。
旧圧縮stateはdecoderで引き続き読める。`miniz_oxide`はその互換処理とテストで使うため依存を保持した。
既存の低水準診断処理はCandidを含めて維持し、旧CLIによる演算分割のオーケストレーションを削除した。

## 検証

- native unit test 10件。Strassenは採用済み1段を384/1536列・INT8極値・異なるscaleでi64直接計算と比較。
- 計測版のGEMM 112ケース、AV 44ケースが参照と完全一致。
- 計測版の256 tokens・4文書と1,365 tokensの層別出力が整理前とbit単位で一致。
- 通常版の旧入力域全511長、追加57ケース、公開API 16ケースを検証。
- 長文96ケースと長文参照11ケースも整理前とbit単位で一致。
- 1 query境界96ケース、query上限、payload、有限値、384次元・L2 normを検証。
- 通常Candid一致とprofileStep非公開を確認。
- `--help`の表示と旧CLIの明示的な拒否を確認。

[通常版の環境・source SHA](../validation/environment.json)、[短文の比較](../validation/regression/environment.json)、[長文出力](../validation/stress-long.json)。
整理後の時間は性能比較として再測定していない。並行する検証suiteの時間を、改善の根拠として使わない。
性能基準と長文方式の詳細は[直前の最適化レポート](../frontier/RESULTS.md)を参照。

## 成果物と再実行

`target/wasm32-unknown-unknown/release/granite_canister.wasm`に検証済み通常版を配置した。
SHA256: `df0dc61b1aec685f4d82180a6a4e54abf336a2c6da3d3b1d8c71d9ed4cb7b887`、6,152,406 bytes。
[artifact一覧](artifacts.json)。

[開発手順](../../docs/DEVELOPMENT.md)の準備後、`python3 scripts/validate_frontier.py`で全体を、
`python3 scripts/validate_regression.py`で旧入力域の比較を再実行できる。
検証先は`benchmarks/validation/`。旧入力域のgoldenと長文のgoldenは過去の計測データを参照する。
