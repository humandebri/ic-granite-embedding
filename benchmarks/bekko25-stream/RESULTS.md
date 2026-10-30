# Bekko a25m streamed activation

2026-10-30、PocketIC 11.0.0 の Application subnet で測定。既存の検証済み pack を保持したまま、旧Wasmから新Wasmへupgradeし、再アップロードせずに `activate()` した。[測定値](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-stream/verification.json)

| 指標 | 旧Wasm | 新Wasm | 差 |
|---|---:|---:|---:|
| Wasm heap | 985,595,904 B | 567,934,976 B | **-417,660,928 B（-42.38%）** |
| stable memory | 418,447,360 B | 418,447,360 B | 0 B |
| heap + stable | 1,404,043,264 B | 986,382,336 B | **-29.75%** |

これはactivate直後の `canister_status.memory_metrics`。canister全体のmemory sizeにはWasmや履歴などが加わる。開始から終了までの瞬間最大の測定ではない。旧・新とも同じpack、同じサブネット、同じcanisterで比較した。upgrade直後はverified維持・inactive、権限のないactivateは拒否され、controllerのactivateに成功した。

2〜256 tokensの各長255ケース、6言語×3長18ケース、日本語40ケースの**313ケースすべてでembeddingはbit単位で一致**し、token数とembedding spaceも一致。queryの観測最大命令数は旧4,803,063,081、新4,803,442,979。各ケースの差は最大増加1,421,464命令、最大減少484,044命令で、相対の最大増加は約0.0566%。284ケースで増え、29ケースで減った。5Bを超えるqueryはなかったが、**命令数の完全一致は達成していない**。これは各長1件の固定seedと追加例の範囲であり、全入力の上限を証明しない。

activateの壁時計は旧2.96秒、新1.02秒だったが、同一環境で各1回の観測であり、速度比較やmainnetの予測には使えない。アップロード所要時間は102.54秒。通常の推論とは別の処理。

- 旧 raw Wasm SHA-256: `22e396d7536d0274d15872bbfc1766849588448d678f43412bfaf9a06a1a106f`
- 新 raw Wasm SHA-256: `c198d9ade07132daf4c9a27151744b9381bb7ebfbbf149c6829bf7f70046cede`
- pack SHA-256: `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`
- `icp build bekko25 -e local` は成功。recipe処理後Wasm SHA-256: `ed29d9c592411f71d1ddf75c0232683cf408a7f37f448e6e8ebc41c3f1e7e928`。**上の実行比較はraw Wasmであり、recipe処理後の実行性能は未測定。**

再現には、上記SHA-256の旧raw Wasmを `.cache/bekko-memory-study/before-stream.wasm`、packと6言語casesを `.cache/bekko25/` に置き、[日本語の同記事selection](../../experiments/bekko25-memory-study/README.md)を生成してから、[比較プログラム](../../crates/bekko25-bench/src/bin/bekko25-stream-verify.rs)をPocketIC 11.0.0で実行する。PocketIC serverのログがstdoutに混ざる場合は最後のJSON objectを抽出する。ローカル実測であり、mainnetでの遅延・cycles・性能を保証しない。
