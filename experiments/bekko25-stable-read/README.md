# Stable vocabulary lookup experiment

This is a local experiment, not a deployable implementation. `prototype.patch` applies to the streamed-activation source that produced baseline raw Wasm SHA-256 `c198d9ade07132daf4c9a27151744b9381bb7ebfbbf149c6829bf7f70046cede`. It removes the resident vocabulary allocation and reads one 1,536-byte F32 row from stable memory for every token. It hard-codes the offset of the verified pinned pack, so it must not be used as a general pack reader.

The reviewed prototype preserves `load_from` and adds `load_from_external_embeddings`; the latter makes `embed` and `full` return an error and requires `full_from_rows` instead. The updated raw prototype Wasm SHA-256 is `0ce0de4b515a543a16a47d6f9ac34cc057261ebf317066bbc7010e83c4a15ef9`. This is a different build from the one in the original report.

The tested source hashes before applying the patch are:

- `crates/bekko-core/src/lib.rs`: `93610dc688996b651f49487020e135e42f33c1d9ccb6970c9ec20e1f4275a2f4`
- `crates/bekko25-canister/src/lib.rs`: `806338b2a859756899b35ca2c739c0df4b41fe0ec1a2e484d7f3f3038b494b6a`

The comparison tool `verify.rs` uses PocketIC 11.0.0, uploads the original verified pack once, measures the baseline, upgrades the same disposable canister to the prototype, and compares 314 inputs. It pins both Wasm hashes, checks that heap shrank by at least 390,000,000 bytes, and exits unsuccessfully if any query fails or any output differs. The required pack and six-language cases are under `.cache/bekko25/`; the Japanese inputs are generated as `retrieval-ja-hard.json` by [the earlier memory study](../bekko25-memory-study/README.md). Server logs may precede the final JSON on stdout.

For reproduction, copy the five crates (`kernel-gen`, `bekko-core`, `bekko25-core`, `bekko25-canister`, `bekko25-bench`) and workspace configuration into an isolated directory, preserve the original files above, apply the patch there, and place `verify.rs` and `api-check.rs` in the copied `bekko25-bench/src/bin/` directory. Build the canister with `wasm32-unknown-unknown` and the tools for the host using the pinned toolchain. Run the tools from the repository root with `POCKET_IC_BIN` set for the verifier. `api-check.rs` loads the real pack through both APIs, checks the external-mode errors and compares full outputs. The verifier's unit tests exercise mismatch, query-error, and memory-failure handling. Do not replace the live local canister or its ID mapping.

The results before and after review, including remaining limitations, are in [the experiment report](../../benchmarks/bekko25-stable-read/RESULTS.md).
