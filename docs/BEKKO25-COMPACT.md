# Bekko a25m compact canister

`bekko25_compact` is a separate canister and embedding space. It stores the token vocabulary as 256,000 independently readable signed INT8 rows, each with a little-endian F32 scale. Other model weights retain the existing quantization. The 384-dimensional result differs numerically from the F32-vocabulary canister; retrieval quality must be evaluated on the application's actual corpus. The two spaces must not share document or query vectors.

The compact artifact is built from the same shared API implementation, with a distinct storage magic and pinned compact pack. It does not change the old canister's storage or API. An attempt to upgrade an existing F32 canister directly to the compact artifact rejects its old storage; it is not a migration. Existing stable memory cannot be shrunk by merely replacing the bytes. Use a new canister ID for the compact pack and retain the original until the new index has been verified and switched over.

## Build and local validation

First prepare the normal pinned a25m pack as described in [BEKKO25.md](BEKKO25.md). With NumPy installed, convert it without modifying the original:

```sh
python3 scripts/prepare_bekko25_compact.py
python3 scripts/build_bekko25.py --target-dir .cache/bekko25-prod-target
python3 scripts/build_bekko25_compact_recipe.py --target-dir .cache/bekko25-prod-target
```

The converter validates the original pack and publishes its output only after verifying the exact 124,516,506-byte compact pack and SHA-256 `71b8c47a0a508dd5cd3e704d83934e900c8e634d2c52e7a02a85e882fa5be9c4`. The result is `.cache/bekko25/model-int8.pack`. A different tokenizer, numerical conversion or model revision requires a separate artifact and evaluation; do not silently replace the pinned file.

The recipe helper validates the raw build, runs the pinned ICP recipe, then records `icp.yaml`, raw Wasm, ICP CLI version and the exact recipe artifact at `.cache/bekko25-prod-target/compact-recipe/bekko25_compact.wasm`. It refuses to attest the result if the source, build record or raw Wasm changes during the recipe build. Rebuild the recipe after any configuration or source change.

The `bekko25-compact-verify` host tool compares separate disposable PocketIC canisters, checks all 255 supported lengths, repeat-token boundaries and natural-language inputs, and checks incompatible storage rejection and recovery. Set `CARGO_TARGET_DIR` to the recorded build directory and provide `BEKKO25_BASELINE_WASM`, `BEKKO25_BASELINE_SHA256`, `BEKKO25_COMPACT_WASM`, `BEKKO25_COMPACT_SHA256` and `POCKET_IC_BIN`. `BEKKO25_COMPACT_WASM` must be the **recipe artifact**, and its hash must match `compact-recipe/record.json`. The tool verifies source, build record, executable, input files and recipe artifact before and after testing, and confirms PocketIC installed that module hash. The baseline should be the trusted F32 release being compared; a caller-supplied hash checks identity but does not certify its quality. The pinned six-language cases and Japanese fixture are described in [product verification](../benchmarks/bekko25-stable-read/PRODUCT.md).

The local regression gate requires cosine similarity of at least 0.97 on every sampled pair. On the 50-question Japanese fixture, hits@5 must not decrease, hits@1 may decrease by at most one, and MRR by at most 0.02. These tolerances catch large regressions; passing them does not establish quality on an application's own data. The verifier archives any prior detailed result before starting, publishes a new result only after all gates pass, and uses a lock to prevent concurrent runs or reports. Run `python3 scripts/report_bekko25_compact.py` with the same `CARGO_TARGET_DIR` to independently check the record and summarize `.cache/bekko25/compact-verification.json`. If a process was forcibly stopped and left `.cache/bekko25/compact-verification.lock`, verify no verifier or report is running before removing the stale lock. Results are local, not a mainnet performance guarantee.

After a successful verification and report, create a deliberately selected **new local canister** and install the exact verified recipe artifact (rather than asking `icp deploy` to rebuild it):

```sh
icp canister create bekko25_compact -e local --cycles 10t
icp canister install bekko25_compact -e local --mode install --wasm .cache/bekko25-prod-target/compact-recipe/bekko25_compact.wasm
CARGO_TARGET_DIR=.cache/bekko25-prod-target python3 scripts/verify_bekko25_compact_deployment.py --environment local --canister-id <CANISTER_ID>
.cache/bekko25-prod-target/release/bekko25-load --compact --environment local
```

Replace `<CANISTER_ID>` with the selected canister's principal; the status check queries that canister and rejects a different principal or module hash. `--compact` selects the compact pack, canister name, checksum and embedding-space ID. For an explicitly chosen canister or network, use the loader's `--canister`, `--environment` and `--identity` options. The loader checks the remote model and embedding space before uploading. Deploying or changing a mainnet canister requires a separate explicit decision and appropriate controller authority.

## Migration of an application index

1. Create and fund a separate canister, upload the pinned compact pack and activate it. Keep the existing F32 service available.
2. Recompute all document vectors with the compact service into a separate index. Persist and check `embedding_space` on both query and document vectors.
3. Evaluate retrieval and operational performance on representative questions and documents. Only switch query traffic after the complete new index is ready.
4. Retain the previous service and index for rollback until the application owner decides otherwise.

The compact embedding-space ID is `bekko-a25m-44f0b8af-vocab-i8row-w8a8-bf16kv-qk13-v1`. The Candid API, token ID validation, 256-token maximum, authorization and post-upgrade activation requirement match the existing service. The core's INT8-resident linear weights also optimize the old canister; this does not change its embedding space because integer arithmetic is unchanged. All optimizations must be revalidated for each release and platform. Ordinary query responses are not certified.
