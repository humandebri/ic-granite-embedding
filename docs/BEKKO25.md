# Bekko a25m canister

The dedicated `bekko25` canister accepts **2–256 token IDs including BOS (2) and EOS (1)**, executes all 13 layers and returns a normalized 384-dimensional vector in one query. It has a separate Wasm artifact, canister ID and stable-storage version from a8m. The a8m service remains available independently.

Model: `hotchpotch/bekko-embedding-v1-a25m`, revision `44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c`. Linear operators use W8A8, the vocabulary table is F32, K/V are BF16-rounded, QK uses per-head QK13 integers with i64 final reduction, and AV accumulation is F32. Approximate GELU and softmax follow the existing Bekko runtime. The AV integer experiment was slower and is **not** part of this release.

The [operational validation report](../benchmarks/bekko25-operations/RESULTS.md) records the exact artifact, all input lengths, upload/upgrade checks and retrieval measured from actual Wasm outputs. [Earlier runtime experiments](../benchmarks/bekko25-runtime/RESULTS.md) explain the additional QK13 precision tradeoff; they are not substitutes for the public API tests.

## Prepare and build

Run from the repository root. The Python tokenizer runs on the client; the canister receives IDs, not UTF-8 text.

```sh
uv venv --python 3.12 .cache/bekko25-venv
uv pip install --python .cache/bekko25-venv/bin/python -r requirements.txt
.cache/bekko25-venv/bin/python scripts/prepare_bekko25.py

.cache/bekko25-venv/bin/python scripts/build_bekko25.py
```

The build helper builds **both a25m and a8m Wasm** plus all three host tools. The a8m Wasm is required by the validation harness's wrong-model upgrade/downgrade test. It records source fingerprints, compiler versions and artifact hashes in `.cache/bekko25-prod-target/build-record.json`, removing any old record before starting. Use `--target-dir PATH` and set the same `CARGO_TARGET_DIR` for subsequent checks when selecting another build directory. `--offline` uses only cached Cargo dependencies.

The exporter pins and verifies the complete 418,404,508-byte pack, SHA256 `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`. It writes `.cache/bekko25/model.pack` and tokenizer smoke cases. `--reuse-pack /absolute/path/to/model.pack` reuses an identical existing export. Basic export/tokenization does not need the retrieval dataset or PyArrow.

## Start and use locally

Use the existing project-local network if it is running. Otherwise start it with `icp network start -d`. The gateway uses a dynamic port; discover it with status rather than hardcoding a URL.

```sh
icp network status --json
CARGO_TARGET_DIR=.cache/bekko25-prod-target icp deploy bekko25 -e local --cycles 10t
.cache/bekko25-prod-target/release/bekko25-load --environment local

.cache/bekko25-venv/bin/python scripts/bekko_tokens.py --model a25m \
  --text 'Find documents about recovering access to an account.' \
  --output .cache/bekko25/request.did
icp canister call bekko25 embed --args-file .cache/bekko25/request.did --query -e local
icp canister call bekko25 modelInfo '()' --query -e local
```

The loader verifies the local pack before upload, verifies the remote model/revision/embedding space, resumes from its uploaded offset, retries transport failures up to three times and activates verified weights. Duplicate chunk retries must match the bytes already stored; they never update the hash twice. Use only one uploader per canister. `--identity`, `--canister` and `--pack` select an explicit identity, canister name/ID and pack path. `--environment` defaults to `local`.

Loader decisions and final readiness checks use `modelInfo` as a replicated query (an `icp canister call` without `--query`) to observe completed updates. After an acknowledged restart, upload starts at zero without re-reading progress. Interactive status queries may use `--query`, as above.

`--restart` explicitly begins a new upload and **deactivates the inference cache**. Ordinary repeated loader runs resume/reuse the existing weights. `--activate-only` requires no pack file and restores the heap cache after upgrade. No mainnet deployment is performed by the local setup commands above.

## API and index identity

The [Candid file](../crates/bekko25-canister/bekko25.did) is generated from Rust and checked during validation.

| Method | Call | Purpose |
| --- | --- | --- |
| `modelInfo()` | query | Model/revision/hash, embedding space, runtime ID, 13 layers, 256-token limit, upload and activation status |
| `embed(vec nat32)` | query | `Result<{ embedding; embedding_space; tokens; instructions }, text>` |
| `beginUpload()` | controller update | Reset upload offset/digest and deactivate the cache |
| `upload(offset, blob)` | controller update | Append 1–1,800,000 bytes or confirm an identical stored retry |
| `activate()` | controller update | Construct the cache from complete, checksum-verified durable weights |

Both query and document vectors must have the same `embedding_space`:

```text
bekko-a25m-44f0b8af-w8a8-bf16kv-qk13-v1
```

Store that ID with the document index and reject query vectors from another space. Rebuild the index when the model/revision or numerical semantics change; equal vector dimensions are insufficient. Bit-equivalent runtime optimizations keep the space ID. `runtime_id` identifies the implementation separately.

For RAG, split documents by sentences/paragraphs before tokenization, with up to 254 body tokens plus BOS/EOS and a modest overlap such as 32 source tokens. Use the same pinned tokenizer for both sides. The CLI refuses oversized inputs; the server rejects oversized vectors, missing framing, PAD 0 and out-of-vocabulary IDs. It never silently truncates. A lower chunk limit is available to the client without changing the embedding space, but changing document chunks requires re-embedding those chunks. Search normalized vectors by dot product/cosine and deduplicate parent documents after taking each parent's best chunk score.

## Upgrades and recovery

Activation reads the verified pack from stable memory into the final model allocations in bounded chunks. It does not allocate a second full pack in Wasm heap. The stable format and embedding API are unchanged. A local old-to-new upgrade measured a 417,660,928-byte heap reduction and bit-identical output for 313 cases; query instruction counts changed slightly. See [streamed activation verification](../benchmarks/bekko25-stream/RESULTS.md) for the exact artifacts and limits.

```sh
CARGO_TARGET_DIR=.cache/bekko25-prod-target icp deploy bekko25 -e local --mode upgrade
.cache/bekko25-prod-target/release/bekko25-load --environment local --activate-only
```

Verified weights remain in stable memory; the heap cache intentionally starts inactive after upgrade. Schedule the upgrade plus activation as a maintenance operation and confirm `modelInfo.active`, `weights_verified`, revision and embedding space before routing queries. Activation is idempotent. An incomplete upload is discarded on upgrade because its hash is transient; restart the loader afterward. A failed checksum leaves progress at the previous chunk, so the correct final chunk can be retried.

Storage ownership is exclusive: `BEK25001` at bytes 0–7, uploaded length at 8–15 (little endian), verified flag at 16, reserved header through byte 63, then the pinned pack. This magic identifies this fixed model revision. Unsupported storage versions produce an error; `beginUpload` does not overwrite them. **Deploy a new canister for a8m → a25m**, rather than upgrading an a8m ID. The isolated tests also check that a wrong-model upgrade refuses reinterpretation and the original loader works again after downgrade.

For any update to the pinned revision, storage schema, compiler, dependency graph or numerical implementation, rebuild and rerun the release checks below. Keep a copy of the previous Wasm and the matching pack, and record the target ID and artifact hash. Reinstall erases stable memory and is not the upgrade path. Inspect cycles and memory with `icp canister status bekko25 -e <environment>` and maintain funding for uploads, activation and future upgrades according to measured usage; the local 10T allocation is not a mainnet cost estimate.

## Release checks

```sh
# Downloads the pinned public multilingual data when absent.
uv pip install --python .cache/bekko25-venv/bin/python pyarrow==26.0.0
.cache/bekko25-venv/bin/python scripts/prepare_bekko25.py --retrieval
CARGO_TARGET_DIR=.cache/bekko25-prod-target cargo test --release --locked -p bekko-core
# Rebuild after changes to runtime or validation sources.
.cache/bekko25-venv/bin/python scripts/build_bekko25.py
BEKKO25_RETRIEVAL=1 CARGO_TARGET_DIR=.cache/bekko25-prod-target \
  .cache/bekko25-prod-target/release/bekko25-bench --preflight
BEKKO25_RETRIEVAL=1 CARGO_TARGET_DIR=.cache/bekko25-prod-target \
  POCKET_IC_BIN=/absolute/path/to/pocket-ic-11.0.0 \
  .cache/bekko25-prod-target/release/bekko25-bench
.cache/bekko25-venv/bin/python scripts/validate_bekko25_operations.py
```

The harness checks all prerequisites and the build record **before starting PocketIC or uploading weights**. `--preflight` runs just those checks. Validation needs Python 3 with the standard library (`python3` by default; override with `BEKKO25_PYTHON`). It verifies the running host executable, both Wasm artifacts and all recorded sources, then repeats those checks after the final lifecycle test. A complete `validation.json` is published only after they pass; it includes the build record and input/evidence hashes.

The independent verifier and `scripts/report_bekko25_operations.py` reject stale sources, changed binaries and verification records from another validation run. The report uses build-time source fingerprints instead of hashing whichever checkout happens to exist at publication time. Reports produced before this provenance format remain historical measurements; regenerating a release manifest requires a new harness run and independent verification, rather than attaching a new fingerprint to an old result. The reporter additionally requires the deployed-module evidence described below, with matching artifact hashes.

The report labels the 2026-10-29 a8m and Granite regression records as historical and links to their immutable repository snapshot. Those records are not checked by the a25m verifier or attested by the current build record. The live a8m comparison targets the original running service, rather than the newly built a8m artifact. After shared-kernel changes, rerun the a8m fixtures/lifecycle/retrieval and Granite generator comparison for the changed build before claiming current compatibility.

CI builds the a25m Wasm, compares generated Candid and runs the Rust loader and Python provenance/report regressions without downloading model weights. Full PocketIC inference and retrieval measurements remain separate release checks.

The harness uses a disposable Application/Production subnet with the real 5B query ceiling. It checks all 255 supported lengths, repeated IDs around tile boundaries, six-language natural inputs, authorization, checksums, duplicate retries, interrupted uploads and actual upgrades. The independent verifier recomputes saved retrieval rankings, validates hashes/vectors, and reserves at least 2% of the ceiling for this measured release. That budget is an artifact-specific release criterion, not a universal platform allowance. The numerical reference comparison needs the saved 256-token reference vectors from the earlier quality experiment; `--without-reference` explicitly omits that optional comparison for a clean environment. The actual Wasm retrieval rankings are still independently checked.

`icp` applies its pinned Rust recipe's metadata transforms after compilation. [The artifact record](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-operations/recipe-artifact.json) identifies both the raw Cargo artifact tested in PocketIC and the exact deployed artifact, and reproduces the recipe transform. The local-only `bekko25-live --sweep` tool checks all 255 lengths on the deployed module and the saved golden vectors; it also needs the historical fixtures/a8m smoke cache for its regression comparison. These are developer release checks rather than dependencies of the embedding API or loader.

The tests establish a bounded local release envelope, not mainnet latency, throughput or multi-node consensus performance. Embeddings are ordinary query responses; no application-level certified-data scheme is added. The retrieval set contains translated English tasks with small candidate pools, so use the saved per-language results alongside a sample of your own documents.
