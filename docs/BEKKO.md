# Bekko a8m canister

Bekko a8m has a dedicated canister in this repository. It accepts **2–256 token IDs including BOS/EOS**, executes all four layers in one query, applies mean pooling and returns a normalized 384-dimensional vector. The tokenizer runs on the client. Longer inputs are rejected rather than truncated.

The model is pinned to `hotchpotch/bekko-embedding-v1-a8m` revision `c721113d59a1d91b447450324f51c4b3332c924a`. Linear weights/activations use W8A8, the embedding table remains F32, and Attention uses F32 with BF16 K/V. See [validation results](../benchmarks/bekko/RESULTS.md).

## Repository layout

Keep the models in one repository for now, with separate crates, Wasm artifacts and canister IDs:

| Component | Purpose |
| --- | --- |
| `crates/bekko-core` | Fixed four-layer Bekko encoder, 6 × 64-dimensional heads, mean pooling |
| `crates/bekko-canister` | Dedicated API, controller-only weight loading, stable weight storage |
| `crates/bekko-bench` | Disposable PocketIC tests and local weight-loading client |
| `crates/kernel-gen` | SIMD GEMM generation shared by Granite and Bekko |
| `crates/core/src/strassen.rs` | Exact Strassen arithmetic compiled for each model's dimensions |

Granite and Bekko have independent public APIs and build targets. Splitting repositories now would require publishing/versioning the shared kernel code or synchronizing copies. Consider a split if ownership, release schedules or licensing requirements diverge. This implementation shares the numerical kernels without introducing a runtime-selectable model framework.

## Prepare and verify

Run from the repository root. The Python setup downloads public weights; generated weights, fixtures and build outputs stay under `.cache/`.

```sh
uv venv --python 3.12 .cache/bekko-venv
uv pip install --python .cache/bekko-venv/bin/python -r requirements.txt pyarrow==26.0.0
.cache/bekko-venv/bin/python scripts/prepare_bekko.py --retrieval

CARGO_TARGET_DIR=.cache/bekko-target cargo test --locked -p bekko-core
CARGO_TARGET_DIR=.cache/bekko-target cargo build --release --locked \
  -p bekko-canister --target wasm32-unknown-unknown
CARGO_TARGET_DIR=.cache/bekko-target cargo build --release --locked \
  -p bekko-bench --bin bekko-bench --bin bekko-load

POCKET_IC_BIN=/absolute/path/to/pocket-ic-11.0.0 \
CARGO_TARGET_DIR=.cache/bekko-target .cache/bekko-target/release/bekko-bench
```

`prepare_bekko.py` exports a 400,967,494-byte pack with the pinned SHA256, exact-length FP32/W8A8 fixtures and an optional small JaQuAD retrieval smoke set. `--reuse-pack <path>` reuses an existing identical export. Without `--retrieval`, PyArrow and dataset files are unnecessary. The retrieval smoke option reuses the pinned selection/chunking helpers under `experiments/multilingual/`.

The harness uses Application/Production limits and saves `benchmarks/bekko/validation.json`. It checks authorization, upload offsets, the final checksum, incomplete-upload recovery, input bounds, vectors and upgrade persistence. After the large stable-memory upload, it allows idle rounds to replenish the production heap-delta quota before upgrading; it does not raise query instruction limits. Its network is disposable and distinct from the live local network below.

## Run a local canister

`icp.yaml` uses a pinned Rust recipe and a random localhost gateway port. It does not stop other projects' networks. The commands below target `local` explicitly. Initial loading needs more headroom than the CLI's default 2T cycles; the example allocates 10T from the managed local network's seeded account.

```sh
icp network start -d
CARGO_TARGET_DIR=.cache/bekko-target icp deploy bekko -e local --cycles 10t
.cache/bekko-target/release/bekko-load

icp network status --json
icp canister call bekko modelInfo '()' --query -e local
.cache/bekko-venv/bin/python scripts/bekko_tokens.py \
  --text '文章の意味で関連する文書を検索したい。' \
  --output .cache/bekko/request.did
icp canister call bekko embed --args-file .cache/bekko/request.did --query -e local
```

Use `icp network status --json` for the current URL: the random port can change after restart. `icp deploy` prints the local Candid UI link. [The saved live smoke result](../benchmarks/bekko/live.json) records the endpoint and canister ID used for the measured run. The local network is left running for interactive use. `icp network stop` stops this project's local network when it is no longer needed.

## API and persistence

The [checked Candid interface](../crates/bekko-canister/bekko.did) is generated from the Rust declarations.

| Method | Call | Behavior |
| --- | --- | --- |
| `modelInfo()` | query | Fixed model/revision/hash, token limit, upload/activation status |
| `embed(vec nat32)` | query | Returns `Result<{ embedding; tokens; instructions }, text>` |
| `beginUpload()` | controller update | Starts/restarts the fixed weight upload |
| `upload(offset, blob)` | controller update | Appends up to 1,800,000 bytes, returns the next offset |
| `activate()` | controller update | Builds the inference cache from verified durable weights |

Uploads stream directly into stable memory and update SHA256 incrementally. The final chunk is accepted only if the complete pack matches the pinned checksum. A rejected final chunk does not advance the offset. Padding ID 0, out-of-vocabulary IDs, incorrect BOS/EOS and more than 256 IDs are rejected.

Stable storage is owned exclusively by this canister: bytes 0–7 contain `BEKKO001`, bytes 8–15 contain the uploaded length in little endian, byte 16 is the verified flag, and the pack starts at byte 64. **This storage version identifies the fixed model revision above. Changing the model/revision requires a new storage version/migration or a new canister.** It is not a stable layout shared with the Granite canister.

Verified weights survive upgrade. The heap inference cache is intentionally cleared; controller activation restores it without uploading weights again:

```sh
CARGO_TARGET_DIR=.cache/bekko-target icp deploy bekko -e local --mode upgrade
.cache/bekko-target/release/bekko-load --activate-only
```

An interrupted upload is discarded on upgrade because its transient hash state cannot be resumed; restart that upload. Upgrade tests use an actual upgrade, not reinstall. Loading a fresh model version and mainnet deployment are separate operations from this local setup.

Different embedding models use different vector spaces even when both output 384 dimensions. Re-embed documents with Bekko before searching them with Bekko query vectors, or keep model-specific indexes. The query result is an ordinary query response; no certification scheme is implemented here.

## RAG chunk length

The [answer-span comparison](../benchmarks/answer-spans/RESULTS.md) evaluates 128/256/512-token windows and sentence-end preference on all 3,939 public JaQuAD validation questions. A practical starting point is **256 tokens, preferring sentence endings**. The 512-token reference rows do not extend this canister's API limit. The evidence measures annotated answer/sentence retention; it does not evaluate generated answers.
