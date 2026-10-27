# Multilingual RAG experiment

[Measured results](../../benchmarks/multilingual/RESULTS.md) compare five pinned models on a bounded Japanese passage-retrieval subset, plus isolated Granite 97M and Bekko a8m Wasm ports.

The scripts create experiment copies under `.cache/multilingual/`. They do not edit the production Rust crates or change the default English model. Builders depend on the current checkout's source shape and deliberately are not a general model loader. `capture.py` records the materialized sources used for a run; compare those hashes before treating a later rebuild as the same experiment.

Run commands from the repository root. Output files in `benchmarks/multilingual/` and `.cache/multilingual/` are overwritten on reruns. Public model weights require a few GB of downloads/cache; Rust builds also create local build artifacts. No mainnet identity or deployment is involved.

## Python setup and retrieval

```sh
uv venv --python 3.12 .cache/multilingual/venv
uv pip install --python .cache/multilingual/venv/bin/python -r experiments/multilingual/requirements.txt

.cache/multilingual/venv/bin/python experiments/multilingual/evaluate.py --export
.cache/multilingual/venv/bin/python experiments/multilingual/evaluate.py --models=granite,bekko8 --limit=128
.cache/multilingual/venv/bin/python experiments/multilingual/evaluate.py --models=granite,bekko8 --limit=512
.cache/multilingual/venv/bin/python -m unittest discover -s experiments/multilingual -p 'test_*.py'
```

The default comparison uses 128 queries, 512 sampled distractors plus every relevant passage, and 256-token chunks. Revisions are fixed in `models.json` and in `evaluate.py` for the dataset. `--export` produces the two ports' weight packs and exact-length FP32/W8A8 reference fixtures. Export separately without retrieval using `export.py granite` or `export.py bekko8`.

Each passage is scored by its highest-scoring chunk, and document IDs are deduplicated before computing metrics. The data filter and this aggregation differ from full MTEB. The 12 handwritten Japanese → English pairs are only smoke tests. Full-corpus Wasm retrieval was not measured, and residual differences between Torch simulation and Wasm are reported in the results.

## Granite 97M Wasm

```sh
.cache/multilingual/venv/bin/python experiments/multilingual/build_granite.py
cargo build --release --locked --manifest-path .cache/multilingual/port/Cargo.toml -p granite-canister --target wasm32-unknown-unknown
cargo build --release --locked --manifest-path .cache/multilingual/port/Cargo.toml -p granite-bench

export POCKET_IC_BIN=/absolute/path/to/pocket-ic-11.0.0
"$POCKET_IC_BIN" --version
.cache/multilingual/port/target/release/granite-bench "$PWD"
```

## Bekko a8m Wasm

```sh
.cache/multilingual/venv/bin/python experiments/multilingual/build_bekko.py
cargo build --release --locked --manifest-path .cache/multilingual/bekko-port/Cargo.toml -p granite-canister --target wasm32-unknown-unknown
cargo build --release --locked --manifest-path .cache/multilingual/bekko-port/Cargo.toml -p granite-bench
.cache/multilingual/bekko-port/target/release/granite-bench "$PWD"
```

Bekko's port adapts hidden/head dimensions, intermediate size, RoPE and mean pooling. It uses F32 Attention with BF16 K/V; the existing integer Attention is specific to 32-dimensional heads. Reusing INT14 QK unchanged at width 64 can overflow i32 and is not supported. Generated packed GEMM includes the required 1152/576 column shapes.

The temporary canister implements only controlled upload, activation and whole-embedding queries (up to 512 tokens). Its `next_layer = 12` value is the old harness completion sentinel; the encoder itself executes four layers. This is an experiment adapter, not a production API proposal.

Both harnesses use a new, disposable PocketIC Application subnet and the actual Production instruction ceiling. They load each model once, then query three content patterns at each length. Query rejections at the measured upper lengths are saved as results, not converted into successful measurements. Each successful result must have 384 finite values and unit norm.

## Validate and capture results

```sh
.cache/multilingual/venv/bin/python experiments/multilingual/capture.py
.cache/multilingual/venv/bin/python experiments/multilingual/report.py
```

`report.py` checks common dataset-selection hashes, both arithmetic modes, zero truncation, query output shape/norm, actual instruction bounds, expected rejection cases, pack hashes and cosine against both FP32 and independent W8A8 references. The numerical acceptance floor of 0.99 is only a coarse consistency check. The report gives the actual observed values and explicitly does not claim numerical equivalence or Wasm retrieval quality.

The saved results were measured using the existing Python environment with the versions in `requirements.txt`. Native latency is not a controlled benchmark: compilation and other inference runs overlapped. Use query instructions for the resource comparison.

## Expanded English and Japanese comparison

The expansion pins eleven models in `expanded-models.json`, adds SciFact and FiQA test sets, and evaluates the six added models on the same Japanese pool as before. Each task selects 128 queries and all their positives plus 512 seeded distractors. English tasks have multiple positives per query, so Recall@5 is a fraction of all relevant documents rather than a yes/no hit rate.

```sh
.cache/multilingual/venv/bin/python experiments/multilingual/expanded.py \
  --models=english,granite,bekko8,bekko25,e5 --datasets=scifact,fiqa
.cache/multilingual/venv/bin/python experiments/multilingual/expanded.py \
  --models=bge,e5en,minilm,granite107,horizon,granite311 --datasets=scifact,fiqa,jaquad
.cache/multilingual/venv/bin/python experiments/multilingual/validate_adapters.py \
  --models=english,granite,bekko8,bekko25,e5,bge,e5en,minilm,granite107,horizon,granite311
.cache/multilingual/venv/bin/python experiments/multilingual/expanded_report.py
```

The adapter check additionally uses `sentence-transformers==6.2.0` (the version used for this expansion). Install it in the same environment if it is absent. `--resume` skips completed comparisons with matching model specs, selection hashes and both precision modes. `--threads` controls CPU inference threads; concurrent runs can use disjoint model lists. They must use identical query count, distractor count and token budget when sharing an output directory.

English BGE and E5 receive their published retrieval prefixes. Horizon's learned output projection is loaded, applied and quantized alongside its transformer Linear layers. `validate_adapters.py` compares the custom FP32 adapters with Sentence Transformers, including the projection. The expansion stores vectors and parent indices under `.cache/multilingual-expanded/`; `expanded_report.py` recomputes all new rankings from those saved vectors, checks normalization, zero truncation and exact selection matching, then calculates paired query-bootstrap intervals. Five Japanese results from the first experiment are reused only after verifying exact query/document/qrel equality.

Outputs are under `benchmarks/multilingual-expanded/`, with separate task directories. `.cache` vectors are needed to rerun report validation. They can be regenerated from the pinned models and datasets. These new comparisons do not imply new canister implementations or measured instruction counts for the added models.

For an interrupted precision pass, `--reuse-fp32` can reuse matching saved FP32 vectors while recomputing W8A8. It checks the model spec, selection hash, vector count and parent indices, and records the original FP32 environment and vector SHA256 in the result. The 311M model's FP32 pass used two CPU threads; its resumed W8A8 pass used six. Neither these timings nor the concurrent runs are controlled latency comparisons.
