# Bekko a25m memory study

This is an offline experiment. It does not change the production model, canister, pack, or embedding-space ID. Results and limits are in [RESULTS.md](../../benchmarks/bekko25-memory-study/RESULTS.md).

The Rust program loads the normal F32 pack into the current native `bekko25-core`. For each test set it rounds only the token rows used by that set, reconstructs them as F32 before model loading, and compares against an unmodified native run. It does not implement a compressed pack or measure Wasm instructions. Evaluation vectors stay in `.cache/bekko-memory-study/`.

Prerequisites: the pinned `.cache/bekko25/model.pack`, `.cache/bekko25/cases.json`, `.cache/bekko25/retrieval.json`, Python with NumPy, and the repository's cached Rust dependencies. Japanese evaluation also requires the JaQuAD files under `.cache/answer-spans/`, Python `tokenizers`, and the pinned Bekko tokenizer JSON. Run from the repository root:

```sh
mkdir -p .cache/bekko-memory-study
python3 experiments/bekko25-memory-study/storage.py > .cache/bekko-memory-study/storage.json
CARGO_TARGET_DIR=.cache/bekko25-prod-target cargo run --release --offline --locked --manifest-path experiments/bekko25-memory-study/Cargo.toml --bin bekko-memory-study > .cache/bekko-memory-study/numerical.json
CARGO_TARGET_DIR=.cache/bekko25-prod-target cargo run --release --offline --locked --manifest-path experiments/bekko25-memory-study/Cargo.toml --bin retrieval > .cache/bekko-memory-study/retrieval-vectors.json
python3 experiments/bekko25-memory-study/analyze.py > .cache/bekko-memory-study/retrieval-summary.json
python3 experiments/bekko25-memory-study/cross.py > .cache/bekko-memory-study/cross-index.json
```

Set `BEKKO_TOKENIZER_JSON` to the pinned revision's `tokenizer.json`, then generate and evaluate both Japanese selections:

```sh
python3 experiments/bekko25-memory-study/make_japanese.py --tokenizer "$BEKKO_TOKENIZER_JSON"
python3 experiments/bekko25-memory-study/make_japanese.py --tokenizer "$BEKKO_TOKENIZER_JSON" --related --out .cache/bekko-memory-study/retrieval-ja-hard.json
CARGO_TARGET_DIR=.cache/bekko25-prod-target cargo run --release --offline --locked --manifest-path experiments/bekko25-memory-study/Cargo.toml --bin retrieval -- --ja > .cache/bekko-memory-study/retrieval-ja-vectors.json
CARGO_TARGET_DIR=.cache/bekko25-prod-target cargo run --release --offline --locked --manifest-path experiments/bekko25-memory-study/Cargo.toml --bin retrieval -- --ja-hard > .cache/bekko-memory-study/retrieval-ja-hard-vectors.json
python3 experiments/bekko25-memory-study/analyze.py --ja > .cache/bekko-memory-study/retrieval-ja-summary.json
python3 experiments/bekko25-memory-study/analyze.py --ja-hard > .cache/bekko-memory-study/retrieval-ja-hard-summary.json
python3 experiments/bekko25-memory-study/cross.py --ja-hard > .cache/bekko-memory-study/cross-index-ja.json
```

Input SHA-256 values for the reported run:

- `.cache/bekko25/retrieval.json`: `56b4fdfdd1ad87bd800ee35443983a192baeac6fe48fe06fbc026eaf55fef771`
- Generated Japanese distinct-article selection: `571407acb9e2f2c291f552a6ec2fa5d2a9cb665dd1f24851d5990e7e1b0d45fa`
- Generated Japanese same-article selection: `60a37fa51fab4a2ce0280dd24d6c7f2c8db987b76e7774067ebd46c6c77c98db`
- Pinned tokenizer JSON: `8bd47075711f75a143d1b78e01a41cc65c1c591b00d3cfeffc23db07adce1392`

Times in JSON are incidental wall-clock readings from concurrent native work. They are not latency or throughput measurements.
