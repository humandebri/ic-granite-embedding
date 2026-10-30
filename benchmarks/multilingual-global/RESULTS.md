# English-first multilingual RAG comparison

Measured locally on 2026-10-28 JST. These are bounded retrieval experiments and disposable local canister probes; no mainnet deployment or change to the Bekko service.

Follow-up: [a25m runtime optimization](../bekko25-runtime/RESULTS.md) subsequently achieved 256 tokens in one query at 4,921,718,962 instructions with additional QK13 quantization. Its precision tradeoff is evaluated separately at 256 tokens. The original probe measurements below remain unchanged.

## Retrieval at 128 tokens

Each cell is the equal-domain mean of NanoSciFact and NanoFiQA2018 **parent-document Recall@5**: 50 queries per domain, all their relevant documents plus 128 seeded random distractors. Same query IDs, document IDs and qrels across all six languages; translations were aligned by ID, never row order. Document score is its maximum chunk similarity, then parent IDs are deduplicated. These are not answer-containing chunk recall or generated-answer accuracy.

| W8A8/BF16 KV reference | English | German | Spanish | French | Arabic | Korean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite English | 84.73% | — | — | — | — | — |
| Granite 97M multi | 81.86% | 80.24% | 78.11% | 80.48% | 69.84% | 69.11% |
| Bekko a8m | 83.71% | 79.01% | 78.91% | 75.83% | 75.74% | 74.29% |
| Bekko a25m | 85.32% | 78.95% | 80.09% | 79.43% | 78.77% | 75.51% |
| multilingual E5 small | 78.67% | 72.64% | 72.60% | 70.36% | 71.55% | 71.93% |
| EmbeddingGemma 2 | 87.66% | 78.85% | 80.25% | 79.83% | 77.98% | 73.47% |
| Harrier 270M | 87.21% | 84.34% | 83.05% | 78.03% | 78.43% | 75.12% |

The English-only Granite baseline is intentionally evaluated only in English. All other models use their pinned revisions from `global_eval.py`. Each input fits 128 tokens including published query/document prefixes and specials, with 32 source-token overlap. Mean/CLS pooling follows the model; Harrier uses last non-padding token and its published web-search instruction. Its adapter matched Sentence Transformers exactly on four EN/AR/KO texts ([adapter check](harrier-adapter.json)). Gemma 2 loads text encoders only. Every vector is L2 normalized. For the original 2026-10-28 evaluation, FP32 and W8A8 results, individual domain scores and per-query ranks are retained in the [historical per-run records](https://github.com/humandebri/ic-granite-embedding/tree/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/multilingual-global) and [summary.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/multilingual-global/summary.json). Per-run JSON files from later evaluations are local and ignored by Git; pinned selections and model metadata remain tracked.

Completed CPU calculations were retained, and the remaining reference calculations ran on local Apple MPS in F32. A two-text EN/AR CPU/MPS check for each of Gemma 2 and Harrier found maximum FP32 differences below 2.1e-7 ([device check](device-check.json)). This is a limited numerical check, not a proof of bit equivalence; W8 rounding can amplify backend differences. Each encoding record states its device (legacy cached records without a device field were CPU). Timings are diagnostic and are not controlled speed comparisons. Wasm instruction measurements use PocketIC independently of these reference devices.

The public dataset contains translated versions of English retrieval sets, not independently collected native-language questions. Candidate pools are smaller than full NanoBEIR, and there are only 50 shared query IDs per domain; translations do not constitute six independent samples. No training-overlap audit was performed. Results do not establish a universal language ranking. The candidate pool/length differ from the older 256-token comparison, so do not interpret changes against its absolute percentages as regressions.

## Quantization effect

W8A8/BF16 KV minus FP32, in Recall@5 percentage points. Non-English is an equal-language/domain mean. Vector cosine is the minimum across all query/chunk vectors available for the model; high cosine alone does not guarantee identical rankings.

| Model | English change (pp) | Non-English change (pp) | Minimum vector cosine |
| --- | ---: | ---: | ---: |
| Granite English | -0.07 | — | 0.998652 |
| Granite 97M multi | +0.00 | -0.08 | 0.994168 |
| Bekko a8m | -0.12 | -0.15 | 0.994828 |
| Bekko a25m | -0.75 | +0.36 | 0.994292 |
| multilingual E5 small | +0.88 | -1.29 | 0.997615 |
| EmbeddingGemma 2 | +0.00 | -0.09 | 0.987924 |
| Harrier 270M | +0.00 | -1.14 | 0.992914 |

## English query → foreign-language documents

The same saved query/document vectors and identical relevance IDs allow cross-language scoring with no translation at inference time. Equal-domain mean Recall@5:

| W8A8/BF16 KV reference | → German | → Spanish | → French | → Arabic | → Korean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite 97M multi | 79.81% | 78.44% | 80.63% | 69.78% | 69.74% |
| Bekko a8m | 79.42% | 79.62% | 77.84% | 70.49% | 64.62% |
| Bekko a25m | 82.53% | 82.58% | 81.93% | 73.97% | 70.59% |
| multilingual E5 small | 65.28% | 71.95% | 72.42% | 62.74% | 44.80% |
| EmbeddingGemma 2 | 81.12% | 84.58% | 81.26% | 79.78% | 75.18% |
| Harrier 270M | 82.22% | 84.83% | 80.44% | 75.75% | 72.21% |

The reverse directions (foreign queries → English documents), both precision modes and per-task ranks are also stored in `summary.json`. These tests rank against one target language at a time; a mixed-language index with duplicate translations was not tested.

## Paired differences against Bekko baselines

Recall@5 differences in percentage points, W8A8 references. English uses the two sets of 50 queries. Non-English averages the five translations per query before resampling; stratified paired bootstrap resamples the 50 query IDs within each of two domains, 5,000 samples, seed 20261028. Intervals are exploratory and not adjusted for multiple comparisons.

| Comparison/scope | Difference (pp) | Paired 95% interval |
| --- | ---: | --- |
| granite-bekko8/en | -1.85 | [-6.52, +2.85] |
| granite-bekko8/non_english | -1.20 | [-4.17, +1.72] |
| bekko25-bekko8/en | +1.61 | [-0.83, +4.43] |
| bekko25-bekko8/non_english | +1.80 | [-0.85, +4.49] |
| e5-bekko8/en | -5.04 | [-11.27, +0.95] |
| e5-bekko8/non_english | -4.94 | [-7.99, -2.04] |
| gemma2-bekko8/en | +3.95 | [-0.66, +9.10] |
| gemma2-bekko8/non_english | +1.32 | [-0.62, +3.28] |
| harrier-bekko8/en | +3.50 | [-0.50, +7.76] |
| harrier-bekko8/non_english | +3.04 | [-0.48, +6.71] |
| gemma2-bekko25/en | +2.35 | [-1.93, +7.05] |
| gemma2-bekko25/non_english | -0.47 | [-3.13, +2.25] |
| gemma2-bekko25/en_to_ar | +5.81 | [+0.79, +11.25] |
| harrier-bekko25/en | +1.89 | [-1.41, +5.63] |
| harrier-bekko25/non_english | +1.24 | [-1.48, +4.08] |
| harrier-bekko25/en_to_ar | +1.77 | [-4.18, +7.54] |

## Actual Gemma 2 layer continuation

Full-token hidden states are carried by the client across **12 query calls**, two complete layers per call. Each call recomputes only the required projection-only PLE slice from the original token embeddings, which is mathematically equivalent to the original full projection under the per-row/per-token quantization scheme. No input chunk pooling replaces full attention. All 24 layers execute. Native split/full inference is bit-identical on the exact-length 32/64/128/256 fixtures in Japanese and English, plus the added Arabic/Korean 128 fixtures.

PocketIC 11, Application subnet / Production defaults, actual 5B/query ceiling:

| Fixture | Queries | Max instructions/query | Total instructions | Candid request+reply bytes | Cosine to Torch W8 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| ja-128 | 12 | 4,650,335,019 | 49,635,030,519 | 9,178,699 | 0.99896579 |
| en-128 | 12 | 4,649,090,021 | 49,627,882,671 | 9,178,699 | 0.99915226 |
| ar-128 | 12 | 4,650,044,836 | 49,641,689,136 | 9,178,699 | 0.99897063 |
| ko-128 | 12 | 4,649,325,162 | 49,636,025,925 | 9,178,699 | 0.99900530 |

These fixtures substitute exported scaled input embeddings for the vocabulary lookup only. All model computation after lookup executes at runtime. Full-vocabulary loading and a production token-ID API were not implemented. Rust/Wasm and Torch remain numerically approximate: dynamic quantization thresholds amplify small floating-point differences, and the original [numerical limitations](../embeddinggemma2/RESULTS.md) still apply. A full Wasm retrieval evaluation was not performed; the quality tables above measure the Torch references.

At 256 tokens, one layer per query passes the first five layers but the first wide global-Attention layer still exceeds 5B. Its completion is not claimed. At 128 tokens, the measured ~9.2 MB aggregate transfer and 12 sequential calls are material costs; mainnet latency was not measured. The client-provided intermediate state is unauthenticated and this API is only an execution prototype.

## Actual Bekko a25m query probe

The isolated port uses the a8m kernels with all 13 layers, the a25m weights and the matching RoPE/Attention settings. Default Production query limit, controller-only local uploads, no changes to the a8m service. The vocabulary lookup is fully loaded for this probe.

| Fixture | Instructions/query | Cosine to Torch W8 reference |
| --- | ---: | ---: |
| en-64 | 1,152,806,372 | 0.99931077 |
| en-128 | 2,607,834,903 | 0.99921485 |
| en-256 | limit exceeded | — |
| ar-64 | 1,152,806,372 | 0.99892335 |
| ar-128 | 2,607,834,903 | 0.99922621 |
| ar-256 | limit exceeded | — |
| ko-64 | 1,152,806,372 | 0.99850301 |
| ko-128 | 2,607,834,903 | 0.99934024 |
| ko-256 | limit exceeded | — |

Existing measured 128-token references: Bekko a8m about 0.811B/query ([dedicated validation](../bekko/RESULTS.md)); Granite 97M multilingual about 2.393B/query ([isolated validation](../multilingual/RESULTS.md)). E5 and Harrier have no measured canister implementation in this comparison; their reference accuracy does not establish a one-query budget.

## Harrier port validation did not pass

An initial independent 18-layer Harrier prototype was implemented with the same generic W8 SIMD matrix kernel, causal grouped Attention, Gemma 3 RMSNorm, RoPE, GELU and last-token pooling. Native full/split calculation is bit-identical on its fixtures, but agreement with the independent Torch W8 reference failed even the coarse cosine floor of 0.99. The first layer of an English 64-token trace closely agrees, then differences grow through later layers. The discrepancy was narrowed but not resolved, so **no canister execution costs or Wasm retrieval-quality claims are reported for Harrier**. This does not invalidate its separate Torch accuracy results. Evidence: [numerical gate and layer trace](harrier-validation.json), [fixture/model/pack hashes](harrier-fixtures.json).

The initial failed prototype remains under `experiments/embeddinggemma2/src/harrier.rs` for reproducible diagnosis. Its harness exits with code 2 on this gate failure and deliberately skips canister measurements. It is not a deployable Harrier service.

## Interpretation

For same-language retrieval under a 128-token single-query constraint, Bekko a25m is the strongest measured candidate here: its reference English and five-language means exceed a8m, while actual EN/AR/KO queries complete at about 2.608B instructions. These small quality gains have paired intervals that include zero, so they are promising rather than conclusive, and a25m Wasm retrieval remains unmeasured.

Gemma 2 has the highest English reference point estimate, but it takes 12 queries in this prototype and does not beat a25m on the five-language same-language mean. English-query → Arabic-document retrieval is a distinct case where Gemma's reference point estimate is higher; see the cross-language table and paired interval. A choice should therefore distinguish same-language search from cross-language search, rather than infer one winner from the English or Japanese score alone.

## Reproduction and evidence

Pinned public data: `LiquidAI/nanobeir-multilingual-extended` revision `8a4be55eb80b3ed4d2e9a423a5212228c217d426`. Download `{NanoSciFact,NanoFiQA2018}_{queries,corpus,qrels}_{en,de,es,fr,ar,ko}.parquet` to `.cache/multilingual-global/data`. Source file hashes and selected IDs/qrels are in `*-selection.json`. Data attribution follows the [dataset card](https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended) and its original NanoBEIR/SciFact/FiQA sources; redistribution of source texts is avoided here.

Use the existing Python evaluation environment (Torch 2.14.1, Transformers 5.20.2, Sentence Transformers 6.2.0). Download/cache models at the revisions in `global_eval.py`; Gemma 2 and Harrier paths are explicit in `model_path`. Run from the repository root:

```sh
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/global_eval.py
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/bekko25_probe.py
CARGO_TARGET_DIR=.cache/global-port-target cargo build --manifest-path .cache/multilingual-global/bekko25/port/Cargo.toml --release -p bekko25-probe --target wasm32-unknown-unknown --locked
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --lib --target wasm32-unknown-unknown --locked
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --bins --locked
POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/bekko25
POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/split
# Extra Arabic/Korean Gemma fixtures:
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/export_gemma2.py --snapshot "$GEMMA_SNAPSHOT" --cache .cache/multilingual-global/gemma-extra --report benchmarks/multilingual-global/gemma-extra-fixtures.json --languages ar,ko --lengths 128
GEMMA_CASES=.cache/multilingual-global/gemma-extra/cases.json GEMMA_OUTPUT=benchmarks/multilingual-global/gemma2-split-extra.json POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/split
"$PY" experiments/multilingual/global_report.py
# Optional Harrier diagnostic (expected exit code 2 for the recorded failed port):
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/export_harrier.py
.cache/gemma2-target/release/harrier
```

Initial JA/EN Gemma fixtures and the non-vocabulary pack are generated by the earlier `export_gemma2.py` default command. The report generator rederives all measured rankings from cached vectors, checks hashes, selections, finite unit vectors and zero truncation, computes both cross-language directions, and checks the successful 128-token query plans. Cached model/vector/pack files and regenerated per-run results are local and ignored. The report, fixture/model pins, selections and source/build hashes remain tracked; historical detailed results are linked above. See [verification.json](verification.json), [Gemma split calls](gemma2-split.json), [Arabic/Korean split calls](gemma2-split-extra.json), [Bekko a25m calls](bekko25-canister.json), [Bekko source/pack hashes](bekko25-fixtures.json).
