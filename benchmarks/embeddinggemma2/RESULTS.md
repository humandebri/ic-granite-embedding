# EmbeddingGemma 2 text-only experiment

Measured locally on 2026-10-28 JST. Model `21ad7b3a498496f5c3e5de0d9afecd706e172a04`. No mainnet deployment; the Bekko service was not changed. The experimental crate has its own workspace inside this repository.

## Retrieval comparison

Same pinned datasets, 128 questions per dataset, positive documents plus 512 seeded random distractors, and candidate IDs/qrels as [the previous comparison](../multilingual-expanded/RESULTS.md). SciFact has 642 candidate documents, FiQA 844, JaQuAD 635. These are bounded candidate-pool results, not full MTEB or a general model ranking. No training-overlap audit was performed.

The metric is **parent-document Recall@5**, scoring each parent by its highest-scoring chunk, then deduplicating parents. It is different from [answer-containing chunk recall](../answer-spans/RESULTS.md); this experiment did not evaluate answer spans or generated answers.

| Dataset | Gemma 2 FP32 | Gemma 2 W8A8/BF16 KV | Bekko a8m FP32 | Bekko a8m W8A8/BF16 KV |
| --- | ---: | ---: | ---: | ---: |
| scifact | 95.31% | 96.09% | 86.33% | 86.33% |
| fiqa | 72.88% | 72.99% | 70.30% | 69.52% |
| jaquad | 94.53% | 95.31% | 96.09% | 96.09% |

All model inputs fit 256 tokens including task prefixes and specials; document chunks have 32 source-token overlap. Query prefix is `task: search result | query: `, document prefix is `title: none | text: `. Original document titles remain in the document body, matching the prior title/body chunking protocol; title fields were not separately tuned or repeated per chunk. This is a reproducible default-prompt comparison, not a prompt optimization study. Mean pooling includes prompt/special tokens, then L2 normalization. Dimensions: 768. The official Sentence Transformers module pipeline matched the adapter exactly on four English/Japanese texts ([adapter.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/embeddinggemma2/adapter.json)).

The simulated quantization replaces every Linear with per-output-channel symmetric W8 and per-token dynamic A8, retains embeddings/norms as F32, and rounds Attention K/V to BF16. FP32 means BF16 checkpoint weights promoted to F32 for computation. All accuracy results above are Torch reference results; actual Wasm retrieval was not evaluated.

## Actual canister execution

PocketIC 11.0.0, Application/Production defaults: query ceiling 5 billion instructions; update ceiling 40 billion. The independent Rust implementation executes all 24 layers, projection-only per-layer inputs (PLE), grouped Attention, head norms, RoPE, GELU MLP, output projection, mean pooling and L2 normalization. The W8 matrix products use a basic four-output-channel i16 SIMD kernel, without the Bekko-specific generated/batched/Strassen optimizations. This is an initial implementation result, not a lower bound or proof that further optimization cannot work.

For this fixture probe only, the exported scaled token embeddings replace the vocabulary lookup. No encoder layers are skipped or precomputed; PLE executes in the canister. Synthetic repeated real Japanese/English tokens exercise exact lengths (BOS/EOS included). This is not a production token-ID API and does not validate full-vocabulary loading, upgrades or production authorization/storage behavior.

| Tokens (Japanese) | Call | Instructions | Cosine to W8 reference |
| ---: | --- | ---: | ---: |
| 2 | query | 552,680,787 | 0.99971779 |
| 16 | query | 4,550,423,457 | 0.99932886 |
| 32 | query | Instruction limit exceeded | — |
| 32 | update | 9,444,711,635 | 0.99901352 |
| 64 | query | Instruction limit exceeded | — |
| 64 | update | 20,265,241,033 | 0.99899286 |
| 128 | query | Instruction limit exceeded | — |
| 128 | update | Instruction limit exceeded | — |
| 256 | query | Instruction limit exceeded | — |
| 256 | update | Instruction limit exceeded | — |

English query fixtures cover the same 2/16/32/64/128/256 lengths; see [canister.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/embeddinggemma2/canister.json) for all measurements and errors. Update was tested on Japanese fixtures only. Numerical cosine values are measured agreement, not bit equality or a retrieval-quality guarantee. The minimum Wasm/native cosine is 0.99900707; strict agreement at 0.999999 was **not achieved** on every case. [native.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/embeddinggemma2/native.json) records a layer trace: the first 13 layers of the 2-token case agree within tiny floating-point errors, after which dynamic quantization thresholds amplify differences. This trace narrows the discrepancy but does not prove all longer-input differences have the same cause. The approximate Rust path therefore still needs numerical investigation and Wasm retrieval validation before serving real inputs.

## Capacity and feasibility

The text-only model has **271,002,624 parameters**, including **134,217,728 vocabulary embedding parameters** and **136,708,096 Linear parameters**. For comparison, Bekko a8m has about 7.7M non-embedding parameters. Total model size alone obscures this large difference in per-token computation.

The exported non-vocabulary pack is 137,912,256 bytes. Adding the full vocabulary in F32 would add 536,870,912 bytes, for roughly 675 MB of packed weights. The probe expands W8 weights to i16 for SIMD, so live/peak memory exceeds pack size. Full-vocabulary residency was not measured; these sizes suggest memory is feasible within wasm32's 4 GiB, while measured execution exceeds the desired query budget. Output truncation/MRL reduces vector storage and search cost; it does not remove the 24-layer encoder cost.

**Decision:** keep Bekko a8m for the current 256-token single-query service. Gemma 2 is a useful English-quality comparison, but this implementation does not achieve 256 tokens per query. A production port would need substantial kernel work or an explicit multi-message inference design, plus full-vocabulary loading and Wasm retrieval validation.

## Reproduction

Use the Python environment/dependencies of the previous experiment (Torch 2.14.1, Transformers 5.20.2, Sentence Transformers 6.2.0). First download the public pinned `google/embeddinggemma-2` snapshot with JSON, safetensors and tokenizer model files into a local Hugging Face cache. The official text-only configuration disables `vision_config` and `audio_config`; no image/audio weights are instantiated.

```sh
# P is the pinned snapshot directory; PY is the existing evaluation Python.
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/gemma2.py --snapshot "$P"
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/export_gemma2.py --snapshot "$P"
HF_HUB_OFFLINE=1 "$PY" experiments/multilingual/validate_gemma2.py --snapshot "$P"
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --lib --target wasm32-unknown-unknown --locked
CARGO_TARGET_DIR=.cache/gemma2-target cargo build --manifest-path experiments/embeddinggemma2/Cargo.toml --release --bin gemma2-probe --locked
.cache/gemma2-target/release/gemma2-probe --native
POCKET_IC_BIN=/path/to/pocket-ic-11 .cache/gemma2-target/release/gemma2-probe
"$PY" experiments/multilingual/gemma2_report.py
```

The experiment uses the repository's Wasm SIMD rustflags. Cached vectors are local/ignored; JSON summaries retain their SHA256, per-query ranks, source selection hashes and encoding statistics. The report generator independently recomputes all 768 per-query rankings from saved vectors. Fixtures/pack hashes: [fixtures.json](fixtures.json). Build hash and ranking verification: [verification.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/embeddinggemma2/verification.json).

Primary references: [official text-only inference](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers), [model and license](https://huggingface.co/google/embeddinggemma-2), [pinned model configuration](https://huggingface.co/google/embeddinggemma-2/blob/21ad7b3a498496f5c3e5de0d9afecd706e172a04/config.json), [IC resource limits](https://docs.internetcomputer.org/references/resource-limits/). The model is Apache 2.0; dataset attribution and limitations remain those in the linked prior experiment.
