# Multilingual embeddings for query-based RAG

Measured locally on 2026-10-27 (JST). **Bekko a8m is the leading candidate in this bounded experiment.** These results do not establish a general multilingual ranking.

## Japanese passage retrieval

Pinned JaQuAD validation subset: 128 queries, 635 passages (all relevant passages for these queries plus 512 seeded distractors). Filter: passage text has at least 80 Unicode characters, removing most answer-only items. Seed: 20261027. Titles are prepended. All models use their own tokenizer; all chunks fit the stated token budget, including special tokens and E5 prefixes. Chunks overlap by 32 source tokens. Document score is the maximum chunk cosine; top-k ranks deduplicate documents. This is document retrieval, not answer-containing chunk recall or end-to-end RAG accuracy.

FP32 uses original public weights converted to F32. W8A8 uses per-output-channel INT8 weights, dynamic per-token INT8 activations, and BF16-rounded K/V. Embedding tables and normalization stay F32. Torch simulates this arithmetic; it is not an INT8 speed benchmark.

| Model, 256-token chunks | FP32 Recall@5 | W8A8 Recall@5 | W8A8 Hit@1 | W8A8 nDCG@10 | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite Small English R2 | 63.28% | 62.50% | 38.28% | 0.5395 | 1459 |
| Granite 97M Multilingual R2 | 88.28% | 89.84% | 65.62% | 0.8054 | 1151 |
| Bekko a8m | 96.09% | 96.09% | 79.69% | 0.8946 | 939 |
| Bekko a25m | 93.75% | 93.75% | 75.00% | 0.8645 | 939 |
| multilingual-e5-small | 96.09% | 96.09% | 79.69% | 0.8892 | 955 |

The Japanese-query → English-document handwritten smoke test has only 12 easy cases. Correct top-1: Granite Small English R2 7/12, Granite 97M Multilingual R2 12/12, Bekko a8m 12/12, Bekko a25m 12/12, multilingual-e5-small 11/12. It is a functionality check, not a cross-lingual benchmark.

## Chunk length

| Model | Token budget | W8A8 Recall@5 | W8A8 Hit@1 | Chunks |
| --- | ---: | ---: | ---: | ---: |
| Granite 97M Multilingual R2 | 128 | 95.31% | 71.09% | 2282 |
| Granite 97M Multilingual R2 | 256 | 89.84% | 65.62% | 1151 |
| Granite 97M Multilingual R2 | 512 | 88.28% | 65.62% | 708 |
| Bekko a8m | 128 | 98.44% | 85.16% | 1794 |
| Bekko a8m | 256 | 96.09% | 79.69% | 939 |
| Bekko a8m | 512 | 96.88% | 79.69% | 655 |

128-token chunks perform best on this subset, but create roughly twice as many vectors as 256-token chunks. More chunks change indexing cost and scoring opportunities. No answer-span check was performed, and no claim is made that 128 tokens is optimal for production documents.

In the repeated Japanese example saved in `tokenizer-lengths.json`, 256 tokens retain 230 characters with the English tokenizer, 370 with Granite multilingual, and 488 with Bekko. These are example-specific, not language-wide conversion rates.

## Real Wasm query execution

PocketIC server 11.0.0, Application subnet, Production instruction limits. Each row below aggregates three exact-length inputs: Japanese, English and mixed. Token counts include special tokens. These are actual successful query calls or actual instruction-limit rejections, not estimates. No mainnet deployment was made.

| Model | Tokens | Successful inputs | Instructions/query range | Min cosine to FP32 | Min cosine to independent W8A8 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Granite 97M Multilingual R2 | 128 | 3/3 | 2,391,705,138–2,392,995,473 | 0.99875230 | 0.99893593 |
| Granite 97M Multilingual R2 | 256 | 3/3 | 4,610,314,929–4,612,600,557 | 0.99874469 | 0.99863725 |
| Granite 97M Multilingual R2 | 304 | 0/3 | exceeded 5,000,000,000 | — | — |
| Bekko a8m | 128 | 3/3 | 811,141,587–811,141,587 | 0.99753183 | 0.99946500 |
| Bekko a8m | 256 | 3/3 | 1,943,095,058–1,943,095,058 | 0.99599229 | 0.99965681 |
| Bekko a8m | 304 | 3/3 | 2,469,046,989–2,469,046,989 | 0.99532029 | 0.99966230 |
| Bekko a8m | 512 | 0/3 | exceeded 5,000,000,000 | — | — |

The independent W8A8 reference is not bit-equivalent to Wasm. Residual cosine differences remain beyond original-FP32 quantization loss; their cause was not isolated. Retrieval scores above are from the Torch simulation, not full-corpus Wasm retrieval. The >0.99 cosine consistency check only rules out gross divergence and does not prove numerical equivalence.

Granite uses the existing optimized GEMM and integer Attention, with multilingual vocabulary/special IDs, SiLU and the model-specific RoPE bases. Bekko uses 4 layers, 6 heads × 64 dimensions, intermediate size 1152, GELU and mean pooling over all tokens after final normalization. Its experiment uses F32 Attention with BF16 K/V and adapted GEMM kernels. The D=32 integer Attention kernel was not reused: both its unrolled width and its INT14 dot-product bound require redesign for D=64. Thus this is a comparison of the two working experimental ports, not equally optimized implementations.

The temporary Bekko adapter uses the existing harness completion sentinel (`next_layer = 12`); the actual encoder loop executes four layers. This adapter is not a proposed production API.

| Model | Weight pack | Allocated Wasm linear memory after activation |
| --- | ---: | ---: |
| Granite 97M Multilingual R2 | 290.9 MiB | 883.8 MiB |
| Bekko a8m | 382.4 MiB | 883.8 MiB |

Large vocabularies dominate the F32 embedding tables. Wasm linear memory includes retained allocator pages and temporary loading allocations; it is not a live-weights-only measurement. The pack sizes above do not imply that the model is fully INT8.

Native inference and compilation overlapped during this experiment. Wall-clock measurements in the raw files are diagnostic observations; they are not controlled latency comparisons. The query instruction measurements are the resource evidence. Three content patterns do not establish a worst-case public API limit.

## Decision and remaining scope

For a local RAG prototype, start with Bekko a8m and compare 128/256-token chunks on actual documents. Keep Granite 97M as an alternative when broader retrieval evaluations favor it. E5 matches Bekko a8m Recall@5 here, but was not ported or instruction-measured. Bekko a25m was also evaluated only in Torch. The current production English canister and its APIs were not modified by this experiment.

Before adopting a default, evaluate independent Japanese and cross-language retrieval sets and answer-containing chunk recall. This subset has fewer distractors than full JaQuAD/MTEB; model training overlap was not audited. There is no claim about final generated-answer quality.

## Evidence and reproduction

- `*-128.json`, `*-256.json`, `*-512.json`: model revisions/configuration, per-query ranks, quantization cosine, token/chunk counts and environment.
- `selection.json`: exact query/document IDs and qrels; `models.json` in the experiment folder pins model revisions.
- `pocket-*.json`: actual Wasm query responses, embeddings, instruction counts and failures.
- `summary.json`: validated metrics and offline comparison of saved Wasm outputs to independent quantized references.
- `*-pack.json`: original weights, pack and reference-fixture hashes.
- `*-materialized.json`: exact generated Rust/Cargo source hashes.
- See [experiment instructions](../../experiments/multilingual/README.md) for commands and limitations.

Primary sources: [JaQuAD retrieval data](https://huggingface.co/datasets/mteb/JaQuADRetrieval), [Granite multilingual](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2), [Bekko a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m), [Bekko a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m), [E5](https://huggingface.co/intfloat/multilingual-e5-small).
