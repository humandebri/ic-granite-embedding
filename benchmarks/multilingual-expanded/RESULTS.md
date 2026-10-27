# Expanded English and Japanese embedding evaluation

Measured locally on 2026-10-27 (JST). Eleven models, three fixed candidate pools, 256-token chunks, FP32 and W8A8/BF16-KV. These are bounded retrieval experiments, not full BEIR, MTEB, JMTEB or a canister performance comparison.

## Data and method

| Task | Language/domain | Queries | Candidate documents |
| --- | --- | ---: | ---: |
| scifact | English scientific claims / abstracts | 128 | 642 |
| fiqa | English finance questions / answers | 128 | 844 |
| jaquad | Japanese questions / passages | 128 | 635 |

Every pool includes all positive documents for the selected queries plus 512 seeded random distractors (seed 20261027). The English datasets use test qrels. Japanese IDs/qrels match the earlier filtered JaQuAD validation experiment; five earlier Japanese results are reused after checking exact query/document/qrel equality. Each model has the same candidate pool per task. Model training/evaluation overlap was not audited. Random distractors are often easier than a full corpus or mined hard negatives.

Each model uses its own tokenizer, max 256 tokens including prefixes/special tokens, 32-token source overlap, and maximum chunk cosine per document with document deduplication. This measures parent-document retrieval, not whether the returned chunk contains an answer. E5 gets query/passage prefixes. BGE gets its recommended retrieval instruction on queries only. Horizon includes its learned 384→1024 projection; it is not evaluated as the raw backbone. Sentence Transformers checks verify all eleven FP32 pooling/projection adapters on four short English/Japanese texts, with max absolute difference <1e-5.

W8A8 simulates per-channel INT8 Linear weights and per-token INT8 activations, plus BF16 K/V; embeddings/norms remain F32. Horizon’s output Linear projection is also quantized. Float simulations are not integer-speed measurements. Vectors are saved locally and every newly measured result is recomputed from saved vectors to validate ranking and aggregation.

## Retrieval quality

nDCG@10 rewards placing relevant documents near the top. Values below are ×100. All three columns use the same metric.

| Model | English SciFact FP32 / W8A8 | English FiQA FP32 / W8A8 | Japanese JaQuAD FP32 / W8A8 |
| --- | ---: | ---: | ---: |
| Granite Small English R2 | 84.18 / 83.99 | 77.97 / 77.67 | 55.34 / 53.95 |
| Granite 97M R2 multilingual | 79.26 / 78.66 | 72.12 / 71.34 | 81.53 / 80.54 |
| Bekko a8m | 81.10 / 81.08 | 70.24 / 69.95 | 89.63 / 89.46 |
| Bekko a25m | 79.71 / 79.46 | 74.99 / 74.73 | 86.21 / 86.45 |
| multilingual-e5-small | 79.44 / 79.89 | 70.81 / 70.86 | 88.82 / 88.92 |
| bge-small-en-v1.5 | 81.35 / 81.50 | 75.26 / 75.24 | 45.66 / 46.29 |
| e5-small-v2 | 77.05 / 76.92 | 72.20 / 71.24 | 44.95 / 45.62 |
| all-MiniLM-L6-v2 | 78.74 / 78.28 | 76.89 / 76.91 | 17.40 / 17.08 |
| Granite 107M multilingual | 77.25 / 77.52 | 68.24 / 68.12 | 82.57 / 82.53 |
| Horizon multilingual-small | 71.08 / 70.17 | 68.64 / 68.74 | 84.32 / 84.29 |
| Granite 311M R2 multilingual | 82.55 / 82.63 | 76.39 / 75.60 | 83.92 / 83.53 |

Recall@5, W8A8/BF16-KV: mean fraction of the query’s known relevant documents found in the first five deduplicated documents. A query can have multiple relevant documents, especially in FiQA.

| Model | English SciFact | English FiQA | Japanese JaQuAD |
| --- | ---: | ---: | ---: |
| Granite Small English R2 | 89.06% | 75.84% | 62.50% |
| Granite 97M R2 multilingual | 83.59% | 69.98% | 89.84% |
| Bekko a8m | 86.33% | 69.52% | 96.09% |
| Bekko a25m | 84.38% | 72.56% | 93.75% |
| multilingual-e5-small | 85.74% | 69.73% | 96.09% |
| bge-small-en-v1.5 | 86.33% | 72.97% | 53.91% |
| e5-small-v2 | 79.30% | 69.87% | 54.69% |
| all-MiniLM-L6-v2 | 83.59% | 72.47% | 21.88% |
| Granite 107M multilingual | 84.38% | 66.20% | 87.50% |
| Horizon multilingual-small | 76.37% | 67.82% | 92.97% |
| Granite 311M R2 multilingual | 88.28% | 73.94% | 91.41% |

ASCII BM25 baseline over whole original documents (k1=1.5, b=0.75, deduplicated query terms; not a tuned baseline):

| Task | nDCG@10 | Recall@5 |
| --- | ---: | ---: |
| scifact | 82.01 | 85.35% |
| fiqa | 53.66 | 51.55% |

BGE-small-en, E5-small-v2, MiniLM-L6-v2 and Granite Small English R2 are English-oriented baselines. Their Japanese results are diagnostic, not claims of official Japanese support. No aggregate score across these tasks is used to declare a universal winner.

## Why Bekko a8m is efficient

The name counts about 7.67M non-embedding parameters, not all weights: total size is about 106M. The model retains mmBERT-small layers [0,1,2,18] from a pretrained 22-layer encoder, then receives approximately 1.1 billion multilingual contrastive pairs and hard-negative fine-tuning. This preserves a large multilingual vocabulary and selected pretrained layers while cutting repeated transformer compute. The embedding training stages do not use teacher distillation. This explains a plausible route to strong quality per unit of compute; it does not prove the cause of a particular benchmark result. [Bekko paper](https://arxiv.org/html/2607.25180v1)

The earlier 96.1% Japanese Recall@5 was a score on 128 queries against 635 candidates, not a universal accuracy rate. a8m versus a25m differed by only a net three queries at top-5 (four wins, one loss). The paired query-bootstrap interval below includes zero for that comparison. The developer’s broader MMTEB retrieval comparison instead reports 56.2 for a8m, 57.5 for a25m and 60.3 for Granite 97M R2. These are different tasks/metrics and are not directly comparable with our Recall percentages. [Published comparison](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m)

## Paired comparisons

Difference in W8A8 Recall@5, a8m minus the comparator, in percentage points. Intervals resample the same 128 queries 10,000 times. They quantify query sampling within this fixed pool only; they do not cover new corpora, annotation gaps, model training overlap or benchmark selection.

| Task | Comparator | Difference | Query bootstrap 95% interval |
| --- | --- | ---: | ---: |
| scifact | Granite Small English R2 | -2.73 | [-6.25, +0.39] |
| scifact | Granite 97M R2 multilingual | +2.73 | [-1.56, +7.42] |
| scifact | Bekko a25m | +1.95 | [-2.34, +6.25] |
| scifact | multilingual-e5-small | +0.59 | [-2.54, +3.91] |
| scifact | Granite 311M R2 multilingual | -1.95 | [-6.25, +1.95] |
| fiqa | Granite Small English R2 | -6.32 | [-11.78, -1.10] |
| fiqa | Granite 97M R2 multilingual | -0.46 | [-5.26, +4.55] |
| fiqa | Bekko a25m | -3.04 | [-8.04, +1.76] |
| fiqa | multilingual-e5-small | -0.21 | [-5.79, +5.31] |
| fiqa | Granite 311M R2 multilingual | -4.42 | [-9.18, +0.02] |
| jaquad | Granite Small English R2 | +33.59 | [+25.00, +42.19] |
| jaquad | Granite 97M R2 multilingual | +6.25 | [+1.56, +11.72] |
| jaquad | Bekko a25m | +2.34 | [-0.78, +6.25] |
| jaquad | multilingual-e5-small | +0.00 | [-2.34, +2.34] |
| jaquad | Granite 311M R2 multilingual | +4.69 | [+0.78, +9.38] |

## Model size and execution scope

| Model | Total parameters M | Non-embedding parameters M | Output dimensions |
| --- | ---: | ---: | ---: |
| Granite Small English R2 | 47.66 | 28.32 | 384 |
| Granite 97M R2 multilingual | 97.44 | 28.32 | 384 |
| Bekko a8m | 105.98 | 7.67 | 384 |
| Bekko a25m | 123.23 | 24.93 | 384 |
| multilingual-e5-small | 117.65 | 21.64 | 384 |
| bge-small-en-v1.5 | 33.36 | 21.64 | 384 |
| e5-small-v2 | 33.36 | 21.64 | 384 |
| all-MiniLM-L6-v2 | 22.71 | 10.99 | 384 |
| Granite 107M multilingual | 106.99 | 10.99 | 384 |
| Horizon multilingual-small | 140.89 | 42.58 | 1024 |
| Granite 311M R2 multilingual | 311.66 | 110.33 | 768 |

Non-embedding parameters are a rough compute indicator, not a query-instruction measurement. Large vocabularies consume memory even though lookup costs less than dense transformer work. Output dimensionality also changes index storage and similarity costs.

This expansion adds retrieval experiments, not six new canister ports. The earlier local Wasm measurements remain: Bekko a8m about 1.94B instructions at 256 tokens, Granite 97M about 4.61B, both one query on three test inputs. There are residual numerical differences between those Wasm ports and the Torch simulation; no full-corpus Wasm retrieval equivalence is claimed. [Earlier Wasm results](../multilingual/RESULTS.md)

EmbeddingGemma-300M was considered but its weight/config access was gated for the available Hugging Face session, so it was not measured. No score was imputed.

## Evidence and reproduction

- `summary.json`: validated metrics, parameter counts and paired comparisons.
- `<task>/selection.json`: exact source revisions, query/document IDs and qrels.
- `<task>/<model>-256.json`: model/config, per-query ranks, both arithmetic modes and truncation diagnostics.
- `adapters.json`: comparison with Sentence Transformers for all eleven models.
- Local `.cache/multilingual-expanded/`: vectors and parent indices for recomputation.
- [Commands and limitations](../../experiments/multilingual/README.md#expanded-english-and-japanese-comparison).

Sources: [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact), [SciFact test qrels](https://huggingface.co/datasets/BeIR/scifact-qrels), [BEIR FiQA](https://huggingface.co/datasets/BeIR/fiqa), [FiQA test qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), [JaQuAD](https://huggingface.co/datasets/mteb/JaQuADRetrieval). Model revisions are pinned in [expanded-models.json](../../experiments/multilingual/expanded-models.json).
