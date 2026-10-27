# Answer-containing chunks: Bekko a8m

Measured locally on 2026-10-27 (JST). Full public JaQuAD validation: 3,939 questions, 1,431 source passages and 101 article titles. All annotated answer offsets matched the original text; no questions were excluded. Every passage is a candidate, without sampled distractors. This is a different and larger evaluation than the earlier 128-query parent-document experiment.

## What counts as success

- Answer hit: a retrieved chunk comes from the annotated source passage and completely contains at least one annotated answer span. The same words in another passage do not count.
- Sentence hit: the chunk also fully contains the sentence/line span enclosing that answer. This is a context-retention proxy, not proof that all reasoning evidence is present.
- Top-5 uses five actual chunks, with no parent-document deduplication. Parent@5 below deduplicates documents as in the earlier experiment, making the difference visible.
- Budget-1024 takes ranked whole chunks until adding the next would exceed 1,024 Bekko tokenizer tokens, including embedding special tokens. It neither truncates that chunk nor skips ahead. Different chunk sizes can leave different unused space. Queries always use the same 256-token maximum across all configurations.

## Quantized-reference results

All percentages are fractions of the 3,939 questions. W8A8 Linear simulation with BF16 K/V, original F32 embedding table and norms. These are local reference-model measurements, not new Wasm measurements.

| Splitting | Max tokens | Vectors | Parent@5 | Answer@1 | Answer@5 | Sentence@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens | 128 | 3930 | 95.30% | 64.76% | 89.54% | 78.52% |
| tokens | 256 | 2060 | 93.55% | 69.99% | 92.13% | 89.52% |
| tokens | 512 | 1459 | 93.55% | 71.79% | 93.50% | 93.42% |
| sentence_end | 128 | 4442 | 95.38% | 65.30% | 88.86% | 83.78% |
| sentence_end | 256 | 2080 | 94.03% | 70.27% | 92.21% | 91.85% |
| sentence_end | 512 | 1459 | 93.55% | 71.72% | 93.53% | 93.53% |

| Splitting | Max tokens | Answer, budget 1024 | Sentence, budget 1024 | Mean tokens used | Mean chunks used |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokens | 128 | 93.37% | 81.92% | 972.9 | 8.65 |
| tokens | 256 | 91.85% | 89.26% | 929.4 | 5.32 |
| tokens | 512 | 90.20% | 90.12% | 888.1 | 4.17 |
| sentence_end | 128 | 92.89% | 87.84% | 971.4 | 9.46 |
| sentence_end | 256 | 92.05% | 91.67% | 927.7 | 5.38 |
| sentence_end | 512 | 90.23% | 90.23% | 887.8 | 4.17 |

## Where the short chunks lose information

| Splitting | Max tokens | Parent retrieved but answer absent from top-5 | Answer retrieved but whole sentence absent | Any candidate contains whole sentence (oracle) |
| --- | ---: | ---: | ---: | ---: |
| tokens | 128 | 4.60% | 11.02% | 88.27% |
| tokens | 256 | 0.91% | 2.61% | 97.39% |
| tokens | 512 | 0.03% | 0.08% | 99.87% |
| sentence_end | 128 | 5.46% | 5.08% | 95.18% |
| sentence_end | 256 | 1.17% | 0.36% | 100.00% |
| sentence_end | 512 | 0.00% | 0.00% | 100.00% |

All six split configurations retain every annotated answer in at least one candidate chunk. Oracle sentence coverage separates splitting limitations from retrieval ranking failures.

## Split rules and scope

`tokens` uses token-offset windows with 32 source-token overlap. `sentence_end` uses the same windows/overlap but prefers the last Japanese sentence-ending punctuation or newline in the latter half of the window. If there is no such boundary, it falls back to the token window. It aligns chunk ends, not necessarily starts; it is not a guarantee of complete sentences. Every emitted chunk is re-tokenized and checked against its budget, and every non-whitespace source character is checked for coverage.

Each source is title + newline + the original paragraph. Titles are not repeated on later chunks. Source paragraphs are relatively short: at 512 tokens most fit in one chunk. Results do not establish an optimum for long technical manuals or whole articles.

JaQuAD includes 829 multiple-sentence-reasoning questions. Merely including an answer string or one enclosing sentence may omit necessary reasoning evidence. No answer-generation model was used and generated-answer correctness was not evaluated. The sentence metric is explicitly a proxy.

## FP32 check

| Splitting | Max tokens | FP32 answer@5 | W8A8 answer@5 | FP32 sentence@5 | W8A8 sentence@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokens | 128 | 89.82% | 89.54% | 78.73% | 78.52% |
| tokens | 256 | 92.03% | 92.13% | 89.41% | 89.52% |
| tokens | 512 | 93.42% | 93.50% | 93.35% | 93.42% |
| sentence_end | 128 | 88.91% | 88.86% | 83.78% | 83.78% |
| sentence_end | 256 | 92.28% | 92.21% | 91.93% | 91.85% |
| sentence_end | 512 | 93.40% | 93.53% | 93.40% | 93.53% |

## Practical choice

For this corpus, start with a 256-token maximum and sentence-end preference. Its quantized-reference answer@5 is 92.21% and full-sentence@5 is 91.85%, with 2,080 vectors. Under the 1,024-token whole-chunk budget it retains an answer in 92.05% and its enclosing sentence in 91.67% of questions. Sentence-end 512 reaches 93.53% at top-5, but falls to 90.23% for both metrics under that budget. Plain 128 windows recover more answers under the fixed budget (93.37%) while retaining the full sentence much less often (81.92%). This supports a practical balance, not a universal optimum or a generated-answer-quality claim.

## Paired uncertainty

10,000 paired bootstrap samples of whole articles (not independent questions). Intervals condition on this fixed validation corpus and do not establish general performance. Differences are percentage points.

| Comparison | Metric | Difference | Article-bootstrap 95% interval |
| --- | --- | ---: | ---: |
| tokens-256 minus tokens-128 | answer@5 | +2.59 | [+1.55, +3.67] |
| tokens-256 minus tokens-128 | sentence@5 | +10.99 | [+9.46, +12.56] |
| tokens-256 minus tokens-128 | answer_budget1024 | -1.52 | [-2.58, -0.47] |
| tokens-256 minus tokens-128 | sentence_budget1024 | +7.34 | [+5.71, +8.96] |
| tokens-512 minus tokens-256 | answer@5 | +1.37 | [+0.71, +2.07] |
| tokens-512 minus tokens-256 | sentence@5 | +3.91 | [+2.97, +4.89] |
| tokens-512 minus tokens-256 | answer_budget1024 | -1.65 | [-2.34, -0.99] |
| tokens-512 minus tokens-256 | sentence_budget1024 | +0.86 | [+0.08, +1.65] |
| sentence_end-256 minus tokens-256 | answer@5 | +0.08 | [-0.38, +0.55] |
| sentence_end-256 minus tokens-256 | sentence@5 | +2.34 | [+1.64, +3.07] |
| sentence_end-256 minus tokens-256 | answer_budget1024 | +0.20 | [-0.29, +0.73] |
| sentence_end-256 minus tokens-256 | sentence_budget1024 | +2.41 | [+1.66, +3.20] |
| sentence_end-256 minus sentence_end-128 | answer@5 | +3.35 | [+2.23, +4.51] |
| sentence_end-256 minus sentence_end-128 | sentence@5 | +8.07 | [+6.48, +9.70] |
| sentence_end-256 minus sentence_end-128 | answer_budget1024 | -0.84 | [-1.86, +0.22] |
| sentence_end-256 minus sentence_end-128 | sentence_budget1024 | +3.83 | [+2.20, +5.45] |
| sentence_end-512 minus sentence_end-256 | answer@5 | +1.32 | [+0.74, +1.93] |
| sentence_end-512 minus sentence_end-256 | sentence@5 | +1.68 | [+1.02, +2.37] |
| sentence_end-512 minus sentence_end-256 | answer_budget1024 | -1.83 | [-2.56, -1.12] |
| sentence_end-512 minus sentence_end-256 | sentence_budget1024 | -1.45 | [-2.20, -0.76] |

## Reproduce and audit

Model: `hotchpotch/bekko-embedding-v1-a8m`, revision `c721113d59a1d91b447450324f51c4b3332c924a`. Dataset: `SkelterLabsInc/JaQuAD`, revision `05600ff310a0970823e70f82f428893b85c71ffe`. Source file hashes, exact question IDs, source spans and context hashes are recorded in `selection.json`. Per-query metrics and top-5 chunk IDs are stored in compressed JSON audit files; six `chunks-*.json` files map those IDs to source offsets. `summary.json` contains all aggregates, question-type breakdowns and intervals.

The implementation checks all aggregate metrics against per-query rows and independently audits 12 sampled query rankings and source-span hits for each of 12 runs from saved vectors (144 audits). Four hand-computed tests cover span boundaries, wrong-source matches, complete sentence bounds and text-preserving chunking.

Use the [Bekko Python setup](../../docs/BEKKO.md#prepare-and-verify), then run from the repository root:

```sh
.cache/bekko-venv/bin/python experiments/multilingual/answer_spans.py --fetch
.cache/bekko-venv/bin/python -m unittest discover -s experiments/multilingual -p test_answer_spans.py
.cache/bekko-venv/bin/python experiments/multilingual/answer_spans_report.py
```

The model must already be cached locally; `--fetch` retrieves only the four pinned public validation JSON files into `.cache/answer-spans`. Dependencies are the existing root `requirements.txt`; PyArrow is also needed by the reused reference module. No canister API or token limit is changed. **512-token rows are model-side reference experiments; the live canister remains capped at 256 tokens.**

Dataset attribution: [Skelter Labs JaQuAD](https://huggingface.co/datasets/SkelterLabsInc/JaQuAD), human-annotated Japanese QA over Wikipedia, CC BY-SA 3.0. The sentence spans and chunk labels are derived here from the supplied answer annotations, not provided as evidence labels by the dataset authors.
