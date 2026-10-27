# Bekko a8m dedicated canister validation

Measured locally on 2026-10-27 (JST). Dedicated fixed-model API; 256-token hard limit including BOS/EOS; normalized 384-dimensional output. The live local service is recorded in [live.json](live.json). No mainnet deployment was performed.

## Query execution

PocketIC server 11.0.0, Application/Production configuration, actual 5B instructions/query ceiling. Each length was tested with Japanese, English and mixed-language fixtures (the special-only length 2 is identical across the three).

| Tokens | Cases | Instructions/query | Minimum cosine to independent W8A8 reference |
| --- | ---: | ---: | ---: |
| 2 | 3 | 14,601,420 | 1.00000000 |
| 32 | 3 | 169,790,796 | 0.99943007 |
| 64 | 3 | 355,441,288 | 0.99949159 |
| 128 | 3 | 811,303,976 | 0.99965894 |
| 256 | 3 | 1,943,374,468 | 0.99976237 |

The managed icp-cli network uses a different server version from the conformance harness. Its 256-token Japanese call measured 1,954,894,606 instructions with 384 finite values and unit norm. These are two actual measurements, not a combined benchmark or a controlled latency comparison.

## Actual Wasm retrieval

JaQuAD smoke set: 32 questions, 160 candidate passages, 252 query/chunk embedding calls. Each document is scored by its best chunk, with document deduplication. All calls used the dedicated canister API after upgrade and activation.

| Implementation | Recall@5 | nDCG@10 |
| --- | ---: | ---: |
| Original FP32 reference | 100.00% | 0.98846655 |
| Independent W8A8 reference | 100.00% | 0.98846655 |
| Actual Wasm API | 100.00% | 0.98846655 |

This is a small integration smoke set with seeded random distractors. The ceiling score is not a general accuracy claim. Numerical vectors are not bit-equivalent to Torch; actual cosine values above and the wider [retrieval comparison](../multilingual-expanded/RESULTS.md) define the evidence.

## Lifecycle checks

- controller-only weight mutation.
- offset and incomplete activation rejected.
- incomplete upload discarded on upgrade.
- wrong weight checksum rejected without advancing progress.
- bad frames, padding, vocabulary and 257-token inputs rejected.
- verified weights survive upgrade; controller activation restores identical embedding.

A real upgrade preserved the verified 400,967,494-byte weight pack. Before activation the model cache was empty; controller activation restored an embedding bit-identical to the pre-upgrade output. Partial uploads were reset rather than treated as verified. Production heap-delta throttling required idle rounds before the post-upload upgrade; no production instruction limits were relaxed.

Allocated Wasm linear memory after testing: 837,877,760 bytes; stable memory: 401,014,784 bytes. Allocated linear memory includes retained allocator pages, not only live tensors.

## Shared-code checks

- Bekko native tests: 5 passed, including the complete dynamic-quantization path, 1152-column accumulation, 64-dimensional Attention, malformed packs and token bounds.
- Granite native regression tests with all features: 18 passed; its production Wasm rebuild also passed.
- Shared generator output matches the previous Granite tree/weighted/score kernels byte for byte. Bekko generated GEMM also matches its validated experimental kernel. Hashes are in [kernel-compatibility.json](kernel-compatibility.json).
- The public Candid declaration matches the Rust export; formatting and Python checks pass.

Raw results: [validation.json](validation.json). Pinned fixture provenance: [fixtures.json](fixtures.json). Reproduction, live calls and repository layout: [Bekko guide](../../docs/BEKKO.md).
