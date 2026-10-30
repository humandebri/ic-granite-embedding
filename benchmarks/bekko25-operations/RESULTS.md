# Bekko a25m operational validation

2026-10-29 JST. The dedicated a25m canister executes all **13 layers, 256 tokens and mean pooling in one query**, with the pinned full vocabulary. [Setup, API and recovery](../../docs/BEKKO25.md). The public API exposes token IDs and embeddings only; no diagnostic raw-state endpoints are deployed.

The release retains the earlier successful QK13 numerical semantics. Sharing K/V loads across four token rows, direct BF16 buffers, complete-tile linear buckets, and removal of redundant recursive output initialization reduce work. **There is no additional precision reduction relative to the previous QK13 success.** Nine EN/AR/KO 64/128/256 golden vectors match that artifact bit-for-bit through the actual local CLI endpoint. QK13 remains approximate relative to F32 Attention; the AV integer experiment was slower and is excluded.

## Query envelope

PocketIC 11, Application subnet / Production defaults, actual 5,000,000,000-instruction ceiling. Maximum counters per length across natural fixtures, fixed-seed sweep, repeated IDs and the deployed-module sweep:

| Tokens including BOS/EOS | Maximum reported instructions | Query calls |
| --- | ---: | ---: |
| 2 | 63,294,099 | 1 |
| 31 | 566,134,232 | 1 |
| 63 | 1,138,077,014 | 1 |
| 64 | 1,064,755,270 | 1 |
| 65 | 1,866,585,596 | 1 |
| 128 | 2,236,305,927 | 1 |
| 192 | 3,518,369,081 | 1 |
| 255 | 4,843,751,277 | 1 |
| 256 | 4,833,400,086 | 1 |

The table includes the recipe-processed deployed module. Raw Cargo and deployed instruction counts differ despite matching golden vectors; the deployed artifact is the operational budget.

The overall observed maximum is **4,843,751,277**, leaving about **3.12%** of the ceiling. The verifier gates this release below 4.9B (at least 2% reserve). Counters span the `embed` body; actual successful query calls establish that the complete message also fits. This is a measured envelope for pinned artifacts and inputs, not a universal bound for future compiler/platform changes.

Both raw and deployed modules pass all **255 lengths, 2–256**. The raw-module harness also checks **102 repeated-token boundary inputs** and **78 natural exact-length inputs in six languages**. The worst length is not assumed to be 256. Linear padding is internal to each operator and is removed before Attention/pooling; PAD IDs are never inserted into model inputs.

The recursive Strassen path is restricted to complete 128-row multiples. In particular, a 192-row bucket uses one level, preventing a partial 32-row leaf. Leaf shape assertions are retained before unsafe SIMD. Native padded/unpadded products and recursive products are checked against exact integer references; integer accumulation widens as documented in the core. Nine focused native tests pass.

[Raw length sweep](sweep.json), [stress inputs](stress.json), [deployed-module sweep](live-sweep.json), [full validation](validation.json), [independent verification](verification.json).

## Retrieval from actual Wasm query outputs

Parent-document Recall@5, equal mean of NanoSciFact and NanoFiQA2018, **50 shared query IDs/domain/language**. Each candidate pool contains all positives plus 128 seeded distractors; source documents use 256-token chunks and 32-token overlap. A parent's score is its best chunk score. Every table row uses the same IDs and relevance labels. The 5,691 query/chunk embeddings were obtained by calling the actual raw production Wasm query API, not by substituting Torch vectors.

| Language | W8A8/BF16 KV Torch | plus QK13 Torch | Actual Wasm | Wasm minus first reference (pp) |
| --- | ---: | ---: | ---: | ---: |
| en | 86.73% | 86.73% | **86.73%** | +0.00 |
| de | 83.28% | 83.12% | **83.95%** | +0.67 |
| es | 82.32% | 80.32% | **81.32%** | -1.00 |
| fr | 81.43% | 81.14% | **81.80%** | +0.38 |
| ar | 76.52% | 77.02% | **77.02%** | +0.50 |
| ko | 76.73% | 76.23% | **75.73%** | -1.00 |

All 600 actual Wasm query rankings, vector hashes, dimensions, finite values and L2 norms are independently checked from saved vectors. The reference-cache hashes are checked against the earlier pinned quality records. Rust ranking uses F64 cosine; the independent scorer normalizes to F64 and reproduces the ranks. Per-task inputs, metrics and ranks are retained in the task JSON files and `*-verified.json` files.

The public data is translated from English, with small candidate pools and only 100 independent source query IDs; translations are not independent samples. Training overlap was not audited. These are parent-document retrieval scores, not generated-answer or answer-span accuracy. They do not establish universal language performance. The older 128-token comparison has different chunks and must not be used as an isolated before/after runtime comparison. Torch/Wasm differences include existing approximate GELU/softmax and quantization rounding, not only QK13.

## Loader, authority and upgrades

- controller-only weight mutation.
- offset and incomplete activation rejected.
- incomplete upload discarded on upgrade.
- wrong weight checksum rejected without advancing progress.
- first/final chunk retries are idempotent; changed retry bytes rejected.
- bad frames, padding, vocabulary and 257-token inputs rejected.
- verified weights survive upgrade; controller activation restores identical embedding.
- all 255 lengths plus 102 repeated-token boundary cases.
- restart deactivates cache and resets digest/offset.
- wrong-model upgrade refuses storage reinterpretation; old loader still works after downgrade.

The loader verifies model/revision/embedding-space identity and the complete local pack hash, resumes from durable progress and retries identical chunks. `beginUpload` is an explicit restart that deactivates inference. Stable magic `BEK25001` belongs to this fixed model revision, separate from a8m. After upgrade, weights remain verified but controller activation is required to reconstruct the heap cache. No whole-heap serialization is used.

`embedding_space` is `bekko-a25m-44f0b8af-w8a8-bf16kv-qk13-v1`. Store it with the index and reject mismatched query vectors. A semantic model/precision change requires re-embedding; bit-equivalent runtime changes do not. Both a8m and a25m compile in the same workspace without model-selecting Cargo features changing the a8m alias.

## Artifacts and existing-service regression

- Raw Cargo Wasm SHA256: `22e396d7536d0274d15872bbfc1766849588448d678f43412bfaf9a06a1a106f`.
- Deployed recipe artifact SHA256: `c7df05a8ecb2f6d114293ed663415fd2cdb23028803e7311b957cd57c9b194de`.
- Pinned pack: 418,404,508 bytes, SHA256 `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`.
- Local canister: `4fbx2-kt777-77775-aaabq-cai`, API `http://localhost:56619/`.
- Measured raw-module heap: 985_595_904 bytes; stable memory: 418_447_360 bytes. These are not mainnet capacity/latency estimates.

The `@dfinity/rust@v3.2.0` recipe re-encodes the Wasm while adding cargo/template/Candid metadata. [The recipe record](recipe-artifact.json) reproduces the deployed hash exactly. Raw-module retrieval results are identified separately from the deployed module's full length sweep and golden comparisons. [The release manifest](release.json) pins source hashes, compiler and metadata-tool versions.

The original a8m service was left unchanged. Its 15 fixtures and lifecycle/retrieval smoke pass with the shared-code build ([regression](a8-regression.json)); eight vectors are bit-identical to the original running a8m service ([live comparison](live.json)). The ordinary Granite generator's six original output files across 1152/1536 widths match the previous source exactly ([generator comparison](generator-regression.json)). The historical exploratory a25m materializer now pins commit `e281da1a289060101a0bfceb3ecf48403415b8d0`, so production refactors cannot silently turn its 13-layer probe into an a8m model.

This work leaves the dedicated local canister active and includes deployment/recovery tooling. **No mainnet deployment or mainnet load test was performed.** For a release on another platform version, rerun the artifact-specific envelope and quality checks; local single-node tests do not establish multi-node consensus or a throughput SLA.
