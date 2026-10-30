# Bekko a25m runtime experiments

Follow-up: [the dedicated operational canister](../bekko25-operations/RESULTS.md) now includes all-length validation, resumable checked uploads, upgrade recovery and actual Wasm retrieval. Its deployed artifact's observed maximum is 4,843,751,277 instructions. The prototype measurements below remain historical records.

Measured locally on 2026-10-28. Pinned model: `hotchpotch/bekko-embedding-v1-a25m`, revision `44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c`. All 13 encoder layers, full vocabulary lookup, 384-dimensional mean pooling and L2 normalization execute in the canister. Length includes BOS/EOS. No input truncation, layer removal or independently pooled subchunks replace full inference.

The isolated builder is [bekko25_runtime.py](../../experiments/multilingual/bekko25_runtime.py); [runtime25.rs](../../experiments/embeddinggemma2/src/bin/runtime25.rs) installs it in a disposable PocketIC 11 Application subnet with Production defaults. The actual query limit is **5,000,000,000 instructions**. Failed whole-query calls are retained, including their instruction-limit errors. Neither Benchmarking limits nor update execution are used to claim query success.

## Successful 256-token query

The final `qk13-full-wo` variant succeeds for **256 tokens in one actual query**, in English, Arabic and Korean. It combines unrolled AV, QK13, direct buffers, one internal Attention call over all rows/heads (preserving the original local/global masks), and a second Strassen level only for MLP output projection. K/V conversion and QK quantization happen once per layer instead of once per frame. The second-level output projection preserves the integer matrix product exactly.

| Tokens including BOS/EOS | Actual instructions/query, each of EN/AR/KO | Calls |
| --- | ---: | ---: |
| 64 | 1,052,479,702 | 1 |
| 128 | 2,250,550,772 | 1 |
| 256 | **4,921,718,962** | **1** |

The 256-token margin is **78,281,038 instructions (1.57%)**. This is a narrow measured margin, not proof that arbitrary documents or a larger production API fit. [iteration-8.json](iteration-8.json) contains successful whole-query responses, normalized 384-element vectors, comparison cosines and the measured Wasm hash. No intermediate states cross canister calls in these successful `embed` invocations. The separately profiled English layer sum is 4,916,002,469; use the whole-query cost above for the claim.

All seven available returned vectors from the preceding QK13/direct variant (EN/AR/KO 64/128 plus English 256 through layer profiling) are bit-identical to the final variant. The full/profile English outputs also match exactly. Against the original non-QK13 Wasm runtime, final cosine ranges from 0.99867843 to 0.99934094 across nine fixtures; against the Torch W8A8 reference it ranges from 0.99874729 to 0.99922075. These are numerical checks, not retrieval guarantees. The original runtime already includes W8A8, BF16 KV, approximate GELU and approximate softmax; QK13 introduces an additional change.

**No bit-identical-to-original 256-token success was found.** The exact direct-buffer variant remains above the instruction ceiling. Choosing the successful variant entails the precision tradeoff documented below.

## Optimization measurements

The following costs are the sum of 13 individually profiled layer queries for the English 256-token fixture. They help locate work but exclude input lookup, final pooling and some whole-query overhead. **They do not prove a successful single query.** All these initial variants still fail the whole 256-token query.

| Variant / record | Layer sum | Output relative to baseline on returned fixtures |
| --- | ---: | --- |
| baseline / iteration-0 | 6,078,665,542 | reference |
| AV loop unrolled across 16 SIMD accumulators / iteration-1 | 5,219,078,352 | bit identical |
| also unroll QK / iteration-2 | 5,219,643,718 | bit identical |
| second Strassen level for all projections / iteration-3 | 5,305,185,218 | bit identical |
| QK13 dot products / iteration-4 | 5,041,455,486 | approximate |
| direct internal Attention buffers / iteration-5 | 5,205,992,060 | bit identical |
| QK13 plus direct buffers / iteration-6 | 5,027,789,378 | approximate |

AV shares probability and address calculations across all 64 head dimensions while preserving each accumulator's key order. Direct buffers retain BF16 round-to-nearest-even K/V values but remove their temporary wire encoding/decoding. All nine baseline fixture vectors are available, including layer-by-layer fallback for failed 256-token calls. Exact variants are checked with `f32::to_bits`, not only cosine.

The second Strassen level reduces leaf products but increases preparation/reconstruction cost. Applying it indiscriminately loses performance. Its integer correctness test compares sampled entries with direct i64 products at 128/256 rows, multiple projection shapes, random values and ±127 extremes. At depth two, operands are bounded by 508; a 288-term leaf is bounded by 74,322,432 and a conservative 16-term reconstruction by 1,189,158,912, below i32 maximum.

## Optional QK13 precision change

QK13 is an additional approximation to the existing W8A8/BF16-KV runtime. Each 64-element Q or K head vector is dynamically quantized to integers in ±8191, stored in i16. SIMD i16 dot products accumulate in four i32 lanes; each lane is bounded by `16*8191² = 1,073,479,696`. Their sum uses i64 because the full 64-term dot can exceed i32. V and probability-weighted accumulation remain F32 over BF16-rounded V. This is not bit-equivalent to the original Attention.

The paired reference retrieval check uses 256-token chunks, the same aligned 50 query IDs/domain, two domains and six languages as [the multilingual comparison](../multilingual-global/RESULTS.md). The scores below are equal-domain parent-document Recall@5. This is Torch reference retrieval, **not a full Wasm retrieval evaluation**. CPU integer dots are exact; surrounding MPS F32 softmax/accumulation still differs from Wasm. Fully masked padding rows return zero, matching SDPA, and all saved vectors are finite and normalized.

| Language | W8A8/BF16 KV | plus QK13 | Difference (pp) |
| --- | ---: | ---: | ---: |
| English | 86.73% | 86.73% | 0.00 |
| German | 83.28% | 83.12% | -0.17 |
| Spanish | 82.32% | 80.32% | -2.00 |
| French | 81.43% | 81.14% | -0.29 |
| Arabic | 76.52% | 77.02% | +0.50 |
| Korean | 76.73% | 76.23% | -0.50 |

These are small, translated, bounded candidate sets, not six independent native-language evaluations. Training overlap was not audited. They establish neither unchanged retrieval quality nor a universal language ranking. Do not compare their absolute 256-token scores with the earlier 128-token table as an isolated runtime effect. [Raw quality records](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-runtime/quality/summary.json) retain per-query rankings and cache hashes; [verification.json](https://github.com/humandebri/ic-granite-embedding/blob/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-runtime/verification.json) records independent recomputation of all 24 runs / 1,200 rankings.

## Reproduction and scope

The tracked iteration records retain the baseline and the two comparison fixtures used below. Other regenerable iteration JSON files are ignored locally; the [complete historical records](https://github.com/humandebri/ic-granite-embedding/tree/3aaaea994534105e422cfaba6136fe984a5ad0d8/benchmarks/bekko25-runtime) remain available at the measured commit. Restore or regenerate those records to repeat checks over every historical iteration.

Validation completed: seven native core tests pass, including direct-buffer/wire equality at local/global window boundaries and recursive integer products against i64; nine final PocketIC whole-query cases pass; all nine runtime artifact hashes and all 24 retrieval runs / 1,200 rankings were independently checked from saved artifacts. Python Ruff and Rust formatting checks pass for the new runtime scripts/fragments. Native tests use the scalar Attention path, while the actual PocketIC fixtures exercise Wasm SIMD. [Source hashes](source-hashes.json) distinguish the measured generated source from the formatted reproduction builder/fragments.

First prepare the cached pinned model and original probe using [bekko25_probe.py](../../experiments/multilingual/bekko25_probe.py), as described in the multilingual report. The pack is 418,404,508 bytes, SHA-256 `8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724`. Python dependencies and model/data caches are shared with that experiment.

```sh
python experiments/multilingual/bekko25_runtime.py --variant qk13-full-wo
CARGO_TARGET_DIR=.cache/global-port-target cargo build \
  --manifest-path .cache/bekko25-runtime/port/Cargo.toml \
  --release -p bekko25-probe --target wasm32-unknown-unknown --offline
cp .cache/global-port-target/wasm32-unknown-unknown/release/bekko25_probe.wasm \
  .cache/bekko25-runtime/qk13-full-wo.wasm
CARGO_TARGET_DIR=.cache/runtime-native-target cargo test \
  --manifest-path .cache/bekko25-runtime/port/Cargo.toml --release -p bekko-core --offline
CARGO_TARGET_DIR=.cache/gemma2-target cargo build \
  --manifest-path experiments/embeddinggemma2/Cargo.toml --release --bin runtime25 --offline
POCKET_IC_BIN=/path/to/pocket-ic RUNTIME_ONCE=1 RUNTIME_ITERATION=100 \
  RUNTIME_APPROX=1 RUNTIME_BASELINE=benchmarks/bekko25-runtime/iteration-0.json \
  .cache/gemma2-target/release/runtime25 .cache/bekko25-runtime/qk13-full-wo.wasm
python experiments/multilingual/bekko25_qk_quality.py
python experiments/multilingual/validate_bekko25_runtime.py
```

Build queued variants before invoking the harness; its optional additional Wasm arguments upgrade the same disposable canister and reactivate stable-memory weights without reuploading. `RUNTIME_APPROX=1` records numerical differences instead of asserting bit equality; omit it for exact variants. Quality evaluation requires cached pinned models/datasets and a supported MPS device in this script.

This experiment does not replace the existing a8m service or deploy to mainnet. The probe's controller-only raw stable-memory uploader and explicit activation support disposable benchmarking and upgrades within that harness. It is not the production a8m loader's checksum-verified upload protocol. Public raw-state profiling methods are diagnostic APIs. Production integration, general input coverage and a Wasm retrieval evaluation remain separate work.
