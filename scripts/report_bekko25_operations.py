"""Write the operational release report from independently verified results."""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bekko25_provenance import validate_release_inputs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/bekko25-operations"

# Keep these links tied to the published records. They are context, not gates
# or evidence that regressions were checked for a subsequent build.
HISTORICAL_REVISION = "0501a20f7ed42a6eb94829d56de12b38362dc24b"
HISTORICAL_RECORDS = (
    "https://github.com/humandebri/ic-granite-embedding/blob/"
    f"{HISTORICAL_REVISION}/benchmarks/bekko25-operations"
)


def main():
    target = ROOT / os.environ.get("CARGO_TARGET_DIR", ".cache/bekko25-prod-target")
    report, build, verified = validate_release_inputs(ROOT, OUT, target)
    assert verified["torch_references_checked"]
    live = json.loads((OUT / "live.json").read_text())
    recipe = json.loads((OUT / "recipe-artifact.json").read_text())
    costs = report["cases"][:]
    for name in ["sweep.json", "stress.json"]:
        costs += json.loads((OUT / name).read_text())
    costs += json.loads((OUT / "live-sweep.json").read_text())["results"]
    perf = []
    for n in [2, 31, 63, 64, 65, 128, 192, 255, 256]:
        perf.append(
            f"| {n} | {max(r['instructions'] for r in costs if r['tokens'] == n):,} | 1 |"
        )
    quality = []
    for lang, values in verified["languages"].items():
        quality.append(
            f"| {lang} | {100 * values['reference_recall5']:.2f}% | {100 * values['qk13_reference_recall5']:.2f}% | **{100 * values['wasm_recall5']:.2f}%** | {100 * (values['wasm_recall5'] - values['reference_recall5']):+.2f} |"
        )
    date_jst = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    manifest = {
        "date_jst": date_jst,
        "runtime_id": report["model"]["runtime_id"],
        "model": report["model"],
        "recipe": recipe,
        "rustc": build["rustc"],
        "cargo": build["cargo"],
        "build_record": build,
        "build_record_sha256": report["build_record_sha256"],
        "source_sha256": {
            **build["runtime_source_sha256"],
            **build["validation_source_sha256"],
        },
        "verification_sha256": hashlib.sha256(
            (OUT / "verification.json").read_bytes()
        ).hexdigest(),
    }
    (OUT / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = "\n".join(
        f"- {text}." for text in report["checks"] + report["final_checks"]
    )
    text = f"""# Bekko a25m operational validation

{date_jst} JST (report generation date). The dedicated a25m canister executes all **13 layers, 256 tokens and mean pooling in one query**, with the pinned full vocabulary. [Setup, API and recovery](../../docs/BEKKO25.md). The public API exposes token IDs and embeddings only; no diagnostic raw-state endpoints are deployed.

The release retains the earlier successful QK13 numerical semantics. Sharing K/V loads across four token rows, direct BF16 buffers, complete-tile linear buckets, and removal of redundant recursive output initialization reduce work. **There is no additional precision reduction relative to the previous QK13 success.** Nine EN/AR/KO 64/128/256 golden vectors match that artifact bit-for-bit through the actual local CLI endpoint. QK13 remains approximate relative to F32 Attention; the AV integer experiment was slower and is excluded.

## Query envelope

PocketIC 11, Application subnet / Production defaults, actual 5,000,000,000-instruction ceiling. Maximum counters per length across natural fixtures, fixed-seed sweep, repeated IDs and the deployed-module sweep:

| Tokens including BOS/EOS | Maximum reported instructions | Query calls |
| --- | ---: | ---: |
{chr(10).join(perf)}

The table includes the recipe-processed deployed module. Raw Cargo and deployed instruction counts differ despite matching golden vectors; the deployed artifact is the operational budget.

The overall observed maximum is **{verified["maximum_instructions"]:,}**, leaving about **{verified["margin_percent"]:.2f}%** of the ceiling. The verifier gates this release below 4.9B (at least 2% reserve). Counters span the `embed` body; actual successful query calls establish that the complete message also fits. This is a measured envelope for pinned artifacts and inputs, not a universal bound for future compiler/platform changes.

Both raw and deployed modules pass all **255 lengths, 2–256**. The raw-module harness also checks **102 repeated-token boundary inputs** and **78 natural exact-length inputs in six languages**. The worst length is not assumed to be 256. Linear padding is internal to each operator and is removed before Attention/pooling; PAD IDs are never inserted into model inputs.

The recursive Strassen path is restricted to complete 128-row multiples. In particular, a 192-row bucket uses one level, preventing a partial 32-row leaf. Leaf shape assertions are retained before unsafe SIMD. Native padded/unpadded products and recursive products are checked against exact integer references; integer accumulation widens as documented in the core. Nine focused native tests pass.

[Raw length sweep](sweep.json), [stress inputs](stress.json), [deployed-module sweep](live-sweep.json), [full validation](validation.json), [independent verification](verification.json).

## Retrieval from actual Wasm query outputs

Parent-document Recall@5, equal mean of NanoSciFact and NanoFiQA2018, **50 shared query IDs/domain/language**. Each candidate pool contains all positives plus 128 seeded distractors; source documents use 256-token chunks and 32-token overlap. A parent's score is its best chunk score. Every table row uses the same IDs and relevance labels. The {verified["retrieval_embeddings"]:,} query/chunk embeddings were obtained by calling the actual raw production Wasm query API, not by substituting Torch vectors.

| Language | W8A8/BF16 KV Torch | plus QK13 Torch | Actual Wasm | Wasm minus first reference (pp) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(quality)}

All 600 actual Wasm query rankings, vector hashes, dimensions, finite values and L2 norms are independently checked from saved vectors. The reference-cache hashes are checked against the earlier pinned quality records. Rust ranking uses F64 cosine; the independent scorer normalizes to F64 and reproduces the ranks. Per-task inputs, metrics and ranks are retained in the task JSON files and `*-verified.json` files.

The public data is translated from English, with small candidate pools and only 100 independent source query IDs; translations are not independent samples. Training overlap was not audited. These are parent-document retrieval scores, not generated-answer or answer-span accuracy. They do not establish universal language performance. The older 128-token comparison has different chunks and must not be used as an isolated before/after runtime comparison. Torch/Wasm differences include existing approximate GELU/softmax and quantization rounding, not only QK13.

## Loader, authority and upgrades

{checks}

The loader verifies model/revision/embedding-space identity and the complete local pack hash, resumes from durable progress and retries identical chunks. `beginUpload` is an explicit restart that deactivates inference. Stable magic `BEK25001` belongs to this fixed model revision, separate from a8m. After upgrade, weights remain verified but controller activation is required to reconstruct the heap cache. No whole-heap serialization is used.

`embedding_space` is `{report["model"]["embedding_space"]}`. Store it with the index and reject mismatched query vectors. A semantic model/precision change requires re-embedding; bit-equivalent runtime changes do not. Both a8m and a25m compile in the same workspace without model-selecting Cargo features changing the a8m alias.

## Artifacts and existing-service regression

- Raw Cargo Wasm SHA256: `{verified["wasm_sha256"]}`.
- Deployed recipe artifact SHA256: `{verified["deployed_sha256"]}`.
- Pinned pack: 418,404,508 bytes, SHA256 `{report["model"]["weight_pack_sha256"]}`.
- Local canister: `{live["status"]["id"]}`, API `{live["network"]["api_url"]}`.
- Measured raw-module heap: {report["wasm_memory_bytes"]} bytes; stable memory: {report["stable_memory_bytes"]} bytes. These are not mainnet capacity/latency estimates.

The `@dfinity/rust@v3.2.0` recipe re-encodes the Wasm while adding cargo/template/Candid metadata. [The recipe record](recipe-artifact.json) reproduces the deployed hash exactly. Raw-module retrieval results are identified separately from the deployed module's full length sweep and golden comparisons. [The release manifest](release.json) pins source hashes, compiler and metadata-tool versions.

### Historical shared-runtime regression records

The **2026-10-29 release records**, preserved in repository snapshot `{HISTORICAL_REVISION}`, recorded 15 a8m fixtures and lifecycle/retrieval smoke for a8m Wasm `6d905c9bb6f68cabca0d7814caf0fd439efcd385fe22e74fa76d19509fef7987` ([historical a8m record]({HISTORICAL_RECORDS}/a8-regression.json)), and six unchanged Granite generator outputs across 1152/1536 widths ([historical generator comparison]({HISTORICAL_RECORDS}/generator-regression.json)). These checks are **not revalidated by this report generator and do not establish regression success for the current build**. The current release manifest's source fingerprints do not attest these historical checks. Shared-kernel changes require new a8m and Granite regression runs before claiming compatibility for the changed build.

The live comparison checks eight cached a8m vectors against the original running service; it does not execute the newly built a8m Wasm. The historical exploratory a25m materializer pins commit `e281da1a289060101a0bfceb3ecf48403415b8d0`, so production refactors cannot silently turn its 13-layer probe into an a8m model.

This work leaves the dedicated local canister active and includes deployment/recovery tooling. **No mainnet deployment or mainnet load test was performed.** For a release on another platform version, rerun the artifact-specific envelope and quality checks; local single-node tests do not establish multi-node consensus or a throughput SLA.
"""
    (OUT / "RESULTS.md").write_text(text)
    print("Wrote operational report and release source manifest")


if __name__ == "__main__":
    main()
