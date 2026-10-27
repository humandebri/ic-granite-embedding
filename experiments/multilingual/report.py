"""Validate saved measurements and summarize the bounded experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/multilingual"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    selection_hash = hashlib.sha256((OUT / "selection.json").read_bytes()).hexdigest()
    summary = {
        "queries": len(selection["queries"]),
        "documents": len(selection["documents"]),
        "retrieval": {},
        "ports": {},
    }
    for p in sorted(OUT.glob("*-*.json")):
        stem = p.stem
        if not stem.rsplit("-", 1)[-1].isdigit():
            continue
        x = json.loads(p.read_text())
        assert x["selection_sha256"] == selection_hash
        assert x["query_count"] == 128 and x["corpus_count"] == 635
        for r in x["runs"].values():
            assert r["encoding"]["truncated"] == 0
            assert len(r["ranks"]) == 128
        summary["retrieval"][stem] = {
            "chunks": x["chunking"]["chunks"],
            "fp32": x["runs"]["fp32"]["metrics"],
            "w8a8": x["runs"]["w8a8-bf16kv"]["metrics"],
            "quantization_cosine": x["runs"]["w8a8-bf16kv"]["cosine_to_fp32"],
            "crosslingual_smoke_correct": round(
                x["runs"]["w8a8-bf16kv"]["crosslingual_handwritten_hit@1"] * 12
            ),
        }
    assert len(summary["retrieval"]) == 9
    for key in ["granite", "bekko8"]:
        x = json.loads((OUT / f"pocket-{key}.json").read_text())
        pack = json.loads((OUT / f"{key}-pack.json").read_text())
        assert x["pack_sha256"] == pack["sha256"]
        cases = {
            c["case"]: c
            for c in json.loads(
                (ROOT / f".cache/multilingual/{key}/cases.json").read_text()
            )
        }
        measured = []
        for c in x["cases"]:
            r = {"case": c["case"], "tokens": c["tokens"]}
            if "error" in c:
                assert (
                    (key == "granite" and c["tokens"] == 304)
                    or (key == "bekko8" and c["tokens"] == 512)
                ) and "CanisterInstructionLimitExceeded" in c["error"]
                r["result"] = "instruction_limit_exceeded"
            else:
                assert c["instructions"] < 5_000_000_000
                assert len(c["embedding"]) == 384
                v = np.array(c["embedding"])
                assert np.isfinite(v).all() and abs(np.linalg.norm(v) - 1) < 1e-5
                fp = cosine(v, cases[c["case"]]["fp32"])
                quant = cosine(v, cases[c["case"]]["w8a8"])
                assert abs(fp - c["cosine_to_fp32"]) < 1e-10
                assert quant > 0.99, (key, c["case"], quant)
                # Coarse consistency check only; residual drift is explicitly reported.
                r.update(
                    result="ok",
                    instructions=c["instructions"],
                    cosine_to_fp32=fp,
                    cosine_to_w8a8=quant,
                )
            measured.append(r)
        summary["ports"][key] = {
            "pack_bytes": pack["bytes"],
            "wasm_memory_bytes": int(x["wasm_memory_bytes"].replace("_", "")),
            "wasm_sha256": x["wasm_sha256"],
            "cases": measured,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Multilingual embeddings for query-based RAG",
        "",
        "Measured locally on 2026-10-27 (JST). **Bekko a8m is the leading candidate in this bounded experiment.** These results do not establish a general multilingual ranking.",
        "",
        "## Japanese passage retrieval",
        "",
        "Pinned JaQuAD validation subset: 128 queries, 635 passages (all relevant passages for these queries plus 512 seeded distractors). Filter: passage text has at least 80 Unicode characters, removing most answer-only items. Seed: 20261027. Titles are prepended. All models use their own tokenizer; all chunks fit the stated token budget, including special tokens and E5 prefixes. Chunks overlap by 32 source tokens. Document score is the maximum chunk cosine; top-k ranks deduplicate documents. This is document retrieval, not answer-containing chunk recall or end-to-end RAG accuracy.",
        "",
        "FP32 uses original public weights converted to F32. W8A8 uses per-output-channel INT8 weights, dynamic per-token INT8 activations, and BF16-rounded K/V. Embedding tables and normalization stay F32. Torch simulates this arithmetic; it is not an INT8 speed benchmark.",
        "",
        "| Model, 256-token chunks | FP32 Recall@5 | W8A8 Recall@5 | W8A8 Hit@1 | W8A8 nDCG@10 | Chunks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "english": "Granite Small English R2",
        "granite": "Granite 97M Multilingual R2",
        "bekko8": "Bekko a8m",
        "bekko25": "Bekko a25m",
        "e5": "multilingual-e5-small",
    }
    for key, name in names.items():
        x = summary["retrieval"][key + "-256"]
        r = x["w8a8"]
        lines.append(
            f"| {name} | {x['fp32']['recall@5'] * 100:.2f}% | {r['recall@5'] * 100:.2f}% | {r['hit@1'] * 100:.2f}% | {r['ndcg@10']:.4f} | {x['chunks']} |"
        )
    lines += [
        "",
        "The Japanese-query → English-document handwritten smoke test has only 12 easy cases. Correct top-1: "
        + ", ".join(
            f"{names[k]} {summary['retrieval'][k + '-256']['crosslingual_smoke_correct']}/12"
            for k in names
        )
        + ". It is a functionality check, not a cross-lingual benchmark.",
        "",
        "## Chunk length",
        "",
        "| Model | Token budget | W8A8 Recall@5 | W8A8 Hit@1 | Chunks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ["granite", "bekko8"]:
        for n in [128, 256, 512]:
            x = summary["retrieval"][f"{key}-{n}"]
            r = x["w8a8"]
            lines.append(
                f"| {names[key]} | {n} | {r['recall@5'] * 100:.2f}% | {r['hit@1'] * 100:.2f}% | {x['chunks']} |"
            )
    lines += [
        "",
        "128-token chunks perform best on this subset, but create roughly twice as many vectors as 256-token chunks. More chunks change indexing cost and scoring opportunities. No answer-span check was performed, and no claim is made that 128 tokens is optimal for production documents.",
        "",
        "In the repeated Japanese example saved in `tokenizer-lengths.json`, 256 tokens retain 230 characters with the English tokenizer, 370 with Granite multilingual, and 488 with Bekko. These are example-specific, not language-wide conversion rates.",
        "",
        "## Real Wasm query execution",
        "",
        "PocketIC server 11.0.0, Application subnet, Production instruction limits. Each row below aggregates three exact-length inputs: Japanese, English and mixed. Token counts include special tokens. These are actual successful query calls or actual instruction-limit rejections, not estimates. No mainnet deployment was made.",
        "",
        "| Model | Tokens | Successful inputs | Instructions/query range | Min cosine to FP32 | Min cosine to independent W8A8 reference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, x in summary["ports"].items():
        lengths = sorted({c["tokens"] for c in x["cases"]})
        for n in lengths:
            cs = [c for c in x["cases"] if c["tokens"] == n]
            good = [c for c in cs if c["result"] == "ok"]
            if good:
                instr = f"{min(c['instructions'] for c in good):,}–{max(c['instructions'] for c in good):,}"
                fp = f"{min(c['cosine_to_fp32'] for c in good):.8f}"
                q = f"{min(c['cosine_to_w8a8'] for c in good):.8f}"
            else:
                instr = "exceeded 5,000,000,000"
                fp = q = "—"
            lines.append(
                f"| {names[key]} | {n} | {len(good)}/{len(cs)} | {instr} | {fp} | {q} |"
            )
    lines += [
        "",
        "The independent W8A8 reference is not bit-equivalent to Wasm. Residual cosine differences remain beyond original-FP32 quantization loss; their cause was not isolated. Retrieval scores above are from the Torch simulation, not full-corpus Wasm retrieval. The >0.99 cosine consistency check only rules out gross divergence and does not prove numerical equivalence.\n\nGranite uses the existing optimized GEMM and integer Attention, with multilingual vocabulary/special IDs, SiLU and the model-specific RoPE bases. Bekko uses 4 layers, 6 heads × 64 dimensions, intermediate size 1152, GELU and mean pooling over all tokens after final normalization. Its experiment uses F32 Attention with BF16 K/V and adapted GEMM kernels. The D=32 integer Attention kernel was not reused: both its unrolled width and its INT14 dot-product bound require redesign for D=64. Thus this is a comparison of the two working experimental ports, not equally optimized implementations.",
        "",
        "The temporary Bekko adapter uses the existing harness completion sentinel (`next_layer = 12`); the actual encoder loop executes four layers. This adapter is not a proposed production API.",
        "",
        "| Model | Weight pack | Allocated Wasm linear memory after activation |",
        "| --- | ---: | ---: |",
    ]
    for key, x in summary["ports"].items():
        lines.append(
            f"| {names[key]} | {x['pack_bytes'] / 2**20:.1f} MiB | {x['wasm_memory_bytes'] / 2**20:.1f} MiB |"
        )
    lines += [
        "",
        "Large vocabularies dominate the F32 embedding tables. Wasm linear memory includes retained allocator pages and temporary loading allocations; it is not a live-weights-only measurement. The pack sizes above do not imply that the model is fully INT8.",
        "",
        "Native inference and compilation overlapped during this experiment. Wall-clock measurements in the raw files are diagnostic observations; they are not controlled latency comparisons. The query instruction measurements are the resource evidence. Three content patterns do not establish a worst-case public API limit.",
        "",
        "## Decision and remaining scope",
        "",
        "For a local RAG prototype, start with Bekko a8m and compare 128/256-token chunks on actual documents. Keep Granite 97M as an alternative when broader retrieval evaluations favor it. E5 matches Bekko a8m Recall@5 here, but was not ported or instruction-measured. Bekko a25m was also evaluated only in Torch. The current production English canister and its APIs were not modified by this experiment.",
        "",
        "Before adopting a default, evaluate independent Japanese and cross-language retrieval sets and answer-containing chunk recall. This subset has fewer distractors than full JaQuAD/MTEB; model training overlap was not audited. There is no claim about final generated-answer quality.",
        "",
        "## Evidence and reproduction",
        "",
        "- `*-128.json`, `*-256.json`, `*-512.json`: model revisions/configuration, per-query ranks, quantization cosine, token/chunk counts and environment.",
        "- `selection.json`: exact query/document IDs and qrels; `models.json` in the experiment folder pins model revisions.",
        "- `pocket-*.json`: actual Wasm query responses, embeddings, instruction counts and failures.",
        "- `summary.json`: validated metrics and offline comparison of saved Wasm outputs to independent quantized references.",
        "- `*-pack.json`: original weights, pack and reference-fixture hashes.",
        "- `*-materialized.json`: exact generated Rust/Cargo source hashes.",
        "- See [experiment instructions](../../experiments/multilingual/README.md) for commands and limitations.",
        "",
        "Primary sources: [JaQuAD retrieval data](https://huggingface.co/datasets/mteb/JaQuADRetrieval), [Granite multilingual](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2), [Bekko a8m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m), [Bekko a25m](https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m), [E5](https://huggingface.co/intfloat/multilingual-e5-small).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("Validated 9 retrieval experiments and both Wasm ports.")


if __name__ == "__main__":
    main()
