"""Paired 256-token retrieval check for optional QK13 attention arithmetic."""

import argparse
import hashlib
import json

import numpy as np
import torch
from evaluate import W8A8, metrics
from expanded import chunks
from global_eval import LANGUAGES, ROOT, TASKS, dataset, encode, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integer-av", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = ROOT / "benchmarks/bekko25-runtime/quality"
    if args.integer_av:
        out = ROOT / "benchmarks/bekko25-operations/reference-quality"
    cache = ROOT / ".cache/bekko25-runtime/quality"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model("bekko25")
    model.to("mps")
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, W8A8(module))
    original = torch.nn.functional.scaled_dot_product_attention

    def bf16(q, k, v, *a, **kw):
        return original(q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw)

    def q13(q, k, v, *a, **kw):
        k = k.bfloat16().float()
        v = v.bfloat16().float()
        qs = (q.abs().amax(-1, keepdim=True) / 8191).clamp_min(1e-12)
        ks = (k.abs().amax(-1, keepdim=True) / 8191).clamp_min(1e-12)
        qi = (q / qs).round().clamp(-8191, 8191).to(device="cpu", dtype=torch.int32)
        ki = (k / ks).round().clamp(-8191, 8191).to(device="cpu", dtype=torch.int32)
        # A half-dot has at most 32*8191^2=2,146,959,392: exact i32.
        # Widen before adding both halves, matching Wasm's exact i64 dot.
        d0 = qi[..., :32] @ ki[..., :32].transpose(-1, -2)
        d1 = qi[..., 32:] @ ki[..., 32:].transpose(-1, -2)
        dot = (d0.long() + d1.long()).float().to(q.device)
        scores = dot * (qs * ks.transpose(-1, -2)) * (1 / 8)
        mask = a[0] if a else kw.get("attn_mask")
        if mask is not None:
            scores = (
                scores.masked_fill(~mask, float("-inf"))
                if mask.dtype == torch.bool
                else scores + mask
            )
        assert not kw.get("is_causal", False)
        if not args.integer_av:
            return scores.softmax(-1).nan_to_num(nan=0.0) @ v
        if mask is not None:
            valid = (mask if mask.dtype == torch.bool else torch.isfinite(mask)).any(-2)
            v = torch.where(valid.unsqueeze(-1), v, 0.0)
        vs = (v.abs().amax(-2, keepdim=True) / 2047).clamp_min(1e-12)
        vi = (v / vs).round().clamp(-2047, 2047).to(device="cpu", dtype=torch.int32)
        ex = (scores - scores.amax(-1, keepdim=True)).exp().nan_to_num(nan=0.0)
        inv = ex.sum(-1, keepdim=True).clamp_min(1e-30).reciprocal()
        pi = (ex * 8191).round().clamp(0, 8191).to(device="cpu", dtype=torch.int32)
        # Each 64-key block <=64*8191*2047=1,073,086,528, exact i32.
        total = None
        for begin in range(0, vi.shape[-2], 64):
            part = (pi[..., begin : begin + 64] @ vi[..., begin : begin + 64, :]).long()
            total = part if total is None else total + part
        return total.float().to(q.device) * (vs * (inv / 8191))

    reports = []
    try:
        for task in TASKS:
            for lang in LANGUAGES:
                qs, ds, rel = dataset(task, lang)
                cs, parents = chunks(tokenizer, ds, 256, "")
                texts = [q["text"] for q in qs] + cs
                result = {
                    "task": task,
                    "language": lang,
                    "token_limit": 256,
                    "queries": len(qs),
                    "documents": len(ds),
                    "chunks": len(parents),
                    "runs": {},
                }
                previous = None
                for mode, sdpa in [
                    ("w8a8-bf16kv", bf16),
                    ("w8a8-qk13-av11" if args.integer_av else "w8a8-qk13-bf16v", q13),
                ]:
                    torch.nn.functional.scaled_dot_product_attention = sdpa
                    version = "-masked-v2" if mode == "w8a8-qk13-bf16v" else ""
                    path = cache / f"{task}-{lang}-{mode}{version}.npz"
                    if path.exists():
                        a = np.load(path)
                        vec = a["vectors"]
                        stats = json.loads(str(a["stats"]))
                        assert np.array_equal(a["parents"], parents)
                    else:
                        vec, stats = encode(
                            model,
                            tokenizer,
                            texts,
                            "mean",
                            256,
                            batch=16 if mode == "w8a8-bf16kv" else 4,
                        )
                        np.savez_compressed(
                            path, vectors=vec, parents=parents, stats=json.dumps(stats)
                        )
                    assert np.isfinite(vec).all(), (task, lang, mode)
                    assert np.allclose(np.linalg.norm(vec, axis=1), 1, atol=1e-5)
                    score, ranks = metrics(
                        vec[: len(qs)], vec[len(qs) :], qs, ds, rel, parents
                    )
                    row = {
                        "metrics": score,
                        "ranks": ranks,
                        "encoding": stats,
                        "vectors_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    if previous is not None:
                        c = (vec * previous).sum(1)
                        row["cosine_to_bf16kv"] = {
                            "min": float(c.min()),
                            "mean": float(c.mean()),
                        }
                    else:
                        previous = vec
                    result["runs"][mode] = row
                    print(task, lang, mode, score, flush=True)
                    (out / f"{task}-{lang}.json").write_text(
                        json.dumps(result, indent=2)
                    )
                reports.append(result)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    (out / "summary.json").write_text(
        json.dumps(
            {
                "model": "hotchpotch/bekko-embedding-v1-a25m",
                "revision": "44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c",
                "note": "Same aligned 50 queries/domain and all positives plus 128 seeded distractors as multilingual-global. CPU exact integer dot, F32/MPS surrounding reference computation; Wasm softmax approximation still differs.",
                "results": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
