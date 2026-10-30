"""Recompute QK13 retrieval results from cached vectors and verify runtime records."""

import hashlib
import json

import numpy as np
from evaluate import metrics
from global_eval import ROOT, dataset


def main():
    out = ROOT / "benchmarks/bekko25-runtime"
    summary = json.loads((out / "quality/summary.json").read_text())
    languages = {}
    for result in summary["results"]:
        task, lang = result["task"], result["language"]
        queries, documents, relevant = dataset(task, lang)
        values = []
        for mode, expected in result["runs"].items():
            suffix = "-masked-v2" if "qk13" in mode else ""
            path = (
                ROOT
                / f".cache/bekko25-runtime/quality/{task}-{lang}-{mode}{suffix}.npz"
            )
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == expected["vectors_sha256"]
            )
            with np.load(path) as cache:
                vectors, parents = cache["vectors"], cache["parents"]
            assert np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            actual, ranks = metrics(
                vectors[: len(queries)],
                vectors[len(queries) :],
                queries,
                documents,
                relevant,
                parents,
            )
            assert ranks == expected["ranks"]
            for key, value in actual.items():
                assert abs(value - expected["metrics"][key]) < 1e-12
            values.append(actual["recall@5"])
        languages.setdefault(lang, []).append(values)
    quality = {
        lang: dict(zip(["baseline_recall5", "qk13_recall5"], np.mean(values, axis=0)))
        for lang, values in languages.items()
    }
    runtimes = []
    for path in sorted(out.glob("iteration-*.json")):
        data = json.loads(path.read_text())
        wasm = ROOT / data["wasm"]
        assert hashlib.sha256(wasm.read_bytes()).hexdigest() == data["wasm_sha256"]
        for row in data["results"]:
            if "instructions" in row:
                assert row["instructions"] < data["query_limit"]
            if "embedding" in row:
                v = np.array(row["embedding"])
                assert v.shape == (384,) and np.isfinite(v).all()
                assert abs(np.linalg.norm(v) - 1) < 1e-5
            if "embedding" in row and path.stem in [
                "iteration-1",
                "iteration-2",
                "iteration-3",
                "iteration-5",
            ]:
                assert row.get("baseline_bit_exact") is True
        runtimes.append({"record": path.name, "wasm_sha256": data["wasm_sha256"]})
    final = json.loads((out / "iteration-8.json").read_text())["results"]
    prior = json.loads((out / "iteration-6.json").read_text())["results"]
    assert len(final) == 9 and all("instructions" in row for row in final)
    for a, b in zip(prior, final):
        assert a["case"] == b["case"]
        if "embedding" in a:
            # Float32 bit equality, including signed zero.
            assert (
                np.array(a["embedding"], dtype=np.float32).tobytes()
                == np.array(b["embedding"], dtype=np.float32).tobytes()
            )
    (out / "verification.json").write_text(
        json.dumps(
            {
                "quality_runs_recomputed": 24,
                "queries_ranked": 1200,
                "quality_recall5": quality,
                "runtime_records": runtimes,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(quality, indent=2))
    print(f"Verified {len(runtimes)} runtime records and 24 retrieval runs")


if __name__ == "__main__":
    main()
