"""Export/check the pinned a25m pack and prepare tokenizer-only canister cases."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-pack", type=Path)
    parser.add_argument("--retrieval", action="store_true")
    args = parser.parse_args()
    cache = ROOT / ".cache/bekko25"
    cache.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": "hotchpotch/bekko-embedding-v1-a25m",
        "revision": "44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c",
    }
    path = snapshot_download(
        spec["id"],
        revision=spec["revision"],
        allow_patterns=["*.json", "*.safetensors"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    pack_path = cache / "model.pack"
    if args.reuse_pack:
        # Hardlink the large immutable pack when possible; preserve old caches.
        if not pack_path.exists():
            try:
                pack_path.hardlink_to(args.reuse_pack.resolve())
            except OSError:
                import shutil

                shutil.copyfile(args.reuse_pack, pack_path)
    elif not pack_path.exists():
        # Same deterministic format as the a8m exporter, pinned to a25m.
        import struct

        import numpy as np
        from safetensors.torch import load_file

        payload = bytearray()
        descriptors = {}
        for name, tensor in load_file(Path(path) / "model.safetensors").items():
            a = tensor.float().numpy()
            if a.ndim == 2 and "tok_embeddings" not in name:
                scale = np.maximum(abs(a).max(1) / 127, 1e-12).astype("<f4")
                data = (
                    np.clip(np.round(a / scale[:, None]), -127, 127)
                    .astype("i1")
                    .tobytes()
                    + scale.tobytes()
                )
                kind = "i8"
            else:
                data, kind = a.astype("<f4").tobytes(), "f32"
            descriptors[name] = {
                "shape": list(a.shape),
                "offset": len(payload),
                "bytes": len(data),
                "kind": kind,
            }
            payload.extend(data)
        header = json.dumps(descriptors, separators=(",", ":")).encode()
        pack_path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    assert pack_path.stat().st_size == 418404508
    assert (
        hashlib.sha256(pack_path.read_bytes()).hexdigest()
        == "8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724"
    )
    # Pinned cached numerical fixtures are optional; tokenizer smoke/sweep works
    # from a clean model download without the exploratory experiments.
    cases = []
    texts = {
        "en": "Document retrieval finds evidence relevant to a question. ",
        "de": "Die Suche findet relevante Informationen in Dokumenten. ",
        "es": "La búsqueda encuentra documentos relevantes para la pregunta. ",
        "fr": "La recherche trouve des documents pertinents pour la question. ",
        "ar": "يبحث النظام في المستندات عن المعلومات المرتبطة بالسؤال. ",
        "ko": "문서 검색은 질문과 관련된 정보를 찾아 적절한 문단을 반환합니다. ",
    }
    for lang, text in texts.items():
        source = tokenizer.encode(text * 300, add_special_tokens=False)
        for length in [2, 31, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256]:
            cases.append(
                {"case": f"{lang}-{length}", "ids": [2] + source[: length - 2] + [1]}
            )
    (cache / "cases.json").write_text(json.dumps(cases))
    if args.retrieval:
        sys.path.insert(0, str(ROOT / "experiments/multilingual"))
        from expanded import chunks
        from global_eval import CACHE, DATA_REVISION, LANGUAGES, TASKS, dataset
        from huggingface_hub import hf_hub_download

        data_path = CACHE / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            for lang in LANGUAGES:
                for split in ["queries", "corpus", "qrels"]:
                    name = f"{task}_{split}_{lang}.parquet"
                    target = data_path / name
                    if not target.exists():
                        source = Path(
                            hf_hub_download(
                                "LiquidAI/nanobeir-multilingual-extended",
                                name,
                                repo_type="dataset",
                                revision=DATA_REVISION,
                            )
                        )
                        import shutil

                        shutil.copyfile(source, target)
            selection = ROOT / f"benchmarks/multilingual-global/{task}-selection.json"
            if selection.exists():
                for source in json.loads(selection.read_text())["sources"]:
                    assert (
                        hashlib.sha256(
                            (data_path / source["file"]).read_bytes()
                        ).hexdigest()
                        == source["sha256"]
                    )

        groups = []
        for task in TASKS:
            for lang in LANGUAGES:
                qs, ds, rel = dataset(task, lang)
                cs, parents = chunks(tokenizer, ds, 256, "")
                texts = [q["text"] for q in qs] + cs
                ids = tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=256,
                    padding=False,
                )["input_ids"]
                assert all(
                    2 <= len(v) <= 256 and v[0] == 2 and v[-1] == 1 and 0 not in v
                    for v in ids
                )
                groups.append(
                    {
                        "task": task,
                        "language": lang,
                        "query_ids": [q["_id"] for q in qs],
                        "document_ids": [d["_id"] for d in ds],
                        "qrels": {q: sorted(r) for q, r in rel.items()},
                        "parents": list(map(int, parents)),
                        "ids": ids,
                    }
                )
        (cache / "retrieval.json").write_text(json.dumps(groups))
        print(f"Prepared {len(groups)} retrieval groups")
    (cache / "model.json").write_text(
        json.dumps(
            {
                "model": spec,
                "pack_bytes": pack_path.stat().st_size,
                "pack_sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    print(f"Prepared {len(cases)} exact-length cases and verified pinned pack")


if __name__ == "__main__":
    main()
