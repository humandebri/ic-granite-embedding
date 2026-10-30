"""Export a text-only W8 pack and bounded real-token fixtures, not a production service."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
from evaluate import W8A8
from gemma2 import MODEL, REVISION, ROOT
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / ".cache/embeddinggemma2")
    p.add_argument(
        "--report", type=Path, default=ROOT / "benchmarks/embeddinggemma2/fixtures.json"
    )
    p.add_argument("--languages", default="ja,en")
    p.add_argument("--lengths", default="2,16,32,64,128,256")
    args = p.parse_args()
    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    config.vision_config = config.audio_config = None
    model = AutoModel.from_pretrained(
        args.snapshot, config=config, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    descriptors = {}
    payload = bytearray()
    for name, tensor in model.state_dict().items():
        if name.endswith("embed_tokens.weight"):
            continue
        t = tensor.detach().float().contiguous()
        if t.ndim == 2:
            s = t.abs().amax(-1).div(127).clamp_min(1e-12)
            b = (
                (t / s[:, None])
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
                .numpy()
                .tobytes()
            )
            b += s.numpy().astype("<f4").tobytes()
            kind = "i8"
        else:
            b = t.numpy().astype("<f4").tobytes()
            kind = "f32"
        descriptors[name.removeprefix("language_model.")] = {
            "shape": list(t.shape),
            "offset": len(payload),
            "bytes": len(b),
            "kind": kind,
        }
        payload.extend(b)
    header = json.dumps(descriptors, separators=(",", ":")).encode()
    pack = struct.pack("<Q", len(header)) + header + payload
    (cache / "text.pack").write_bytes(pack)
    cases = []
    for language, text in [
        (
            "ja",
            "文書検索では質問と文章の意味を比較します。正確な情報を探すために関連する段落を選びます。",
        ),
        (
            "en",
            "Document retrieval compares the meaning of a question and a passage to find relevant evidence. ",
        ),
        (
            "ar",
            "يبحث النظام في المستندات عن المعلومات المرتبطة بالسؤال ويسترجع الفقرات ذات الصلة. ",
        ),
        ("ko", "문서 검색은 질문과 관련된 정보를 찾아 적절한 문단을 반환합니다. "),
    ]:
        if language not in args.languages.split(","):
            continue
        source = tokenizer.encode(
            "task: search result | query: " + text * 200, add_special_tokens=False
        )
        for n in map(int, args.lengths.split(",")):
            assert 2 <= n <= 256
            ids = [2] + source[: n - 2] + [1]
            with torch.inference_mode():
                embedding = model.get_input_embeddings()(torch.tensor([ids])).float()[0]
                output = (
                    model(
                        torch.tensor([ids]),
                        attention_mask=torch.ones(1, n, dtype=torch.long),
                    )
                    .last_hidden_state.float()
                    .mean(1)
                )
                output = torch.nn.functional.normalize(output, dim=-1)[0]
            cases.append(
                {
                    "name": f"{language}-{n}",
                    "ids": ids,
                    "input": embedding.flatten().tolist(),
                    "fp32": output.tolist(),
                }
            )
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, W8A8(module))
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        return original(q, k.bfloat16().float(), v.bfloat16().float(), *a, **kw)

    torch.nn.functional.scaled_dot_product_attention = sdpa
    for case in cases:
        ids = torch.tensor([case["ids"]])
        traces = []
        hooks = []
        if case["name"] == "ja-2":

            def capture(_module, _args, output, traces=traces):
                traces.append(output.detach().float().flatten().tolist())

            hooks = [
                layer.register_forward_hook(capture)
                for layer in model.language_model.layers
            ]
        with torch.inference_mode():
            h = (
                model(ids, attention_mask=torch.ones_like(ids))
                .last_hidden_state.float()
                .mean(1)
            )
            case["w8a8"] = torch.nn.functional.normalize(h, dim=-1)[0].tolist()
        for hook in hooks:
            hook.remove()
        if traces:
            (cache / "trace.json").write_text(json.dumps(traces))
    (cache / "cases.json").write_text(json.dumps(cases))
    source = {
        "model": MODEL,
        "revision": REVISION,
        "pack_bytes": len(pack),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "cases_sha256": hashlib.sha256((cache / "cases.json").read_bytes()).hexdigest(),
        "languages": args.languages.split(","),
        "lengths": list(map(int, args.lengths.split(","))),
        "note": "Fixture-specific scaled input embeddings replace the lookup table only; all 24 layers, PLE, projection and pooling execute at runtime. Synthetic repeated real-token text with exact lengths, not retrieval accuracy fixtures.",
    }
    args.report.write_text(json.dumps(source, indent=2))
    print(source, flush=True)


if __name__ == "__main__":
    main()
