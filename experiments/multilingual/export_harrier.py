"""Pinned Harrier W8 fixture-only encoder pack and independent numerical references."""

import hashlib
import json
import struct

import torch
from evaluate import W8A8
from global_eval import CACHE, MODELS, OUT, load_model


def main():
    torch.set_num_threads(2)
    model, tokenizer = load_model("harrier")
    dest = CACHE / "harrier"
    dest.mkdir(exist_ok=True)
    payload = bytearray()
    metadata = {}
    for name, tensor in model.state_dict().items():
        if name == "embed_tokens.weight":
            continue
        t = tensor.float().contiguous()
        if t.ndim == 2:
            s = t.abs().amax(-1).div(127).clamp_min(1e-12)
            b = (t / s[:, None]).round().clamp(-127, 127).to(
                torch.int8
            ).numpy().tobytes() + s.numpy().astype("<f4").tobytes()
            kind = "i8"
        else:
            b = t.numpy().astype("<f4").tobytes()
            kind = "f32"
        metadata[name] = {
            "shape": list(t.shape),
            "offset": len(payload),
            "bytes": len(b),
            "kind": kind,
        }
        payload.extend(b)
    header = json.dumps(metadata, separators=(",", ":")).encode()
    pack = struct.pack("<Q", len(header)) + header + payload
    (dest / "text.pack").write_bytes(pack)
    cases = []
    for lang, text in [
        (
            "en",
            "Document retrieval compares questions and passages to identify relevant evidence. ",
        ),
        (
            "ar",
            "يبحث النظام في المستندات عن المعلومات المرتبطة بالسؤال ويسترجع الفقرات ذات الصلة. ",
        ),
        ("ko", "문서 검색은 질문과 관련된 정보를 찾아 적절한 문단을 반환합니다. "),
    ]:
        source = tokenizer.encode(MODELS["harrier"]["query_prefix"] + text * 200)
        for n in [16, 64, 128]:
            ids = source[:n]
            with torch.inference_mode():
                inputs = model.get_input_embeddings()(torch.tensor([ids]))[0].float()
                h = model(torch.tensor([ids])).last_hidden_state[0, -1].float()
                fp = torch.nn.functional.normalize(h, dim=-1).tolist()
            cases.append(
                {
                    "name": f"{lang}-{n}",
                    "input": inputs.flatten().tolist(),
                    "ids": ids,
                    "fp32": fp,
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
    try:
        for c in cases:
            trace = []
            hooks = []
            if c["name"] == "en-64":

                def capture(_m, _args, output, _trace=trace):
                    _trace.append(output.detach().float().flatten().tolist())

                hooks = [layer.register_forward_hook(capture) for layer in model.layers]
            with torch.inference_mode():
                h = model(torch.tensor([c["ids"]])).last_hidden_state[0, -1].float()
                c["w8a8"] = torch.nn.functional.normalize(h, dim=-1).tolist()
            for hook in hooks:
                hook.remove()
            if trace:
                (dest / "trace.json").write_text(json.dumps(trace))
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    (dest / "cases.json").write_text(json.dumps(cases))
    report = {
        "model": MODELS["harrier"],
        "pack_bytes": len(pack),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "cases_sha256": hashlib.sha256((dest / "cases.json").read_bytes()).hexdigest(),
        "fixture_only": True,
        "note": "Scaled embedding lookup only is exported; all 18 causal encoder layers and last-token pooling run in the canister.",
    }
    (OUT / "harrier-fixtures.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
