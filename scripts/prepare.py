"""Acquire pinned public weights, export per-output-channel INT8, generate references."""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

MODEL = "ibm-granite/granite-embedding-small-english-r2"
REV = "2ab6fa8ea2d674564defd37171ae19079b864b33"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".cache"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--lengths", default="32,64,128,256,512,1024,5000")
    parser.add_argument("--stress", action="store_true", help="Also generate three distinct 512-token texts")
    parser.add_argument("--stress-lengths", default="256,512", help="Token lengths for the three stress topics")
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    path = snapshot_download(MODEL, revision=REV, allow_patterns=["*.json", "*.safetensors"])
    original = Path(path) / "model.safetensors"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert digest == "dcbfaf2ee50d358763cbc0c08e5b045ecbf6aeb02dc0e86ffb910029bf9ebb5f"
    weights = load_file(original)
    metadata, data = {}, bytearray()
    for name, tensor in weights.items():
        value = tensor.float().numpy()
        offset = len(data)
        if value.ndim == 2 and "tok_embeddings" not in name:
            scales = np.maximum(np.abs(value).max(axis=1) / 127, 1e-12).astype("<f4")
            quant = np.clip(np.round(value / scales[:, None]), -127, 127).astype("i1")
            data += quant.tobytes() + scales.tobytes()
            kind = "i8"
        else:
            data += value.astype("<f4").tobytes()
            kind = "f32"
        metadata[name] = {"shape": list(value.shape), "offset": offset, "bytes": len(data) - offset, "kind": kind}
    header = json.dumps(metadata, separators=(",", ":")).encode()
    pack = struct.pack("<Q", len(header)) + header + data
    (OUT / "model.pack").write_bytes(pack)
    print(json.dumps({"pack_bytes": len(pack), "tensor_names": list(metadata)}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    passage = (
        "The Internet Computer executes WebAssembly smart contracts called canisters. "
        "A search system converts documents into numerical embeddings and compares their meaning. "
        "ModernBERT uses bidirectional attention, combining local context with global information. "
        "Quantization reduces the memory needed for neural network weights. "
        "This experiment preserves every token of a long document while measuring computation. "
    )
    # Real tokenizer IDs, exactly the requested total length including CLS and SEP.
    body = tokenizer.encode(passage * 150, add_special_tokens=False)
    references = {"model": MODEL, "revision": REV, "weights_sha256": digest,
                  "pack_sha256": hashlib.sha256(pack).hexdigest(), "cases": []}
    specs = [(n, str(n), body) for n in map(int, args.lengths.split(","))]
    if args.stress:
        topics = {
            "space": "Astronomers study stars, galaxies, planetary orbits and the expansion of the universe. Telescopes measure the light emitted by distant objects. A satellite can observe the atmosphere of a planet during a transit. ",
            "cooking": "A baker measures flour and water, mixes the dough and allows it to rise. The oven temperature and baking time change the texture of bread. Fermentation creates gas that expands inside the dough. ",
            "weather": "Meteorologists observe air pressure, temperature, humidity and wind speed. Weather stations record changes over time. Moist air rises, cools and forms clouds, while pressure systems influence the movement of storms. ",
        }
        specs += [(length, f"{name}-{length}", tokenizer.encode(text * 100, add_special_tokens=False)) for name, text in topics.items() for length in map(int, args.stress_lengths.split(","))]
    for length, case_id, source in specs:
        assert 2 <= length <= 5000, "supported reference length: 2..5000"
        if len(source) < length - 2:
            source = source * ((length - 2 + len(source) - 1) // len(source))
        ids = [50281] + source[:length - 2] + [50282]
        assert len(ids) == length
        (OUT / f"tokens-{case_id}.json").write_text(json.dumps(ids))
        started = time.monotonic()
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, length, dtype=torch.long))
            vector = torch.nn.functional.normalize(result.last_hidden_state[0, 0].float(), dim=0).numpy()
        vector.astype("<f4").tofile(OUT / f"reference-{case_id}.f32")
        case = {"case_id": case_id, "tokens": length, "reference_seconds": time.monotonic() - started, "embedding": vector.tolist()}
        references["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "embedding"}), flush=True)
        (OUT / "references.json").write_text(json.dumps(references, indent=2))

    class W8A8(torch.nn.Module):
        def __init__(self, name, original):
            super().__init__()
            w = original.weight.detach().float()
            descriptor = metadata[name + ".weight"]
            offset = descriptor["offset"]
            count = w.numel()
            quant = np.frombuffer(data, dtype="i1", count=count, offset=offset).copy().reshape(w.shape)
            scales = np.frombuffer(data, dtype="<f4", count=w.shape[0], offset=offset + count).copy()
            self.register_buffer("scales", torch.from_numpy(scales))
            self.register_buffer("quant", torch.from_numpy(quant).float())

        def forward(self, x):
            scales = x.abs().amax(dim=-1, keepdim=True).div(127).clamp_min(1e-12)
            quant = (x / scales).round().clamp(-127, 127)
            return torch.nn.functional.linear(quant, self.quant) * (scales * self.scales)

    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, W8A8(name, module))
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def transport_sdpa(q, k, v, *args, **kwargs):
        return original_sdpa(q, k.bfloat16().float(), v.bfloat16().float(), *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = transport_sdpa
    for case in references["cases"]:
        length = case["tokens"]
        case_id = case["case_id"]
        ids = json.loads((OUT / f"tokens-{case_id}.json").read_text())
        started = time.monotonic()
        with torch.inference_mode():
            hidden = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, length, dtype=torch.long)).last_hidden_state
            vector = torch.nn.functional.normalize(hidden[0, 0].float(), dim=0).numpy()
        vector.astype("<f4").tofile(OUT / f"reference-int8-{case_id}.f32")
        expected = np.fromfile(OUT / f"reference-{case_id}.f32", dtype="<f4")
        case["w8a8_reference_seconds"] = time.monotonic() - started
        case["w8a8_cosine_to_f32"] = float(np.dot(expected, vector))
        print(json.dumps({k: v for k, v in case.items() if k != "embedding"}), flush=True)
    (OUT / "references.json").write_text(json.dumps(references, indent=2))


if __name__ == "__main__":
    main()
