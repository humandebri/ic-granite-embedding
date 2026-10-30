"""Validate the text adapter with Sentence Transformers without image/audio extras."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from evaluate import encode
from gemma2 import ROOT
from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True)
    config.vision_config = config.audio_config = None
    model = AutoModel.from_pretrained(
        args.snapshot, config=config, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    texts = [
        "task: search result | query: How do plants store energy?",
        "title: none | text: Plants store energy as sugars.",
        "task: search result | query: 文書検索とは？",
        "title: none | text: 質問と文章の意味を比較します。",
    ]
    ours, _ = encode(model, tokenizer, texts, "mean", 256, batch=4)
    del model
    # AutoProcessor otherwise requires image/audio packages even with their encoders
    # disabled. Use identical tokenizer files with only the processor_class removed.
    text_tokenizer = ROOT / ".cache/embeddinggemma2/text-tokenizer"
    text_tokenizer.mkdir(exist_ok=True)
    for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]:
        shutil.copyfile(args.snapshot / name, text_tokenizer / name)
    cfg = json.loads((text_tokenizer / "tokenizer_config.json").read_text())
    cfg.pop("processor_class", None)
    (text_tokenizer / "tokenizer_config.json").write_text(json.dumps(cfg))
    transformer = models.Transformer(
        str(args.snapshot),
        tokenizer_name_or_path=str(text_tokenizer),
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        max_seq_length=256,
    )
    reference = SentenceTransformer(
        modules=[transformer, models.Pooling(768), models.Normalize()], device="cpu"
    )
    theirs = reference.encode(
        texts, prompt="", convert_to_numpy=True, show_progress_bar=False
    )
    error = float(abs(ours - theirs).max())
    assert error < 1e-5, error
    report = {
        "shape": list(ours.shape),
        "max_abs_difference": error,
        "min_cosine": float((ours * theirs).sum(1).min()),
        "reference": "SentenceTransformer Transformer + mean Pooling + Normalize matching official modules.json",
        "tokenizer_override": "Identical tokenizer files; only processor_class removed to avoid unused multimodal dependencies",
    }
    (ROOT / "benchmarks/embeddinggemma2/adapter.json").write_text(
        json.dumps(report, indent=2)
    )
    print(report)


if __name__ == "__main__":
    main()
