"""Check the new Harrier last-token adapter against Sentence Transformers."""

import json
import shutil

import torch
from global_eval import CACHE, MODELS, OUT, encode, load_model, model_path
from sentence_transformers import SentenceTransformer, models


def main():
    torch.set_num_threads(2)
    spec = MODELS["harrier"]
    qp = spec["query_prefix"]
    texts = [
        qp + "How do plants store energy?",
        "Plants use photosynthesis to store energy as sugars.",
        qp + "كيف يتم البحث في المستندات؟",
        "문서 검색은 질문과 관련된 정보를 찾습니다.",
    ]
    model, tokenizer = load_model("harrier")
    ours, _ = encode(model, tokenizer, texts, "last", 128, batch=4)
    del model
    source = model_path("harrier")
    token_path = CACHE / "harrier-text-tokenizer"
    token_path.mkdir(exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        if (source / name).exists():
            shutil.copyfile(source / name, token_path / name)
    config = json.loads((token_path / "tokenizer_config.json").read_text())
    config.pop("processor_class", None)
    (token_path / "tokenizer_config.json").write_text(json.dumps(config))
    transformer = models.Transformer(
        str(source),
        tokenizer_name_or_path=str(token_path),
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        processor_kwargs={"padding_side": "right", "model_max_length": 128},
    )
    pooling = models.Pooling(
        640, pooling_mode_mean_tokens=False, pooling_mode_lasttoken=True
    )
    reference = SentenceTransformer(
        modules=[transformer, pooling, models.Normalize()], device="cpu"
    )
    theirs = reference.encode(
        texts, prompt="", show_progress_bar=False, convert_to_numpy=True
    )
    error = float(abs(ours - theirs).max())
    assert error < 1e-5, error
    result = {
        "model": spec,
        "texts": len(texts),
        "dimensions": 640,
        "max_abs_difference": error,
        "min_cosine": float((ours * theirs).sum(1).min()),
        "tokenizer_override": "Identical tokenizer files, processor_class removed; right padding explicitly set",
        "reference": "Sentence Transformers Transformer + last-token Pooling + Normalize",
    }
    (OUT / "harrier-adapter.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
