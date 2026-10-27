"""Produce bounded Candid token IDs for the dedicated Bekko canister."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    text = args.text if args.text is not None else args.input.read_text()
    tokenizer = AutoTokenizer.from_pretrained(
        "hotchpotch/bekko-embedding-v1-a8m",
        revision="c721113d59a1d91b447450324f51c4b3332c924a",
    )
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if not 2 <= len(ids) <= 256:
        p.error(
            f"{len(ids)} tokens including BOS/EOS; split the text into chunks of at most 256 tokens"
        )
    if ids[0] != 2 or ids[-1] != 1 or 0 in ids:
        p.error("expected unpadded BOS ... EOS input")
    result = "(vec { " + "; ".join(f"{i} : nat32" for i in ids) + " })\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
