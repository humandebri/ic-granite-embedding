"""Convert the pinned a25m F32 vocabulary to independently readable INT8 rows."""
import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724"
COMPACT_SHA256 = "71b8c47a0a508dd5cd3e704d83934e900c8e634d2c52e7a02a85e882fa5be9c4"
COMPACT_BYTES = 124516506
VOCAB_NAME = "embeddings.tok_embeddings.weight"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def convert(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    if digest(source) != ORIGINAL_SHA256:
        raise ValueError("original pack SHA-256 mismatch")
    with source.open("rb") as src:
        header_len = struct.unpack("<Q", src.read(8))[0]
        old = json.loads(src.read(header_len))
        start = 8 + header_len
        items = sorted(old.items(), key=lambda item: item[1]["offset"])
        if old[VOCAB_NAME]["shape"] != [256000, 384] or old[VOCAB_NAME]["kind"] != "f32":
            raise ValueError("unexpected vocabulary descriptor")
        desc = {}
        old_offset = new_offset = 0
        for name, item in items:
            if item["offset"] != old_offset:
                raise ValueError("non-contiguous input pack")
            length = 256000 * 388 if name == VOCAB_NAME else item["bytes"]
            desc[name] = {"shape": item["shape"], "offset": new_offset, "bytes": length,
                          "kind": "i8-row" if name == VOCAB_NAME else item["kind"]}
            old_offset += item["bytes"]
            new_offset += length
        if start + old_offset != source.stat().st_size:
            raise ValueError("input pack length mismatch")
        encoded = json.dumps(desc, separators=(",", ":")).encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            with temporary.open("wb") as out:
                out.write(struct.pack("<Q", len(encoded)))
                out.write(encoded)
                for name, item in items:
                    src.seek(start + item["offset"])
                    if name == VOCAB_NAME:
                        for _ in range(256):
                            x = np.frombuffer(src.read(1000 * 384 * 4), dtype="<f4").reshape(1000, 384)
                            if not np.isfinite(x).all():
                                raise ValueError("nonfinite vocabulary")
                            scales = np.maximum(np.max(np.abs(x), axis=1), np.float32(1e-12)) / np.float32(127)
                            q = x / scales[:, None]
                            q = (np.sign(q) * np.floor(np.abs(q).astype(np.float64) + 0.5)).clip(-127, 127).astype(np.int8)
                            packed = np.empty((1000, 388), dtype=np.uint8)
                            packed[:, :384] = q.view(np.uint8)
                            packed[:, 384:] = scales.astype("<f4").view(np.uint8).reshape(-1, 4)
                            out.write(packed.tobytes())
                    else:
                        remaining = item["bytes"]
                        while remaining:
                            chunk = src.read(min(remaining, 1048576))
                            if not chunk:
                                raise ValueError("truncated input pack")
                            out.write(chunk)
                            remaining -= len(chunk)
            if temporary.stat().st_size != COMPACT_BYTES or digest(temporary) != COMPACT_SHA256:
                raise ValueError("compact pack differs from pinned export")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination.stat().st_size, digest(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/bekko25/model.pack")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/bekko25/model-int8.pack")
    args = parser.parse_args()
    size, sha = convert(args.source, args.output)
    print(f"compact pack: {size} bytes sha256={sha}")


if __name__ == "__main__":
    main()
