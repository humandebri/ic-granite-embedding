"""Materialize an isolated 13-layer Bekko probe using the validated a8m kernels."""

import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from evaluate import W8A8
from expanded import SPECS
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/multilingual-global/bekko25"
OUT = ROOT / "benchmarks/multilingual-global"
SOURCE_REVISION = "e281da1a289060101a0bfceb3ecf48403415b8d0"


def materialize_sources(dest):
    """Keep historical probe measurements independent of production refactors."""
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                "--name-only",
                SOURCE_REVISION,
                "--",
                "crates/bekko-core",
                "crates/kernel-gen",
                "crates/core/src/strassen.rs",
                ".cargo",
                "Cargo.lock",
                "rust-toolchain.toml",
            ],
            cwd=ROOT,
        )
        .decode()
        .splitlines()
    )
    for name in paths:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(
                ["git", "show", f"{SOURCE_REVISION}:{name}"], cwd=ROOT
            )
        )


def main():
    torch.set_num_threads(2)
    spec = SPECS["bekko25"]
    path = Path(
        snapshot_download(
            spec["id"],
            revision=spec["revision"],
            allow_patterns=["*.json", "*.safetensors"],
            local_files_only=True,
        )
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    descriptors = {}
    payload = bytearray()
    for name, t in load_file(path / "model.safetensors").items():
        a = t.float().numpy()
        if a.ndim == 2 and "tok_embeddings" not in name:
            s = np.maximum(abs(a).max(1) / 127, 1e-12).astype("<f4")
            b = (
                np.clip(np.round(a / s[:, None]), -127, 127).astype("i1").tobytes()
                + s.tobytes()
            )
            kind = "i8"
        else:
            b = a.astype("<f4").tobytes()
            kind = "f32"
        descriptors[name] = {
            "shape": list(a.shape),
            "offset": len(payload),
            "bytes": len(b),
            "kind": kind,
        }
        payload.extend(b)
    header = json.dumps(descriptors, separators=(",", ":")).encode()
    pack = struct.pack("<Q", len(header)) + header + payload
    (CACHE / "model.pack").write_bytes(pack)
    model = AutoModel.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
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
        source = tokenizer.encode(text * 200, add_special_tokens=False)
        for n in [64, 128, 256]:
            ids = [2] + source[: n - 2] + [1]
            with torch.inference_mode():
                h = model(torch.tensor([ids])).last_hidden_state.float().mean(1)
                fp = torch.nn.functional.normalize(h, dim=-1)[0].tolist()
            cases.append({"name": f"{lang}-{n}", "ids": ids, "fp32": fp})
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
            with torch.inference_mode():
                h = model(torch.tensor([c["ids"]])).last_hidden_state.float().mean(1)
                c["w8a8"] = torch.nn.functional.normalize(h, dim=-1)[0].tolist()
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    (CACHE / "cases.json").write_text(json.dumps(cases))
    provenance = {
        "source_revision": SOURCE_REVISION,
        "model": spec,
        "pack_bytes": len(pack),
        "pack_sha256": hashlib.sha256(pack).hexdigest(),
        "cases_sha256": hashlib.sha256((CACHE / "cases.json").read_bytes()).hexdigest(),
        "materialized_sources": {},
    }
    dest = CACHE / "port"
    materialize_sources(dest)
    core = dest / "crates/bekko-core/src/lib.rs"
    source = (
        core.read_text()
        .replace("for l in 0..4 {", "for l in 0..13 {")
        .replace("for layer in 0..4 {", "for layer in 0..13 {")
    )
    source = source.replace("hotchpotch/bekko-embedding-v1-a8m", spec["id"]).replace(
        "c721113d59a1d91b447450324f51c4b3332c924a", spec["revision"]
    )
    source = source.replace("400967494", str(len(pack))).replace(
        "1feddfa11f8e232a30b39c4965328a396fa9388dc65bb06cad043045e2dc56ef",
        provenance["pack_sha256"],
    )
    core.write_text(source)
    (dest / "Cargo.toml").write_text("""[workspace]
members=["crates/bekko-core","crates/kernel-gen","crates/probe"]
resolver="2"
[workspace.dependencies]
serde={version="1",features=["derive"]}
serde_json="1"
[profile.release]
opt-level=3
lto="thin"
codegen-units=1
strip="debuginfo"
""")
    probe = dest / "crates/probe"
    (probe / "src").mkdir(parents=True, exist_ok=True)
    (probe / "Cargo.toml").write_text("""[package]
name="bekko25-probe"
version="0.1.0"
edition="2024"
[lib]
crate-type=["cdylib","rlib"]
[dependencies]
bekko-core={path="../bekko-core"}
ic-cdk="=0.20.3"
candid="=0.10.39"
serde_bytes="0.11"
""")
    (probe / "src/lib.rs").write_text("""use std::cell::RefCell;
thread_local!{static PACK:RefCell<Vec<u8>>=const{RefCell::new(Vec::new())};static MODEL:RefCell<Option<bekko_core::Model>>=const{RefCell::new(None)};}
#[ic_cdk::update]
fn upload(offset:u64,bytes:serde_bytes::ByteBuf){assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));assert!(bytes.len()<=1_800_000);PACK.with_borrow_mut(|p|{assert_eq!(p.len(),offset as usize);assert!(p.len()+bytes.len()<=430_000_000);p.extend(bytes);});}
#[ic_cdk::update]
fn activate(){assert!(ic_cdk::api::is_controller(&ic_cdk::api::msg_caller()));let p=PACK.with_borrow_mut(std::mem::take);MODEL.with_borrow_mut(|m|*m=Some(bekko_core::Model::load(&p).unwrap()));}
#[ic_cdk::query]
fn embed(ids:Vec<u32>)->(Vec<f32>,u64){let t=ic_cdk::api::performance_counter(0);let v=MODEL.with_borrow(|m|m.as_ref().unwrap().full(&ids).unwrap());(v,ic_cdk::api::performance_counter(0)-t)}
""")
    for file in dest.rglob("*"):
        if file.is_file() and file.suffix in [".rs", ".toml"]:
            provenance["materialized_sources"][str(file.relative_to(dest))] = (
                hashlib.sha256(file.read_bytes()).hexdigest()
            )
    (OUT / "bekko25-fixtures.json").write_text(json.dumps(provenance, indent=2))
    print(dest, flush=True)


if __name__ == "__main__":
    main()
