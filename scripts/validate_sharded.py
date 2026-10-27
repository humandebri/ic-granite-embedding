"""Validate full-context, client-sharded ordinary queries through 5,000 tokens."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
from validate_regression import wasm_exports

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
OUT = ROOT / 'benchmarks/full-context/sharded'
KEYS = ['1365', '1366', '2048', '4096', '5000', 'space-5000', 'cooking-5000', 'weather-5000']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(values):
    return struct.pack('<' + str(len(values)) + 'f', *values)


def cosine(a, b):
    return sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT/'target'))
    wasm = Path(os.environ.get('GRANITE_WASM', target/'wasm32-unknown-unknown/release/granite_canister.wasm')).resolve()
    sha = digest(wasm)
    assert 'canister_query profileStep' not in wasm_exports(wasm.read_bytes())
    reference_dir = CACHE/'next/references'
    manifest = json.loads((reference_dir/'references.json').read_text())
    assert manifest['pack_sha256'] == digest(CACHE/'model.pack')
    for case in manifest['cases']:
        key = case['case_id']
        assert bits(case['embedding']) == (CACHE/f'reference-{key}.f32').read_bytes()
        for prefix, suffix in [('tokens-','.json'), ('reference-','.f32'), ('reference-int8-','.f32')]:
            name = prefix+key+suffix
            assert (reference_dir/name).read_bytes() == (CACHE/name).read_bytes()
    if not args.summarize_only:
        with (CACHE/'next/sharded-validation.log').open('w') as log:
            subprocess.run([str(target/'release/granite-bench'),'5000','--sharded-suite'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    OUT.mkdir(parents=True,exist_ok=True)
    summary = []
    for key in KEYS:
        path = CACHE/f'sharded-{key}-packed-{sha}.json'
        result = json.loads(path.read_text())
        assert result['wasm_sha256'] == sha and result['case_id'] == key
        ids = json.loads((CACHE/f'tokens-{key}.json').read_text())
        token_hash = hashlib.sha256(b''.join(struct.pack('<I',v) for v in ids)).hexdigest()
        assert result['token_ids_sha256'] == token_hash
        assert len(ids) == result['tokens'] and ids[0] == 50281 and ids[-1] == 50282
        assert 50283 not in ids and all(0 <= v < 50368 for v in ids)
        embedding = result['embedding']
        assert len(embedding) == 384 and all(math.isfinite(v) for v in embedding)
        assert abs(sum(v*v for v in embedding)-1) < 1e-4
        reference = struct.unpack('<384f',(CACHE/f'reference-{key}.f32').read_bytes())
        actual_cosine = cosine(embedding, reference)
        assert actual_cosine > .999 and abs(actual_cosine-result['cosine_to_f32']) < 1e-12
        stats = result['stats']
        assert stats['max_instructions'] < 5_000_000_000
        assert stats['max_request_bytes'] < 2*1024**2 and stats['max_response_bytes'] < 3*1024**2
        assert [t['layer'] for t in result['trace']] == list(range(12))
        if key == '5000':
            baseline = json.loads((ROOT/'benchmarks/full-context/sharded-baseline.json').read_text())
            assert bits(embedding) == bits(baseline['embedding']), 'Packing changed the full-context result'
        if key == '1365':
            baseline = json.loads((ROOT/'benchmarks/frontier/final/api-1365.json').read_text())
            assert bits(embedding) == bits(baseline['embedding']), 'Sharding changed public API output'
        shutil.copyfile(path,OUT/f'{key}.json')
        summary.append({'case_id':key,'tokens':len(ids),'cosine_to_f32':actual_cosine,**stats})
    shutil.copyfile(reference_dir/'references.json',OUT/'references.json')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'environment.json').write_text(json.dumps({'wasm_sha256':sha,'query_instruction_limit':5_000_000_000,'subnet':'Application / Production','cases':len(KEYS),'malformed_requests_checked_per_case':9,'reference_pack_sha256':manifest['pack_sha256'],'fixture_sha256':{name:digest(CACHE/name) for key in KEYS for name in [f'tokens-{key}.json',f'reference-{key}.f32',f'reference-int8-{key}.f32']},'bit_exact_5000_raw_to_packed':True,'bit_exact_1365_public_to_sharded':True},indent=2)+'\n')
    print(f'Sharded validation passed: {len(KEYS)} cases; max 5000 tokens.')


if __name__ == '__main__':
    main()
