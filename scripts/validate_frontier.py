"""Validate the adopted kernels, old-domain exactness, and bounded long queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tomllib
from zoneinfo import ZoneInfo
from validate_regression import wasm_exports

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
OUT = ROOT / 'benchmarks/validation'
TARGET = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target'))


def run(name, command):
    with (CACHE / 'frontier' / f'final-{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def bits(values):
    return struct.pack('<' + str(len(values)) + 'f', *values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--already-built', action='store_true')
    parser.add_argument('--summarize-only', action='store_true', help='Recheck completed suite artifacts without rerunning inference')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (CACHE / "frontier").mkdir(parents=True, exist_ok=True)
    assert os.environ.get('POCKET_IC_BIN')
    if not args.already_built and not args.summarize_only:
        run('tests', ['cargo', 'test', '--locked', '-p', 'granite-core', '--features',
                     'approx-gelu,strassen,chunked-gemm,strassen-wo,strassen-qkv,strassen-out'])
        run('build', ['cargo', 'build', '--release', '--locked', '-p', 'granite-canister',
                     '--target', 'wasm32-unknown-unknown'])
        run('bench', ['cargo', 'build', '--release', '--locked', '-p', 'granite-bench'])
    test_counts = re.findall(r'test result: ok\. (\d+) passed; 0 failed', (CACHE / 'frontier/final-tests.log').read_text())
    assert test_counts and int(test_counts[0]) > 0
    shutil.copyfile(CACHE / 'frontier/final-tests.log', CACHE / 'regression-final-tests.log')
    wasm = Path(os.environ.get('GRANITE_WASM', TARGET / 'wasm32-unknown-unknown/release/granite_canister.wasm')).resolve()
    digest = hashlib.sha256(wasm.read_bytes()).hexdigest()
    os.environ['GRANITE_WASM'] = str(wasm)
    exports = wasm_exports(wasm.read_bytes())
    assert 'canister_query profileStep' not in exports
    # All suites use the same immutable module. Reference fixtures and shared
    # 256-token diagnostic results agree; their distinct suites write separate files.
    # Concurrent-suite timings are not used as latency measurements.
    if not args.summarize_only:
        with ThreadPoolExecutor(max_workers=3) as workers:
            checks = [workers.submit(run, 'regression', [sys.executable, 'scripts/validate_regression.py',
                                                        '--already-built', '--output', str(OUT / 'regression')]),
                      workers.submit(run, 'long', [str(TARGET / 'release/granite-bench'), '1365',
                                                  '--stress-long', '--validate-long']),
                      workers.submit(run, 'capacity', [str(TARGET / 'release/granite-bench'), '256',
                                                      '--capacity-range=289:320','--capacity-patterns=3'])]
            for check in checks:
                check.result()
    capacity=json.loads((CACHE/'capacity.json').read_text())
    assert capacity['wasm_sha256']==digest and len(capacity['cases'])==96
    for c in capacity['cases']:
        if c['raw_instructions'] is not None:assert c['raw_instructions']<5_000_000_000
        else:assert 'CanisterInstructionLimitExceeded' in c['error']
    shutil.copyfile(CACHE/'capacity.json',OUT/'capacity.json')
    regression = json.loads((OUT / 'regression/environment.json').read_text())
    assert regression['wasm_sha256'] == digest and regression['sweep_bit_exact_cases'] == 511
    stress = json.loads((CACHE / 'stress-long.json').read_text())
    assert stress['wasm_sha256'] == digest and len(stress['cases']) == 96
    for c in stress['cases']:
        assert 513 <= c['tokens'] <= 1365 and 1 < c['queries'] <= 12
        assert c['max_instructions'] < 5_000_000_000
        assert len(c['embedding']) == 384 and all(math.isfinite(v) for v in c['embedding'])
        assert abs(sum(v*v for v in c['embedding']) - 1) < 1e-4
    previous_long=json.loads((ROOT/'benchmarks/frontier/final/stress-long.json').read_text())
    for current,previous in zip(stress['cases'],previous_long['cases']):
        assert current['tokens']==previous['tokens'] and current['pattern']==previous['pattern'] and current['ids']==previous['ids']
        assert bits(current['embedding'])==bits(previous['embedding']), 'Long-domain output changed'
    shutil.copyfile(CACHE / 'stress-long.json', OUT / 'stress-long.json')
    keys = ['640', '768', '1024', '1280', '1365']
    keys += [f'{topic}-{n}' for topic in ['space', 'cooking', 'weather'] for n in [1280, 1365]]
    for name in ['references','references-max']:
        reference=CACHE/'frontier'/name/'references.json'
        manifest=json.loads(reference.read_text())
        assert manifest['pack_sha256']==hashlib.sha256((CACHE/'model.pack').read_bytes()).hexdigest()
        shutil.copyfile(reference,OUT/f'{name}-provenance.json')
    summary = []
    for key in keys:
        c = json.loads((CACHE / f'api-{key}.json').read_text())
        assert c['wasm_sha256'] == digest and c['layers'] == 12
        assert c['cosine_to_f32'] > .999
        previous=json.loads((ROOT/'benchmarks/frontier/final'/f'api-{key}.json').read_text())
        assert bits(c['embedding'])==bits(previous['embedding']), 'Long reference output changed'
        assert c['stats']['max_instructions'] < 5_000_000_000
        assert c['stats']['max_request_bytes'] < 2*1024**2 and c['stats']['max_response_bytes'] < 3*1024**2
        shutil.copyfile(CACHE / f'api-{key}.json', OUT / f'api-{key}.json')
        summary.append({k: c[k] for k in ['case_id', 'tokens', 'cosine_to_f32']} | c['stats'])
    prof = json.loads((ROOT / 'benchmarks/frontier/profile-1365.json').read_text())
    public = json.loads((OUT / 'api-1365.json').read_text())
    assert bits(prof['embedding']) == bits(public['embedding']), 'Layer-by-layer and public long-query output differ'
    current_profile = json.loads((ROOT/'benchmarks/query-fusion/profile-1365.json').read_text())
    assert bits(current_profile['embedding']) == bits(public['embedding'])
    av = json.loads((ROOT/'benchmarks/query-fusion/av.json').read_text())
    qk = json.loads((ROOT/'benchmarks/query-fusion/qk.json').read_text())
    linear = json.loads((ROOT/'benchmarks/query-fusion/linear.json').read_text())
    assert av['exact_i64_cases'] == qk['exact_i64_cases'] == qk['exact_previous_cases'] == 44
    codec=json.loads((ROOT/'benchmarks/query-fusion/codec.json').read_text())
    assert codec['exact_scalar_cases'] == 30 and codec['wasm_sha256'] == current_profile['wasm_sha256']
    assert linear['exact_native_matches'] == 112
    assert av['wasm_sha256'] == qk['wasm_sha256'] == linear['wasm_sha256'] == current_profile['wasm_sha256']
    source = [ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'rust-toolchain.toml', ROOT/'.cargo/config.toml', ROOT/'requirements.txt']
    source += sorted((ROOT/'crates').rglob('*.rs')) + sorted((ROOT/'crates').rglob('Cargo.toml')) + sorted((ROOT/'scripts').glob('*.py'))
    cfg = tomllib.loads((ROOT/'crates/canister/Cargo.toml').read_text())
    code = (ROOT/'crates/core/src/lib.rs').read_text()
    meta = {'date': datetime.now(ZoneInfo('Asia/Tokyo')).date().isoformat(), 'wasm_sha256': digest,
            'wasm_bytes': wasm.stat().st_size, 'features': cfg['features']['default'], 'profiling': False,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
            'single_query_max_tokens': int(re.search(r'SINGLE_QUERY_MAX_TOKENS: usize = (\d+)', code)[1]),
            'max_tokens': int(re.search(r'DEFAULT_MAX_TOKENS: usize = (\d+)', code)[1]),
            'old_domain_lengths_bit_exact': 511, 'old_domain_stress_cases': 57, 'old_domain_api_cases': 16,
            'long_stress_cases': 96, 'long_stress_distinct_lengths': len({c['tokens'] for c in stress['cases']}),
            'capacity_cases':96, 'long_reference_cases': len(keys), 'long_stress_bit_exact_cases':96, 'long_reference_bit_exact_cases':len(keys), 'av_i64_exact_cases': 44, 'long_layer_boundary_bit_exact_cases': 1,
            'codec_scalar_exact_cases':30, 'qk_i64_exact_cases':44, 'linear_native_exact_cases':112, 'profile_wasm_sha256': current_profile['wasm_sha256'], 'exports': exports,
            'query_instruction_limit': 5_000_000_000, 'subnet': 'Application / Production',
            'timing_note': 'Regression and long validation ran concurrently; use separate uncached sequential measurements for latency.'}
    (OUT/'environment.json').write_text(json.dumps(meta, indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(meta, indent=2))
    print('Frontier validation passed.')


if __name__ == '__main__':
    main()
