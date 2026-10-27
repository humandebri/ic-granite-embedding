"""Validate selected optimizations, all supported single-query lengths, and exact regression outputs."""
import argparse
import hashlib
import json
import math
import os
import re
import platform
import subprocess
import struct
import tomllib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache'
TARGET=Path(os.environ.get('CARGO_TARGET_DIR', ROOT/'target'))
OUT=ROOT/'benchmarks/validation/regression'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(label,args):
    with (CACHE/f'regression-{label}.log').open('w') as log:
        subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def cosine(a,b):return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def wasm_exports(data):
    assert data[:8]==b'\0asm\1\0\0\0'
    at=8
    def leb():
        nonlocal at
        value=0;shift=0
        while True:
            byte=data[at];at+=1;value|=(byte&127)<<shift
            if byte<128:return value
            shift+=7
    names=[]
    while at<len(data):
        section=data[at];at+=1;size=leb();end=at+size
        if section==7:
            for _ in range(leb()):
                size=leb();names.append(data[at:at+size].decode());at+=size
                at+=1;leb()
        at=end
    return names


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,help='Directory for this validation run; historical references stay pinned')
    parser.add_argument('--already-built',action='store_true',help='Use the freshly tested normal build already in target/')
    args=parser.parse_args()
    baseline_out=ROOT/'benchmarks/exhaustive-optimization'
    OUT=(args.output or OUT).resolve()
    default_features=tomllib.loads((ROOT/'crates/canister/Cargo.toml').read_text())['features']['default']
    single_max=int(re.search(r'SINGLE_QUERY_MAX_TOKENS: usize = (\d+)',(ROOT/'crates/core/src/lib.rs').read_text())[1])
    assert os.environ.get('POCKET_IC_BIN')
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.already_built:
        run('final-tests',['cargo','test','--locked','-p','granite-core','--features','approx-gelu,strassen,chunked-gemm,strassen-wo,strassen-qkv,strassen-out'])
        run('production-build',['cargo','build','--release','--locked','-p','granite-canister','--target','wasm32-unknown-unknown'])
        run('bench-build',['cargo','build','--release','--locked','-p','granite-bench'])
    test_counts = re.findall(r'test result: ok\. (\d+) passed; 0 failed', (CACHE/'regression-final-tests.log').read_text())
    assert test_counts and int(test_counts[0]) > 0
    wasm=Path(os.environ.get('GRANITE_WASM', TARGET/'wasm32-unknown-unknown/release/granite_canister.wasm'))
    wasm_hash=digest(wasm)
    exports=wasm_exports(wasm.read_bytes())
    assert 'canister_query startEmbedding' in exports and 'canister_query profileStep' not in exports
    (OUT/'module-exports.json').write_text(json.dumps({'wasm_sha256':wasm_hash,'exports':exports},indent=2)+'\n')
    candid=subprocess.check_output(['cargo','run','--release','--locked','-p','granite-canister','--example','export_candid'],cwd=ROOT)
    assert candid.strip()==(ROOT/'crates/canister/granite.did').read_bytes().strip(),'Public Candid changed'
    assert b'profileStep' not in candid,'Profiling endpoint leaked into production'
    os.environ['GRANITE_WASM']=str(wasm.resolve())
    run('api-suite',[str(TARGET/'release/granite-bench'),'256','--validate-regression','--stress','--sweep','--repeat-single'])
    raw=json.loads((CACHE/f'probe-single-{wasm_hash}.json').read_text())
    assert raw['wasm_sha256']==wasm_hash
    assert len(raw['cases'])==4 and all(r['ok'] and r['instructions']<5_000_000_000 for r in raw['cases'])
    pre_rope=ROOT/'benchmarks/goal-profile/pre-rope-single.json'
    rope_matches=0
    if pre_rope.exists():
        before=json.loads(pre_rope.read_text())
        for result in raw['cases']:
            previous=next(r for r in before['cases'] if r['case']==result['case'])
            bits=lambda v:struct.pack('<'+str(len(v))+'f',*v)
            assert bits(result['embedding'])==bits(previous['embedding']),'RoPE preparation changed output'
            rope_matches+=1
    (OUT/'ordinary-single-256.json').write_text(json.dumps(raw,indent=2)+'\n')
    keys=[str(n) for n in [32,64,128,160,176,256,512]]
    keys += [f'{name}-{n}' for name in ['space','cooking','weather'] for n in [160,256,512]]
    results=[]
    for key in keys:
        r=json.loads((CACHE/f'api-{key}.json').read_text())
        assert r['wasm_sha256']==wasm_hash and r['layers']==12
        assert r['stats']['max_instructions']<5_000_000_000
        assert r['stats']['max_request_bytes']<2*1024**2 and r['stats']['max_response_bytes']<3*1024**2
        assert r['cosine_to_f32']>0.999,f'FP32 accuracy {key}'
        assert r['stats']['queries']==(1 if r['tokens']<=single_max else 2)
        assert all(math.isfinite(v) for v in r['embedding']) and len(r['embedding'])==384
        assert abs(sum(v*v for v in r['embedding'])-1)<1e-4
        previous=ROOT/'benchmarks/single-query'/f'api-{key}.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            r['cosine_to_previous']=cosine(r['embedding'],old['embedding'])
            r['max_abs_to_previous']=max(abs(a-b) for a,b in zip(r['embedding'],old['embedding']))
            assert r['cosine_to_previous']>0.999,f'Previous-module accuracy {key}'
        baseline=json.loads((ROOT/'benchmarks/further-optimization'/f'api-{key}.json').read_text())
        bits=lambda v:struct.pack('<'+str(len(v))+'f',*v)
        assert bits(r['embedding'])==bits(baseline['embedding']),f'Output changed from goal baseline: {key}'
        r['baseline_instructions']=baseline['stats']['instructions']
        r['instructions_saved']=baseline['stats']['instructions']-r['stats']['instructions']
        (OUT/f'api-{key}.json').write_text(json.dumps(r,indent=2)+'\n')
        results.append({k:r[k] for k in ['case_id','tokens','seconds','cosine_to_f32'] } | {'queries':r['stats']['queries'],'instructions':r['stats']['instructions'],'max_instructions':r['stats']['max_instructions']})
        print(json.dumps(results[-1]),flush=True)
    source=[ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'rust-toolchain.toml',ROOT/'.cargo/config.toml',ROOT/'requirements.txt']
    source+=sorted((ROOT/'crates').rglob('*.rs'))+sorted((ROOT/'crates').rglob('Cargo.toml'))+sorted((ROOT/'scripts').glob('*.py'))
    env={'date':datetime.now(ZoneInfo('Asia/Tokyo')).date().isoformat(),'wasm_sha256':wasm_hash,'wasm_bytes':wasm.stat().st_size,
         'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source},
         'subnet':'Application / Production','query_instruction_limit':5_000_000_000,
         'features':default_features,'profiling':False,'unit_tests':int(test_counts[0]),
         'baseline_bit_exact_cases':len(keys),'rope_lookup_exact_cases':rope_matches,'public_api_cases':len(keys),'ordinary_single_query_cases':4,'typed_vs_raw_single_query_exact':True,
         'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
         'architecture':platform.machine(),'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
         'pocket_ic':subprocess.check_output([os.environ['POCKET_IC_BIN'],'--version'],text=True).strip()}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    stress=json.loads((CACHE/'stress.json').read_text())
    baseline_stress=json.loads((baseline_out/'stress-baseline.json').read_text())
    assert stress['wasm_sha256']==wasm_hash
    assert len(stress['cases'])==len(baseline_stress['cases'])==57
    bits=lambda v:struct.pack('<'+str(len(v))+'f',*v)
    for a,b in zip(stress['cases'],baseline_stress['cases']):
        assert a['ids']==b['ids'] and a['queries']==(1 if a['tokens']<=single_max else 2)
        assert bits(a['embedding'])==bits(b['embedding']),f"Stress output changed: {a['tokens']}/{a['pattern']}"
        assert a['max_instructions']<5_000_000_000
    (OUT/'stress.json').write_text(json.dumps(stress,indent=2)+'\n')
    env['stress_bit_exact_cases']=len(stress['cases'])
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    sweep=json.loads((CACHE/'sweep.json').read_text())
    baseline_sweep=json.loads((baseline_out/'sweep.json').read_text())
    assert sweep['wasm_sha256']==wasm_hash
    assert len(sweep['cases'])==511 and len(baseline_sweep['cases'])==511
    assert [c['tokens'] for c in sweep['cases']]==list(range(2,513))
    reference_by_length={c['tokens']:c for c in baseline_sweep['cases']}
    reference_by_length.update({c['tokens']:c for c in baseline_stress['cases'] if c['pattern']==0})
    exact_sweep_cases=0
    for a in sweep['cases']:
        assert a['queries']==(1 if a['tokens']<=single_max else 2)
        assert a['max_instructions']<5_000_000_000
        if a['tokens'] in reference_by_length:
            b=reference_by_length[a['tokens']]
            assert a['ids']==b['ids']
            assert bits(a['embedding'])==bits(b['embedding']),f"Length sweep changed: {a['tokens']}"
            exact_sweep_cases+=1
    (OUT/'sweep.json').write_text(json.dumps(sweep,indent=2)+'\n')
    env['sweep_bit_exact_cases']=exact_sweep_cases
    env['length_sweep_cases']=len(sweep['cases'])
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    memory=json.loads((CACHE/f'model-memory-{wasm_hash}.json').read_text())
    assert memory['wasm_sha256']==wasm_hash
    (OUT/'model-memory.json').write_text(json.dumps(memory,indent=2)+'\n')
    (OUT/'repeat-single.json').write_bytes((CACHE/'repeat-single.json').read_bytes())
    (OUT/'references.json').write_bytes((CACHE/'references.json').read_bytes())
    print('All ordinary-query and accuracy checks passed.',flush=True)


if __name__=='__main__':main()
