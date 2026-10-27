"""Aggregate non-overlapping metered stages from a profiling-only Wasm run."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    total=report['stats']['instructions']
    aggregate=defaultdict(int)
    layers=defaultdict(lambda:defaultdict(int))
    projections=defaultdict(lambda:defaultdict(int))
    endings={'/quantize':'linear.quantize','/gemm_and_scale':'linear.gemm_and_scale','/gelu':'gelu',
             '/qk_scores':'attention.qk','/softmax':'attention.softmax','/av_weighted_sum':'attention.av',
             '/wire':'attention.wire','/rope':'rope','/prepare_attention':'attention.prepare',
             '/quantize_qk':'attention.quantize_qk','/quantize_q':'attention.quantize_q',
             '/quantize_k':'attention.quantize_k','/transpose_quantize_v':'attention.quantize_v'}
    for query in report['trace']:
        for name,instructions,calls in query.get('phases',[]):
            for ending,key in endings.items():
                if name.endswith(ending):
                    aggregate[key]+=instructions
                    layers[name.split('/')[0]][key]+=instructions
                    if key.startswith('linear.'):
                        projection=next((p for p in ['out_projection','mlp_wi','mlp_wo','projection'] if f'/{p}/' in name),'unknown')
                        projections[projection][key]+=instructions
                    break
    aggregate['other_including_profile_overhead']=total-sum(aggregate.values())
    summary={'wasm_sha256':report['wasm_sha256'],'instructions':total,
             'stages':{k:{'instructions':v,'percent':100*v/total} for k,v in sorted(aggregate.items(),key=lambda x:-x[1])},
             'layers':dict(layers),'projections':dict(projections)}
    if args.output:args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
