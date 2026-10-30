import json, sys
import numpy as np
hard='--ja-hard' in sys.argv
datasets=json.load(open('.cache/bekko-memory-study/retrieval-ja-hard.json' if hard else '.cache/bekko25/retrieval.json'))
items=json.load(open('.cache/bekko-memory-study/retrieval-ja-hard-vectors.json' if hard else '.cache/bekko-memory-study/retrieval-vectors.json'))['results']
out=[]
for lang in (('ja',) if hard else ('en','ar')):
    data=next(x for x in datasets if x['language']==lang and (hard or x['task']=='NanoSciFact'))
    vectors={r['mode']:np.array(r['vectors'],dtype=np.float64) for r in items if r['language']==lang}
    for x in vectors.values(): x /= np.linalg.norm(x,axis=1,keepdims=True)
    nq=len(data['query_ids']);nd=len(data['document_ids']);parents=data['parents']
    def metrics(q,docs):
        sim=q[:nq] @ docs[nq:].T; scores=np.full((nq,nd),-np.inf)
        for i,p in enumerate(parents): scores[:,p]=np.maximum(scores[:,p],sim[:,i])
        ranks=np.argsort(-scores,axis=1,kind='stable')
        recall=[sum(data['document_ids'][j] in data['qrels'][qid] for j in ranks[i,:5])/len(data['qrels'][qid]) for i,qid in enumerate(data['query_ids'])]
        return np.array(recall),ranks
    ref,r0=metrics(vectors['original'],vectors['original'])
    for mode in ('fp16','bf16','int8'):
        score,ranks=metrics(vectors[mode],vectors['original'])
        out.append({'language':lang,'query':mode,'index':'original','recall_at_5':float(score.mean()),'worsened_queries':int((score<ref).sum()),'improved_queries':int((score>ref).sum()),'top5_set_changed':sum(set(a[:5])!=set(b[:5]) for a,b in zip(ranks,r0))})
print(json.dumps(out,indent=2))
