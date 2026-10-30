import json, sys
import numpy as np

hard = '--ja-hard' in sys.argv
japanese = hard or '--ja' in sys.argv
datasets = json.load(open('.cache/bekko-memory-study/retrieval-ja-hard.json' if hard else '.cache/bekko-memory-study/retrieval-ja.json' if japanese else '.cache/bekko25/retrieval.json'))
all_results = json.load(open('.cache/bekko-memory-study/retrieval-ja-hard-vectors.json' if hard else '.cache/bekko-memory-study/retrieval-ja-vectors.json' if japanese else '.cache/bekko-memory-study/retrieval-vectors.json'))
out = []
for lang in (('ja',) if japanese else ('en', 'ar')):
    data = next(x for x in datasets if x['language'] == lang and (japanese or x['task'] == 'NanoSciFact'))
    results = {x['mode']: x for x in all_results['results'] if x['language'] == lang}
    nq, nd = len(data['query_ids']), len(data['document_ids'])
    parents = np.array(data['parents'], dtype=int)
    rankings, recalls = {}, {}
    original = np.array(results['original']['vectors'], dtype=np.float64)
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    for mode, result in results.items():
        x = np.array(result['vectors'], dtype=np.float64)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        sims = x[:nq] @ x[nq:].T
        scores = np.full((nq, nd), -np.inf)
        for idx, parent in enumerate(parents):
            scores[:, parent] = np.maximum(scores[:, parent], sims[:, idx])
        rank = np.argsort(-scores, axis=1, kind='stable')
        recall = np.array([sum(data['document_ids'][j] in data['qrels'][qid] for j in rank[i,:5]) / len(data['qrels'][qid]) for i,qid in enumerate(data['query_ids'])])
        rankings[mode], recalls[mode] = rank, recall
        cosine = (x * original).sum(axis=1)
        top1_hits=sum(data['document_ids'][rank[i,0]] in data['qrels'][qid] for i,qid in enumerate(data['query_ids']))
        out.append({'language':lang,'mode':mode,'queries':nq,'chunks':len(parents),'recall_at_5':float(recall.mean()),'top1_hits':int(top1_hits),'cosine_min':float(cosine.min()),'cosine_mean':float(cosine.mean()),'vectors_count':len(x),'seconds':result['seconds']})
    for row in out:
        if row['language'] == lang and row['mode'] != 'original':
            mode=row['mode']; row['top1_changed'] = int(np.count_nonzero(rankings[mode][:,0] != rankings['original'][:,0])); row['top5_set_changed'] = sum(set(a[:5]) != set(b[:5]) for a,b in zip(rankings[mode],rankings['original'])); row['recall_improved_queries'] = int(np.count_nonzero(recalls[mode] > recalls['original'])); row['recall_worsened_queries'] = int(np.count_nonzero(recalls[mode] < recalls['original']))
print(json.dumps(out,indent=2))
