"""Select 50 non-truncated JaQuAD question/paragraph pairs, one per article."""
import argparse, glob, json
from tokenizers import Tokenizer

p=argparse.ArgumentParser()
p.add_argument('--tokenizer',required=True)
p.add_argument('--data-dir',default='.cache/answer-spans')
p.add_argument('--out',default='.cache/bekko-memory-study/retrieval-ja.json')
p.add_argument('--related',action='store_true',help='Select five paragraphs from each of ten articles')
a=p.parse_args(); tok=Tokenizer.from_file(a.tokenizer)
questions=[];docs=[];qids=[];dids=[];titles=set()
for path in sorted(glob.glob(a.data_dir+'/jaquad_dev_*.json')):
    for article in json.load(open(path))['data']:
        title=article['title']
        if title in titles: continue
        selected=[]
        for j,paragraph in enumerate(article['paragraphs']):
            doc=tok.encode(paragraph['context']).ids
            if not 2 <= len(doc) <= 256: continue
            for q in paragraph.get('qas',[]):
                ids=tok.encode(q['question']).ids
                if 2 <= len(ids) <= 256 and q.get('answers'):
                    selected.append((doc,ids,q['id'],f'{title}::{j}')); break
            if len(selected)==(5 if a.related else 1): break
        if len(selected)==(5 if a.related else 1):
            for doc,ids,qid,did in selected:
                docs.append(doc); questions.append(ids); qids.append(qid); dids.append(did)
            titles.add(title)
        if len(docs)>=50: break
    if len(docs)>=50: break
assert len(docs)==50
out=[dict(task='JaQuAD-related' if a.related else 'JaQuAD',language='ja',query_ids=qids,document_ids=dids,qrels={q:[d] for q,d in zip(qids,dids)},parents=list(range(len(docs))),ids=questions+docs)]
open(a.out,'w').write(json.dumps(out,ensure_ascii=False))
