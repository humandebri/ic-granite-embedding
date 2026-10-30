import hashlib, json, struct, time, zlib
import numpy as np

path = '.cache/bekko25/model.pack'
sha = hashlib.sha256()
with open(path,'rb') as f:
    for b in iter(lambda: f.read(4*1024*1024), b''): sha.update(b)
assert sha.hexdigest() == '8fc88d56d68f550d1a920bab7d1ec42af9449240048393bd6daabbfcfa861724'
with open(path,'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); d=h['embeddings.tok_embeddings.weight']; f.seek(8+n+d['offset'])
    remaining=d['bytes']; packed=0; exact_bf=exact_fp=total=0; maxerr=0.; chunks=0; start=time.perf_counter()
    while remaining:
        raw=f.read(min(remaining,4*1024*1024)); v=np.frombuffer(raw,dtype='<u4'); x=v.view('<f4')
        fp=x.astype(np.float16).astype(np.float32); exact_fp+=int(np.count_nonzero(x==fp)); exact_bf+=int(np.count_nonzero((v&0xffff)==0)); maxerr=max(maxerr,float(np.max(np.abs(x-fp)))); total+=len(x)
        shuffled=np.frombuffer(raw,dtype=np.uint8).reshape(-1,4).T.copy().tobytes(); comp=zlib.compress(shuffled,6)
        restored=np.frombuffer(zlib.decompress(comp),dtype=np.uint8).reshape(4,-1).T.copy().tobytes(); assert raw==restored
        packed+=len(comp); remaining-=len(raw); chunks+=1
print(json.dumps({'pack_sha256':sha.hexdigest(),'embedding_bytes':d['bytes'],'value_count':total,'bf16_exact_count':exact_bf,'fp16_exact_count':exact_fp,'fp16_max_abs_error':maxerr,'block_bytes':4*1024*1024,'zlib_level':6,'byte_shuffle':True,'compressed_table_bytes_excluding_framing':packed,'chunks':chunks,'roundtrip_exact':True,'seconds':round(time.perf_counter()-start,3)},indent=2))
