use std::{collections::BTreeSet,fs,time::Instant};
use bekko25_core::Model;
use serde_json::{Value,json};
fn main(){
 let hard=std::env::args().any(|x|x=="--ja-hard");
 let japanese=hard || std::env::args().any(|x|x=="--ja");
 let file=if hard {".cache/bekko-memory-study/retrieval-ja-hard.json"} else if japanese {".cache/bekko-memory-study/retrieval-ja.json"} else {".cache/bekko25/retrieval.json"};
 let data:Vec<Value>=serde_json::from_slice(&fs::read(file).unwrap()).unwrap();
 let subset:Vec<_>=data.iter().filter(|x|if japanese {x["language"]=="ja"} else {x["task"]=="NanoSciFact" && (x["language"]=="en" || x["language"]=="ar")}).collect();
 let inputs:Vec<Vec<Vec<u32>>>=subset.iter().map(|d|d["ids"].as_array().unwrap().iter().map(|row|row.as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u32).collect()).collect()).collect();
 let used:BTreeSet<u32>=inputs.iter().flatten().flatten().copied().collect();
 let mut pack=fs::read(".cache/bekko25/model.pack").unwrap();
 let n=u64::from_le_bytes(pack[..8].try_into().unwrap()) as usize;
 let desc:Value=serde_json::from_slice(&pack[8..8+n]).unwrap();
 let offset=8+n+desc["embeddings.tok_embeddings.weight"]["offset"].as_u64().unwrap() as usize;
 let mut results=Vec::new();
 for mode in ["original","fp16","bf16","int8"] {
  if mode!="original" {
   pack=fs::read(".cache/bekko25/model.pack").unwrap();
   for id in &used {
    let start=offset+*id as usize*384*4;
    let row:Vec<f32>=(0..384).map(|i|f32::from_le_bytes(pack[start+i*4..start+i*4+4].try_into().unwrap())).collect();
    let scale=row.iter().map(|x|x.abs()).fold(0f32,f32::max).max(1e-12)/127.;
    for (i,x) in row.iter().enumerate() {
     let y=match mode {"fp16"=>half::f16::from_f32(*x).to_f32(),"bf16"=>half::bf16::from_f32(*x).to_f32(),_=>((*x/scale).round().clamp(-127.,127.))*scale};
     pack[start+i*4..start+i*4+4].copy_from_slice(&y.to_le_bytes());
    }
   }
  }
  let model=Model::load(&pack).unwrap();
  for (di,ds) in inputs.iter().enumerate() {
   let start=Instant::now();
   let vectors=std::thread::scope(|scope| {
    let m=&model;
    let handles:Vec<_>=ds.chunks(ds.len().div_ceil(4)).map(|chunk|scope.spawn(move || chunk.iter().map(|ids|m.full(ids).unwrap()).collect::<Vec<_>>())).collect();
    handles.into_iter().flat_map(|h|h.join().unwrap()).collect::<Vec<_>>()
   });
   eprintln!("{mode} {} {}/{} {:.1}s",subset[di]["language"],ds.len(),ds.len(),start.elapsed().as_secs_f64());
   results.push(json!({"mode":mode,"task":subset[di]["task"],"language":subset[di]["language"],"vectors":vectors,"seconds":start.elapsed().as_secs_f64()}));
  }
 }
 println!("{}",serde_json::to_string(&json!({"results":results,"used_token_rows":used.len(),"note":"native f32-resident inference; quantized token rows expanded before inference; not Wasm"})).unwrap());
}
