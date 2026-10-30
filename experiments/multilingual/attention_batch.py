"""Generate independent four-row AV accumulators with shared V loads."""


def generate():
    s = """
#[cfg(target_arch="wasm32")]
fn attention_batch(layer:usize,start:usize,rows:usize,heads:usize,ks:usize,kn:usize,q:&[f32],k:&[f32],v:&[f32])->Vec<f32>{
    use core::arch::wasm32::*;
    let width=heads*D;
    let (quant_keys,key_scales)={
        let mut quant=vec![0i16;k.len()];let mut scales=vec![0f32;kn*heads];
        for j in 0..kn {for h in 0..heads {let at=j*width+h*D;scales[j*heads+h]=quant13(&k[at..at+D],&mut quant[at..at+D]);}}
        (quant,scales)
    };
    let mut out=vec![0f32;rows*width];
    for t in (0..rows).step_by(4){let count=(rows-t).min(4);
        for h in 0..heads {
            let mut scores=vec![0f32;4*kn];let mut inverse=[0f32;4];
            let mut first=[0usize;4];let mut last=[0usize;4];
            for a in 0..count {
                (first[a],last[a])=if layer%3==0 {(0,kn)}else{((start+t+a).saturating_sub(64).saturating_sub(ks),(start+t+a+65-ks).min(kn))};
                let query=&q[(t+a)*width+h*D..(t+a)*width+(h+1)*D];
                let active=&mut scores[a*kn+first[a]..a*kn+last[a]];
                let max=score13(query,&quant_keys,&key_scales,first[a]*width+h*D,width,first[a]*heads+h,heads,active);
                inverse[a]=1./softmax_row(active,max);
            }
            unsafe {
"""
    for a in range(4):
        for d in range(16):
            s += f"let mut s{a}_{d}=f32x4_splat(0.);\n"
    s += "for j in first[0]..last[count-1] {let src=v.as_ptr().add(j*width+h*D);\n"
    for d in range(16):
        s += f"let v{d}=v128_load(src.add({4 * d}).cast());\n"
    for a in range(4):
        s += f"if {a}<count && j>=first[{a}] && j<last[{a}] {{let prob=f32x4_splat(scores[{a}*kn+j]*inverse[{a}]);\n"
        for d in range(16):
            s += f"s{a}_{d}=f32x4_add(s{a}_{d},f32x4_mul(prob,v{d}));\n"
        s += "}\n"
    s += "}\n"
    for a in range(4):
        s += f"if {a}<count {{let dst=out.as_mut_ptr().add((t+{a})*width+h*D);\n"
        for d in range(16):
            s += f"v128_store(dst.add({4 * d}).cast(),s{a}_{d});\n"
        s += "}\n"
    s += "}}}out}\n"
    return s
