"""Bounded P13/V11 integer AV experiment. QK13 precision is unchanged."""


def generate():
    s = r"""
#[cfg(target_arch="wasm32")]
fn attention_integer(layer:usize,start:usize,rows:usize,heads:usize,ks:usize,kn:usize,q:&[f32],k:&[f32],v:&[f32])->Vec<f32>{
 use core::arch::wasm32::*;
 let width=heads*D;let stride=kn.next_multiple_of(8);
 let mut keys=vec![0i16;k.len()];let mut key_scales=vec![0f32;kn*heads];
 for j in 0..kn {for h in 0..heads {let at=j*width+h*D;key_scales[j*heads+h]=quant13(&k[at..at+D],&mut keys[at..at+D]);}}
 let mut values=vec![0i16;width*stride];let mut value_scales=vec![0f32;width];let mut column=vec![0f32;stride];
 for d in 0..width {
   for j in 0..kn {column[j]=v[j*width+d];}
   let scale=(max_abs(&column)/2047.).max(1e-12);value_scales[d]=scale;
   unsafe {for j in (0..stride).step_by(8) {
     let a=i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_div(v128_load(column.as_ptr().add(j).cast()),f32x4_splat(scale))));
     let b=i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_div(v128_load(column.as_ptr().add(j+4).cast()),f32x4_splat(scale))));
     v128_store(values.as_mut_ptr().add(d*stride+j).cast(),i16x8_narrow_i32x4(a,b));
   }}
 }
 let mut out=vec![0f32;rows*width];let mut scores=vec![0f32;stride];let mut probs=vec![0i16;stride];
 for t in 0..rows {for h in 0..heads {
   let (first,last)=if layer%3==0 {(0,kn)}else{((start+t).saturating_sub(64).saturating_sub(ks),(start+t+65-ks).min(kn))};
   let query=&q[t*width+h*D..t*width+(h+1)*D];let active=&mut scores[first..last];
   let max=score13(query,&keys,&key_scales,first*width+h*D,width,first*heads+h,heads,active);
   let inverse=1./softmax_row(active,max);probs.fill(0);
   for j in first..last {probs[j]=(scores[j]*8191.).round_ties_even().clamp(0.,8191.)as i16;}
   unsafe {integer_av(probs.as_ptr(),values.as_ptr().add(h*D*stride),stride,value_scales.as_ptr().add(h*D),inverse/8191.,out.as_mut_ptr().add(t*width+h*D));}
 }}out
}
#[cfg(target_arch="wasm32")]
unsafe fn integer_av(p:*const i16,v:*const i16,stride:usize,scales:*const f32,inverse:f32,out:*mut f32){
 use core::arch::wasm32::*;
 unsafe {
 for block in (0..64).step_by(16) {
"""
    for d in range(16):
        s += f"let mut s{d}=i32x4_splat(0);let v{d}=v.add((block+{d})*stride);\n"
    s += "for j in (0..stride).step_by(8) {let prob=v128_load(p.add(j).cast());\n"
    for d in range(16):
        s += f"s{d}=i32x4_add(s{d},i32x4_dot_i16x8(prob,v128_load(v{d}.add(j).cast())));\n"
    s += "}\n"
    for d in range(16):
        s += f"let total=i32x4_extract_lane::<0>(s{d})as i64+i32x4_extract_lane::<1>(s{d})as i64+i32x4_extract_lane::<2>(s{d})as i64+i32x4_extract_lane::<3>(s{d})as i64;*out.add(block+{d})=total as f32*(*scales.add(block+{d})*inverse);\n"
    s += "}}}\n"
    return s
