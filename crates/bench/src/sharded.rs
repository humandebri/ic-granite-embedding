//! Client-owned hidden state; bounded ordinary queries only.
use super::*;
use granite_core::{D, hidden_rows};

fn call_bytes(
    pic: &pocket_ic::PocketIc,
    id: Principal,
    fields: [usize; 8],
    body: &[u8],
    stats: &mut Stats,
) -> Vec<u8> {
    let payload = Encode!(&serde_bytes::ByteBuf::from(request(fields, body))).unwrap();
    assert!(payload.len() < 2 * 1024 * 1024);
    let now = Instant::now();
    let response = pic
        .query_call(id, Principal::anonymous(), "step", payload.clone())
        .unwrap();
    stats.query_seconds += now.elapsed().as_secs_f64();
    assert!(response.len() < 3 * 1024 * 1024);
    let m = Decode!(&response,Result<Measurement,String>)
        .unwrap()
        .unwrap();
    assert!(m.instructions < 5_000_000_000);
    stats.queries += 1;
    stats.instructions += m.instructions;
    stats.max_instructions = stats.max_instructions.max(m.instructions);
    stats.request_bytes += payload.len() as u64;
    stats.response_bytes += response.len() as u64;
    stats.max_request_bytes = stats.max_request_bytes.max(payload.len());
    stats.max_response_bytes = stats.max_response_bytes.max(response.len());
    m.data
}

fn call(
    pic: &pocket_ic::PocketIc,
    id: Principal,
    fields: [usize; 8],
    body: &[u8],
    stats: &mut Stats,
) -> Vec<f32> {
    decode_floats(&call_bytes(pic, id, fields, body, stats)).unwrap()
}

fn check_bounds(pic: &pocket_ic::PocketIc, id: Principal) {
    let invalid = [
        (
            [11, 0, 0, 1024, 0, 0, 0, 1024],
            Vec::new(),
            "packed qkv rows",
        ),
        (
            [12, 0, 0, 1089, 0, 0, 0, 1089],
            Vec::new(),
            "packed finish length",
        ),
        (
            [13, 0, 0, 1, 0, 1, 0, 1],
            Vec::new(),
            "packed local attention heads",
        ),
        (
            [13, 1, 0, 1, u32::MAX as usize, 2, 0, 1],
            Vec::new(),
            "packed local attention heads",
        ),
        (
            [13, 1, 0, 1, 0, 1, 0, 1],
            Vec::new(),
            "packed attention length",
        ),
        (
            [1, 0, u32::MAX as usize, 2, 0, 0, 0, 2],
            Vec::new(),
            "shape bounds",
        ),
    ];
    for (fields, body, expected) in invalid {
        let response = pic
            .query_call(
                id,
                Principal::anonymous(),
                "step",
                Encode!(&serde_bytes::ByteBuf::from(request(fields, &body))).unwrap(),
            )
            .unwrap();
        let error = Decode!(&response,Result<Measurement,String>)
            .unwrap()
            .err()
            .expect("invalid input was accepted");
        assert_eq!(error, expected);
    }
    for bad in 0..3 {
        let mut body = vec![0u8; 196];
        body[64..68].copy_from_slice(&1f32.to_le_bytes());
        let expected = if bad == 2 {
            body[68..70].copy_from_slice(&0x7fc0u16.to_le_bytes());
            "packed attention nonfinite"
        } else {
            if bad == 0 {
                body[..2].copy_from_slice(&8192i16.to_le_bytes());
            } else {
                body[64..68].fill(0);
            }
            "packed attention quantization bounds"
        };
        let response = pic
            .query_call(
                id,
                Principal::anonymous(),
                "step",
                Encode!(&serde_bytes::ByteBuf::from(request(
                    [13, 1, 0, 1, 0, 1, 0, 1],
                    &body
                )))
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Decode!(&response,Result<Measurement,String>)
                .unwrap()
                .err()
                .unwrap(),
            expected
        );
    }
}

pub(super) fn run(
    pic: &pocket_ic::PocketIc,
    id: Principal,
    ids: &[u32],
    reference: &[f32],
    int8_reference: &[f32],
    cache: &std::path::Path,
    hash: &str,
    packed: bool,
    case_id: &str,
) {
    assert!((2..=5000).contains(&ids.len()));
    if packed {
        check_bounds(pic, id);
    }
    let inference_started = Instant::now();
    let n = ids.len();
    let mut stats = Stats::default();
    let mut trace = Vec::new();
    let mut x = Vec::with_capacity(n * H);
    for chunk in ids.chunks(1984) {
        let body: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        x.extend(call(
            pic,
            id,
            [0, 0, 0, chunk.len(), 0, 0, 0, chunk.len()],
            &body,
            &mut stats,
        ));
    }
    for layer in 0..12 {
        let before = stats.instructions;
        let calls_before = stats.queries;
        let inputs = x.len() / H;
        let outputs = hidden_rows(n, layer + 1);
        let mut qkv = Vec::with_capacity(inputs * 3 * H);
        let qkv_tile = if packed { 1000 } else { 640 };
        for start in (0..inputs).step_by(qkv_tile) {
            let rows = (inputs - start).min(qkv_tile);
            let bytes = call_bytes(
                pic,
                id,
                [if packed { 11 } else { 1 }, layer, start, n, 0, 0, 0, rows],
                &float_bytes(&x[start * H..(start + rows) * H]),
                &mut stats,
            );
            qkv.extend(if packed {
                granite_core::unpack_qkv(&bytes).unwrap()
            } else {
                decode_floats(&bytes).unwrap()
            });
        }
        let qkv_instructions = stats.instructions - before;
        let qkv_queries = stats.queries - calls_before;
        let attn_before = stats.instructions;
        let attn_calls_before = stats.queries;
        let mut context = vec![0.; outputs * H];
        // Full key range preserves the per-channel V quantization scale even
        // when Q rows or heads are split. Last local layers need only a prefix.
        let packed_attention = packed && layer % 3 != 0;
        let head_tile = [12, 6, 4, 3, 2, 1]
            .into_iter()
            .find(|h| {
                if packed_attention {
                    h * (outputs * (2 * D + 4) + inputs * 4 * D) + 128 < 2 * 1024 * 1024
                } else {
                    4 * D * h * (inputs + outputs) + 128 < 2 * 1024 * 1024
                }
            })
            .unwrap();
        for head in (0..12).step_by(head_tile) {
            let wire = if packed_attention {
                granite_core::packed_attention_wire(&qkv, 0, outputs, head, head_tile, 0, inputs)
            } else {
                granite_core::attention_wire(&qkv, 0, outputs, head, head_tile, 0, inputs)
            };
            let result = call(
                pic,
                id,
                [
                    if packed_attention { 13 } else { 2 },
                    layer,
                    0,
                    inputs,
                    head,
                    head_tile,
                    0,
                    outputs,
                ],
                &wire,
                &mut stats,
            );
            assert_eq!(result.len(), outputs * head_tile * D);
            for t in 0..outputs {
                context[t * H + head * D..t * H + (head + head_tile) * D]
                    .copy_from_slice(&result[t * head_tile * D..(t + 1) * head_tile * D]);
            }
        }
        let attention_instructions = stats.instructions - attn_before;
        let attention_queries = stats.queries - attn_calls_before;
        let finish_before = stats.instructions;
        let finish_calls_before = stats.queries;
        let mut next = Vec::with_capacity(outputs * H);
        let finish_tile = if packed { 1088 } else { 640 };
        for start in (0..outputs).step_by(finish_tile) {
            let rows = (outputs - start).min(finish_tile);
            let residual = &x[start * H..(start + rows) * H];
            let ctx = &context[start * H..(start + rows) * H];
            let body = if packed {
                granite_core::pack_finish(residual, ctx)
            } else {
                let mut b = float_bytes(residual);
                b.extend(float_bytes(ctx));
                b
            };
            next.extend(call(
                pic,
                id,
                [if packed { 12 } else { 3 }, layer, start, n, 0, 0, 0, rows],
                &body,
                &mut stats,
            ));
        }
        x = next;
        trace.push(json!({"layer":layer,"input_rows":inputs,"output_rows":outputs,"queries":stats.queries-calls_before,"instructions":stats.instructions-before,"phases":{"qkv":{"instructions":qkv_instructions,"queries":qkv_queries},"attention":{"instructions":attention_instructions,"queries":attention_queries},"finish":{"instructions":stats.instructions-finish_before,"queries":stats.queries-finish_calls_before}}}));
        println!(
            "sharded layer={layer} instructions={} queries={}",
            stats.instructions - before,
            stats.queries - calls_before
        );
    }
    let output = call(
        pic,
        id,
        [4, 11, 0, n, 0, 0, 0, 1],
        &float_bytes(&x[..H]),
        &mut stats,
    );
    assert_eq!(output.len(), H);
    assert!(output.iter().all(|x| x.is_finite()));
    let cosine = similarity(&output, reference);
    assert!(cosine > 0.999, "FP32 cosine {cosine}");
    let report = json!({"mode":"sharded-query","inference_wall_seconds":inference_started.elapsed().as_secs_f64(),"tokens":n,"case_id":case_id,"token_ids_sha256":format!("{:x}",Sha256::digest(ids.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())) ,"packed_qkv":packed,"wasm_sha256":hash,"stats":stats,"trace":trace,"embedding":output,"cosine_to_f32":cosine,"cosine_to_int8_reference":similarity(&output,int8_reference)});
    fs::write(
        cache.join(format!(
            "sharded-{case_id}-{}-{hash}.json",
            if packed { "packed" } else { "raw" }
        )),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "sharded tokens={n} queries={} instructions={} max={} seconds={} cosine={cosine}",
        stats.queries, stats.instructions, stats.max_instructions, stats.query_seconds
    );
}
