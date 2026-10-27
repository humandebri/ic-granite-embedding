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
    let request_bytes = payload.len();
    let now = Instant::now();
    let response = pic
        .query_call(id, Principal::anonymous(), "step", payload)
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
    stats.request_bytes += request_bytes as u64;
    stats.response_bytes += response.len() as u64;
    stats.max_request_bytes = stats.max_request_bytes.max(request_bytes);
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

fn check_bounds(pic: &pocket_ic::PocketIc, id: Principal, chained: bool) {
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
        (
            [15, 0, 0, 681, 0, 0, 0, 681],
            Vec::new(),
            "initial projection bounds",
        ),
        (
            [15, 1, 0, 1, 0, 0, 0, 1],
            Vec::new(),
            "initial projection bounds",
        ),
        (
            [16, 0, 0, 681, 0, 0, 0, 681],
            Vec::new(),
            "chained finish bounds",
        ),
        (
            [16, 11, 0, 2, 0, 0, 0, 2],
            Vec::new(),
            "chained finish bounds",
        ),
        (
            [16, 11, 1, 2, 0, 0, 0, 1],
            Vec::new(),
            "chained finish bounds",
        ),
        (
            [15, 0, 0, 1, 0, 0, 0, 1],
            50368u32.to_le_bytes().to_vec(),
            "token bounds",
        ),
    ];
    for (fields, body, expected) in invalid {
        if fields[0] >= 15 && !chained {
            continue;
        }
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

pub(super) fn run_split(
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
        check_bounds(pic, id, false);
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

// Preserve the unfused path as a protocol comparison, including raw F32 mode.
#[allow(clippy::too_many_arguments)]
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
    if !packed {
        return run_split(
            pic,
            id,
            ids,
            reference,
            int8_reference,
            cache,
            hash,
            false,
            case_id,
        );
    }
    assert!((2..=5000).contains(&ids.len()));
    assert_eq!(ids.first(), Some(&50281));
    assert_eq!(ids.last(), Some(&50282));
    assert!(!ids.contains(&50283));
    check_bounds(pic, id, true);
    let started = Instant::now();
    let n = ids.len();
    let mut stats = Stats::default();
    let mut x = Vec::<u8>::with_capacity(n * H * 4);
    let mut qkv = granite_core::PackedQkv::with_capacity(n);
    for start in (0..n).step_by(640) {
        let rows = (n - start).min(640);
        let body: Vec<u8> = ids[start..start + rows]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = call_bytes(pic, id, [15, 0, start, n, 0, 0, 0, rows], &body, &mut stats);
        absorb(&bytes, rows, &mut x, &mut qkv);
    }
    let initial = json!({"queries":stats.queries,"instructions":stats.instructions});
    let mut trace = Vec::new();
    let mut output = Vec::new();
    for layer in 0..12 {
        let before = stats.instructions;
        let calls_before = stats.queries;
        let inputs = qkv.rows();
        assert_eq!(x.len(), inputs * H * 4);
        let outputs = hidden_rows(n, layer + 1);
        let quantized = layer % 3 != 0;
        let head_tile = [12, 6, 4, 3, 2, 1]
            .into_iter()
            .find(|h| {
                let size = if quantized {
                    h * (outputs * (2 * D + 4) + inputs * 4 * D)
                } else {
                    4 * D * h * (inputs + outputs)
                };
                size + 256 < 2 * 1024 * 1024
            })
            .unwrap();
        let mut context = vec![0.; outputs * H];
        for head in (0..12).step_by(head_tile) {
            let wire = qkv.attention_wire(0, outputs, head, head_tile, 0, inputs, quantized);
            let result = call(
                pic,
                id,
                [
                    if quantized { 13 } else { 2 },
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
        let attention_instructions = stats.instructions - before;
        let attention_queries = stats.queries - calls_before;
        let finish_before = stats.instructions;
        let finish_calls = stats.queries;
        let mut next = Vec::with_capacity(outputs * H * 4);
        let mut next_qkv =
            granite_core::PackedQkv::with_capacity(if layer < 11 { outputs } else { 0 });
        for start in (0..outputs).step_by(640) {
            let rows = (outputs - start).min(640);
            let body = granite_core::pack_finish_bytes(
                &x[start * H * 4..(start + rows) * H * 4],
                &context[start * H..(start + rows) * H],
            );
            let bytes = call_bytes(
                pic,
                id,
                [16, layer, start, n, 0, 0, 0, rows],
                &body,
                &mut stats,
            );
            if layer == 11 {
                output = decode_floats(&bytes).unwrap();
            } else {
                absorb(&bytes, rows, &mut next, &mut next_qkv);
            }
        }
        trace.push(json!({"layer":layer,"input_rows":inputs,"output_rows":outputs,"queries":stats.queries-calls_before,"instructions":stats.instructions-before,"phases":{"attention":{"queries":attention_queries,"instructions":attention_instructions},"finish_and_next_qkv":{"queries":stats.queries-finish_calls,"instructions":stats.instructions-finish_before}}}));
        println!(
            "chained layer={layer} instructions={} queries={}",
            stats.instructions - before,
            stats.queries - calls_before
        );
        x = next;
        qkv = next_qkv;
    }
    assert_eq!(output.len(), H);
    assert!(output.iter().all(|v| v.is_finite()));
    let cosine = similarity(&output, reference);
    assert!(cosine > 0.999, "FP32 cosine {cosine}");
    let report = json!({"mode":"chained-query","case_id":case_id,"tokens":n,"wasm_sha256":hash,"inference_wall_seconds":started.elapsed().as_secs_f64(),"token_ids_sha256":format!("{:x}",Sha256::digest(ids.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())) ,"packed_qkv":true,"initial":initial,"stats":stats,"trace":trace,"embedding":output,"cosine_to_f32":cosine,"cosine_to_int8_reference":similarity(&output,int8_reference)});
    fs::write(
        cache.join(format!("chained-{case_id}-{hash}.json")),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "chained tokens={n} queries={} instructions={} max={} seconds={} cosine={cosine}",
        stats.queries, stats.instructions, stats.max_instructions, stats.query_seconds
    );
}

fn absorb(bytes: &[u8], rows: usize, hidden: &mut Vec<u8>, qkv: &mut granite_core::PackedQkv) {
    assert_eq!(bytes.len(), rows * H * 12);
    hidden.extend_from_slice(&bytes[..rows * H * 4]);
    qkv.extend(&bytes[rows * H * 4..]).unwrap();
}

// Exercise the published row cap independently of the client's aligned tile.
pub(super) fn check_fused_limits(pic: &pocket_ic::PocketIc, id: Principal) -> serde_json::Value {
    let rows = 680;
    let ids: Vec<u8> = (0..rows).flat_map(|_| 42u32.to_le_bytes()).collect();
    let mut stats = Stats::default();
    let fused = call_bytes(pic, id, [15, 0, 0, rows, 0, 0, 0, rows], &ids, &mut stats);
    let hidden = call_bytes(pic, id, [0, 0, 0, rows, 0, 0, 0, rows], &ids, &mut stats);
    let qkv = call_bytes(
        pic,
        id,
        [11, 0, 0, rows, 0, 0, 0, rows],
        &hidden,
        &mut stats,
    );
    assert_eq!(&fused[..rows * H * 4], &hidden);
    assert_eq!(&fused[rows * H * 4..], &qkv);
    let body = granite_core::pack_finish_bytes(&hidden, &vec![0.; rows * H]);
    let fused = call_bytes(pic, id, [16, 0, 0, rows, 0, 0, 0, rows], &body, &mut stats);
    let hidden = call_bytes(pic, id, [12, 0, 0, rows, 0, 0, 0, rows], &body, &mut stats);
    let qkv = call_bytes(
        pic,
        id,
        [11, 1, 0, rows, 0, 0, 0, rows],
        &hidden,
        &mut stats,
    );
    assert_eq!(&fused[..rows * H * 4], &hidden);
    assert_eq!(&fused[rows * H * 4..], &qkv);
    json!({"exact_cases":2,"rows":rows,"stats":stats})
}
