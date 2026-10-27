mod sharded;
use candid::{CandidType, Decode, Deserialize, Encode, Principal};
use granite_core::{D, H, Model, decode_floats, float_bytes, request};
use pocket_ic::PocketIcBuilder;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

#[derive(CandidType, Deserialize)]
struct Measurement {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    instructions: u64,
}
#[derive(CandidType, Deserialize)]
struct ProfileMeasurement {
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
    instructions: u64,
    phases: Vec<(String, u64, u64)>,
}
#[derive(CandidType, Deserialize)]
struct EmbeddingStep {
    next_layer: u32,
    tokens: u32,
    state: serde_bytes::ByteBuf,
    embedding: Vec<f32>,
    instructions: u64,
}
#[derive(Default, Serialize, Deserialize)]
struct Stats {
    queries: usize,
    instructions: u64,
    max_instructions: u64,
    request_bytes: u64,
    response_bytes: u64,
    max_request_bytes: usize,
    max_response_bytes: usize,
    query_seconds: f64,
}
fn similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn run_api(
    pic: &pocket_ic::PocketIc,
    id: Principal,
    ids: &[u32],
    reference: &[f32],
    int8_reference: &[f32],
    cache: &std::path::Path,
    key: &str,
    wasm_hash: &str,
) {
    let rejected = pic
        .query_call(
            id,
            Principal::anonymous(),
            "startEmbedding",
            Encode!(&vec![50281u32; granite_core::DEFAULT_MAX_TOKENS + 1]).unwrap(),
        )
        .unwrap();
    assert!(
        Decode!(&rejected,Result<EmbeddingStep,String>)
            .unwrap()
            .is_err(),
        "oversize input must be rejected"
    );
    let rejected = pic
        .query_call(
            id,
            Principal::anonymous(),
            "continueEmbedding",
            Encode!(&512u32, &1u32, &serde_bytes::ByteBuf::new()).unwrap(),
        )
        .unwrap();
    assert!(
        Decode!(&rejected,Result<EmbeddingStep,String>)
            .unwrap()
            .is_err(),
        "invalid progress must be rejected"
    );
    let mut nan = b"GRC1".to_vec();
    nan.extend(0u32.to_le_bytes());
    nan.extend(((512 * H) as u32).to_le_bytes());
    nan.extend(float_bytes(&vec![f32::NAN; 512 * H]));
    let nan = serde_bytes::ByteBuf::from(nan);
    let rejected = pic
        .query_call(
            id,
            Principal::anonymous(),
            "continueEmbedding",
            Encode!(&512u32, &5u32, &nan).unwrap(),
        )
        .unwrap();
    let error = Decode!(&rejected,Result<EmbeddingStep,String>)
        .unwrap()
        .err()
        .unwrap();
    assert!(
        error.contains("nonfinite"),
        "unexpected NaN rejection: {error}"
    );
    let before = Instant::now();
    let mut stats = Stats::default();
    let mut trace = Vec::new();
    let mut next_layer = 0u32;
    let mut state = serde_bytes::ByteBuf::new();
    let output = loop {
        let method = if next_layer == 0 {
            "startEmbedding"
        } else {
            "continueEmbedding"
        };
        let payload = if next_layer == 0 {
            Encode!(&ids.to_vec()).unwrap()
        } else {
            Encode!(&(ids.len() as u32), &next_layer, &state).unwrap()
        };
        assert!(payload.len() < 2 * 1024 * 1024);
        let request_bytes = payload.len();
        let now = Instant::now();
        let response = pic
            .query_call(id, Principal::anonymous(), method, payload)
            .unwrap();
        let elapsed = now.elapsed().as_secs_f64();
        let result = Decode!(&response,Result<EmbeddingStep,String>)
            .unwrap()
            .unwrap();
        assert_eq!(result.tokens as usize, ids.len());
        assert!(result.next_layer > next_layer && result.next_layer <= 12);
        assert!(result.instructions < 5_000_000_000);
        assert!(response.len() < 3 * 1024 * 1024);
        stats.queries += 1;
        stats.query_seconds += elapsed;
        stats.instructions += result.instructions;
        stats.max_instructions = stats.max_instructions.max(result.instructions);
        stats.request_bytes += request_bytes as u64;
        stats.response_bytes += response.len() as u64;
        stats.max_request_bytes = stats.max_request_bytes.max(request_bytes);
        stats.max_response_bytes = stats.max_response_bytes.max(response.len());
        trace.push(json!({"method":method,"first_layer":next_layer,"next_layer":result.next_layer,"instructions":result.instructions,"seconds":elapsed}));
        println!(
            "api {key} layer={next_layer} next={} instructions={}",
            result.next_layer, result.instructions
        );
        next_layer = result.next_layer;
        if next_layer == 12 {
            assert!(result.state.is_empty());
            break result.embedding;
        }
        assert!(result.embedding.is_empty());
        let expected = granite_core::hidden_rows(ids.len(), next_layer as usize) * H;
        assert!(result.state.len() >= 12 && result.state.len() <= 12 + expected * 4);
        assert_eq!(&result.state[..4], b"GRC1");
        assert_eq!(
            u32::from_le_bytes(result.state[8..12].try_into().unwrap()) as usize,
            expected
        );
        state = result.state;
    };
    assert_eq!(output.len(), H);
    assert!(output.iter().all(|v| v.is_finite()));
    assert!((output.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-5);
    let plan = granite_core::query_plan(ids.len()).unwrap();
    assert_eq!(stats.queries, plan.len());
    let report = json!({"mode":"pocketic-production-public-query-api","case_id":key,"tokens":ids.len(),"plan":plan,
        "layers":12,"seconds":before.elapsed().as_secs_f64(),"stats":stats,"trace":trace,"wasm_sha256":wasm_hash,
        "validation_checks":3,"cosine_to_f32":similarity(&output,reference),"cosine_to_int8_reference":similarity(&output,int8_reference),"embedding":output});
    fs::write(
        cache.join(format!("api-{key}.json")),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("REPORT {report}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("benchmark_pid={}", std::process::id());
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!(
            "Usage: granite-bench [TOKENS] [OPTIONS]\nModes: --sharded, --sharded-suite, --native, --validate-regression, --validate-long, --stress, --sweep, --stress-long\nDiagnostics: --probe, --probe-single, --micro, --micro-av, --micro-qk, --micro-attention, --micro-gelu\nOptions: --case=NAME, --micro-extended, --micro-repeat=N, --repeat-single, --repeat-count=N\nCapacity: --capacity, --capacity-range=FIRST:LAST, --capacity-patterns=N"
        );
        return;
    }
    for (i, arg) in args.iter().enumerate().skip(1) {
        if i == 1 && !arg.starts_with("--") {
            continue;
        }
        let known = [
            "--native",
            "--sharded",
            "--sharded-raw",
            "--sharded-suite",
            "--validate-regression",
            "--validate-long",
            "--stress",
            "--sweep",
            "--stress-long",
            "--probe",
            "--probe-single",
            "--micro",
            "--micro-av",
            "--micro-qk",
            "--micro-attention",
            "--micro-gelu",
            "--micro-extended",
            "--repeat-single",
            "--capacity",
        ]
        .contains(&arg.as_str())
            || [
                "--case=",
                "--micro-repeat=",
                "--repeat-count=",
                "--capacity-range=",
                "--capacity-patterns=",
            ]
            .iter()
            .any(|p| arg.starts_with(p));
        assert!(known, "unknown option {arg}; use --help");
    }
    let length: usize = args
        .get(1)
        .filter(|s| !s.starts_with("--"))
        .map(|s| s.parse().expect("TOKENS must be an integer"))
        .unwrap_or(256);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cache = root.join(".cache");
    let key = args
        .iter()
        .find_map(|s| s.strip_prefix("--case="))
        .map(|name| format!("{name}-{length}"))
        .unwrap_or_else(|| length.to_string());
    let ids: Vec<u32> =
        serde_json::from_slice(&fs::read(cache.join(format!("tokens-{key}.json"))).unwrap())
            .unwrap();
    assert_eq!(ids.len(), length);
    let reference =
        decode_floats(&fs::read(cache.join(format!("reference-{key}.f32"))).unwrap()).unwrap();
    let int8_reference =
        decode_floats(&fs::read(cache.join(format!("reference-int8-{key}.f32"))).unwrap()).unwrap();
    let pack = fs::read(cache.join("model.pack")).unwrap();
    if args.iter().any(|s| s == "--native") {
        let model = Model::load(&pack).unwrap();
        let before = Instant::now();
        let output = model.full(&ids).unwrap();
        let report = json!({"mode":"native-w8a8","tokens":length,"seconds":before.elapsed().as_secs_f64(),"cosine_to_f32":similarity(&output,&reference),"cosine_to_int8_reference":similarity(&output,&int8_reference),"embedding":output});
        fs::write(
            cache.join(format!("native-{length}.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{report}");
        return;
    }
    // Production instruction configuration, never the benchmarking subnet variant.
    let pic = PocketIcBuilder::new()
        .with_application_subnet()
        .with_server_binary(PathBuf::from(
            std::env::var("POCKET_IC_BIN").expect("set POCKET_IC_BIN to PocketIC server 11.0.0"),
        ))
        .with_max_request_time_ms(Some(300_000))
        .build();
    let id = pic.create_canister();
    pic.add_cycles(id, 1_000_000_000_000_000);
    let wasm_path = std::env::var("GRANITE_WASM").unwrap_or_else(|_| {
        root.join("target/wasm32-unknown-unknown/release/granite_canister.wasm")
            .display()
            .to_string()
    });
    let wasm = fs::read(wasm_path).unwrap();
    let has_profile = wasm
        .windows(b"canister_query profileStep".len())
        .any(|s| s == b"canister_query profileStep");
    let wasm_hash = format!("{:x}", Sha256::digest(&wasm));
    let setup_started = Instant::now();
    pic.install_canister(id, wasm, Encode!().unwrap(), None);
    let install_seconds = setup_started.elapsed().as_secs_f64();
    for (i, chunk) in pack.chunks(1_800_000).enumerate() {
        let b = Encode!(
            &((i * 1_800_000) as u64),
            &serde_bytes::ByteBuf::from(chunk.to_vec())
        )
        .unwrap();
        pic.update_call(id, Principal::anonymous(), "upload", b)
            .unwrap();
    }
    let activated = pic
        .update_call(id, Principal::anonymous(), "activate", Encode!().unwrap())
        .unwrap();
    Decode!(&activated,Result<(),String>).unwrap().unwrap();
    drop(pack);
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let status = pic.canister_status(id, None).unwrap();
    fs::write(
        cache.join("model-memory.json"),
        serde_json::to_vec_pretty(&json!({
            "wasm_sha256":wasm_hash,
            "wasm_memory_size_bytes":status.memory_metrics.wasm_memory_size.to_string(),
            "total_memory_size_bytes":status.memory_size.to_string(),
            "wasm_binary_size_bytes":status.memory_metrics.wasm_binary_size.to_string(),
            "install_seconds":install_seconds,"setup_seconds":setup_seconds
        }))
        .unwrap(),
    )
    .unwrap();

    fs::copy(
        cache.join("model-memory.json"),
        cache.join(format!("model-memory-{wasm_hash}.json")),
    )
    .unwrap();
    println!(
        "activated {id}, tokens={length}, install={install_seconds:.3}s, setup={setup_seconds:.3}s"
    );
    if args.iter().any(|s| s == "--sharded-suite") {
        for key in [
            "1365",
            "1366",
            "2048",
            "4096",
            "5000",
            "space-5000",
            "cooking-5000",
            "weather-5000",
        ] {
            let ids: Vec<u32> = serde_json::from_slice(
                &fs::read(cache.join(format!("tokens-{key}.json"))).unwrap(),
            )
            .unwrap();
            let reference =
                decode_floats(&fs::read(cache.join(format!("reference-{key}.f32"))).unwrap())
                    .unwrap();
            let int8 =
                decode_floats(&fs::read(cache.join(format!("reference-int8-{key}.f32"))).unwrap())
                    .unwrap();
            sharded::run(
                &pic, id, &ids, &reference, &int8, &cache, &wasm_hash, true, key,
            );
        }
        return;
    }
    if args
        .iter()
        .any(|s| s == "--sharded" || s == "--sharded-raw")
    {
        sharded::run(
            &pic,
            id,
            &ids,
            &reference,
            &int8_reference,
            &cache,
            &wasm_hash,
            !args.iter().any(|s| s == "--sharded-raw"),
            &key,
        );
        return;
    }
    if args
        .iter()
        .any(|s| s == "--capacity" || s.starts_with("--capacity-range="))
    {
        let mut cases = Vec::new();
        let lengths: Vec<usize> = if let Some(range) = args
            .iter()
            .find_map(|s| s.strip_prefix("--capacity-range="))
        {
            let (first, last) = range.split_once(':').expect("use first:last");
            let (first, last): (usize, usize) = (first.parse().unwrap(), last.parse().unwrap());
            assert!(2 <= first && first <= last && last <= 512);
            (first..=last).collect()
        } else {
            vec![256, 272, 288, 296, 304]
        };
        let patterns: usize = args
            .iter()
            .find_map(|s| s.strip_prefix("--capacity-patterns="))
            .map(|s| s.parse().unwrap())
            .unwrap_or(1);
        assert!((1..=3).contains(&patterns));
        for n in lengths {
            for pattern in 0..patterns {
                let mut seed = 0x9e3779b9u32 ^ n as u32;
                let mut ids: Vec<u32> = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let v = match pattern {
                            0 => seed % 50368,
                            1 => 0,
                            _ => 50367,
                        };
                        if v == 50283 { 42 } else { v }
                    })
                    .collect();
                ids[0] = 50281;
                ids[n - 1] = 50282;
                let body: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
                let request = request([7, 0, 0, n, 12, 0, 999, n], &body);
                let raw = pic.query_call(
                    id,
                    Principal::anonymous(),
                    "step",
                    Encode!(&serde_bytes::ByteBuf::from(request)).unwrap(),
                );
                let mut next = 0;
                let mut state = serde_bytes::ByteBuf::new();
                let mut queries = 0;
                let expected = loop {
                    let (method, payload) = if next == 0 {
                        ("startEmbedding", Encode!(&ids).unwrap())
                    } else {
                        (
                            "continueEmbedding",
                            Encode!(&(n as u32), &next, &state).unwrap(),
                        )
                    };
                    let response = pic
                        .query_call(id, Principal::anonymous(), method, payload)
                        .unwrap();
                    let r = Decode!(&response,Result<EmbeddingStep,String>)
                        .unwrap()
                        .unwrap();
                    queries += 1;
                    if r.next_layer == 12 {
                        break r.embedding;
                    }
                    next = r.next_layer;
                    state = r.state;
                };
                let (instructions, error) = match raw {
                    Ok(response) => {
                        let m = Decode!(&response,Result<Measurement,String>)
                            .unwrap()
                            .unwrap();
                        assert_eq!(m.data, float_bytes(&expected));
                        (Some(m.instructions), None)
                    }
                    Err(e) => {
                        let error = format!("{e:?}");
                        assert!(
                            error.contains("CanisterInstructionLimitExceeded"),
                            "unexpected capacity error: {error}"
                        );
                        (None, Some(error))
                    }
                };
                println!("capacity tokens={n} raw_instructions={instructions:?} error={error:?}");
                cases.push(json!({"tokens":n,"pattern":pattern,"ids":ids,"raw_instructions":instructions,"error":error,"public_queries":queries,"embedding":expected}));
            }
        }
        fs::write(
            cache.join("capacity.json"),
            serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"cases":cases})).unwrap(),
        )
        .unwrap();
        return;
    }
    if args.iter().any(|s| s == "--micro-av") {
        let mut results = Vec::new();
        for n in [1usize, 7, 8, 511, 512, 513, 519, 1024, 1280, 1365, 5000] {
            for pattern in 0..4 {
                let payload = request([10, 0, 0, n, pattern, 0, 0, n], &[]);
                let response = pic
                    .query_call(
                        id,
                        Principal::anonymous(),
                        "profileStep",
                        Encode!(&serde_bytes::ByteBuf::from(payload)).unwrap(),
                    )
                    .unwrap();
                let measured = Decode!(&response,Result<ProfileMeasurement,String>)
                    .unwrap()
                    .unwrap();
                let got = decode_floats(&measured.data).unwrap();
                let expected: Vec<f32> = (0..32)
                    .map(|c| {
                        let sum: i64 = (0..n)
                            .map(|t| {
                                let p = if pattern == 3 {
                                    ((t * 73 + 11) % 2048) as i64
                                } else {
                                    2047
                                };
                                let w = match pattern {
                                    0 => 2047,
                                    1 => -2047,
                                    2 => {
                                        if t / 512 % 2 == 0 {
                                            2047
                                        } else {
                                            -2047
                                        }
                                    }
                                    _ => ((t * 41 + c * 97 + 3) % 4095) as i64 - 2047,
                                };
                                p * w
                            })
                            .sum();
                        sum as f32
                    })
                    .collect();
                assert_eq!(
                    float_bytes(&got),
                    float_bytes(&expected),
                    "AV n={n}, pattern={pattern}"
                );
                results.push(
                    json!({"tokens":n,"pattern":pattern,"instructions":measured.instructions}),
                );
            }
        }
        fs::write(
            cache.join("micro-av.json"),
            serde_json::to_vec_pretty(
                &json!({"wasm_sha256":wasm_hash,"exact_i64_cases":results.len(),"cases":results}),
            )
            .unwrap(),
        )
        .unwrap();
        println!("AV micro: all {} i64 reference cases passed", results.len());
        if !args.iter().any(|s| {
            s == "--micro" || s == "--micro-qk" || s == "--probe" || s == "--validate-long"
        }) {
            return;
        }
    }
    if args.iter().any(|s| s == "--micro-attention") {
        let mut cases = Vec::new();
        for n in [256, 512, 1365, 5000] {
            let qkv: Vec<f32> = (0..n * 3 * H)
                .map(|i| ((i * 73 + 11) % 2049) as f32 / 256. - 4.)
                .collect();
            let wire = granite_core::attention_wire(&qkv, 0, n, 0, 1, 0, n);
            let response = pic
                .query_call(
                    id,
                    Principal::anonymous(),
                    "profileStep",
                    Encode!(&serde_bytes::ByteBuf::from(request(
                        [2, 0, 0, n, 0, 1, 0, n],
                        &wire
                    )))
                    .unwrap(),
                )
                .unwrap();
            let measured = Decode!(&response,Result<ProfileMeasurement,String>)
                .unwrap()
                .unwrap();
            assert_eq!(measured.data.len(), n * D * 4);
            cases.push(json!({"keys":n,"rows":n,"heads":1,"instructions":measured.instructions,"phases":measured.phases,"output_sha256":format!("{:x}",Sha256::digest(&measured.data))}));
        }
        fs::write(
            cache.join(format!("micro-attention-{wasm_hash}.json")),
            serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"cases":cases})).unwrap(),
        )
        .unwrap();
        if !args
            .iter()
            .any(|s| s == "--micro-qk" || s == "--micro" || s == "--probe")
        {
            return;
        }
    }
    if args.iter().any(|s| s == "--micro-qk") {
        let mut cases = Vec::new();
        for n in [1, 3, 4, 7, 8, 511, 512, 513, 1365, 4096, 5000] {
            for pattern in 0..4 {
                let response = pic
                    .query_call(
                        id,
                        Principal::anonymous(),
                        "profileStep",
                        Encode!(&serde_bytes::ByteBuf::from(request(
                            [14, 0, 0, n, pattern, 0, 0, n],
                            &[]
                        )))
                        .unwrap(),
                    )
                    .unwrap();
                let measured = Decode!(&response,Result<ProfileMeasurement,String>)
                    .unwrap()
                    .unwrap();
                assert_eq!(measured.data.len(), 8);
                cases.push(json!({"keys":n,"pattern":pattern,"instructions":measured.instructions,"hash":measured.data}));
            }
        }
        fs::write(cache.join("micro-qk.json"),serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"exact_i64_cases":cases.len(),"exact_previous_cases":cases.len(),"cases":cases})).unwrap()).unwrap();
        println!(
            "QK micro: all {} scalar i64 and previous-path cases passed",
            cases.len()
        );
        if !args
            .iter()
            .any(|s| s == "--micro" || s == "--probe" || s == "--probe-single")
        {
            return;
        }
    }
    if args.iter().any(|s| s == "--micro") {
        let model = Model::load(&fs::read(cache.join("model.pack")).unwrap()).unwrap();
        let mut results = Vec::new();
        let latency_repeats: usize = args
            .iter()
            .find_map(|s| s.strip_prefix("--micro-repeat="))
            .map(|s| s.parse().unwrap())
            .unwrap_or(1);
        assert!((1..=100).contains(&latency_repeats));
        let row_counts: &[usize] = if args.iter().any(|s| s.starts_with("--micro-repeat=")) {
            &[256]
        } else if args.iter().any(|s| s == "--micro-extended") {
            &[1, 4, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256]
        } else {
            &[1, 8, 64, 256]
        };
        for &rows in row_counts {
            for which in 0..4 {
                for pattern in 0..2 {
                    let cols = if which == 3 { 1536 } else { 384 };
                    let x: Vec<f32> = (0..rows * cols)
                        .map(|i| {
                            if pattern == 0 {
                                ((i * 73 + which * 11) % 2049) as f32 / 256. - 4.
                            } else {
                                0.
                            }
                        })
                        .collect();
                    let payload = request([8, 0, 0, rows, which, 0, 0, rows], &float_bytes(&x));
                    let expected = model.run(&payload).unwrap();
                    let mut times = Vec::new();
                    for iteration in 0..latency_repeats {
                        let mut nonce_payload = payload.clone();
                        nonce_payload[24..28]
                            .copy_from_slice(&(iteration as u32 + 1).to_le_bytes());
                        let began = Instant::now();
                        let response = pic
                            .query_call(
                                id,
                                Principal::anonymous(),
                                if has_profile { "profileStep" } else { "step" },
                                Encode!(&serde_bytes::ByteBuf::from(nonce_payload)).unwrap(),
                            )
                            .unwrap();
                        let seconds = began.elapsed().as_secs_f64();
                        let measured = if has_profile {
                            Decode!(&response,Result<ProfileMeasurement,String>)
                                .unwrap()
                                .unwrap()
                        } else {
                            let m = Decode!(&response,Result<Measurement,String>)
                                .unwrap()
                                .unwrap();
                            ProfileMeasurement {
                                data: m.data,
                                instructions: m.instructions,
                                phases: Vec::new(),
                            }
                        };
                        assert_eq!(
                            expected, measured.data,
                            "micro rows={rows} projection={which} pattern={pattern}"
                        );
                        times.push(seconds);
                        if iteration + 1 == latency_repeats {
                            results.push(json!({"rows":rows,"cols":cols,"projection":which,"pattern":pattern,"instructions":measured.instructions,"hash":measured.data,"phases":measured.phases,"uncached_seconds":times}));
                        }
                    }
                }
            }
        }
        let report = json!({"wasm_sha256":wasm_hash,"kind":"linear-micro","exact_native_matches":results.len(),"cases":results});
        fs::write(
            cache.join("micro-linear.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("Microbenchmark passed all exact native comparisons.");
        if !args.iter().any(|s| {
            s == "--probe"
                || s == "--probe-single"
                || s == "--validate-long"
                || s == "--validate-regression"
        }) {
            return;
        }
    }
    for flag in ["--stress", "--sweep", "--stress-long"] {
        if !args.iter().any(|s| s == flag) {
            continue;
        }
        let sweep = flag.contains("sweep");
        let mut cases = Vec::new();
        let lengths: Vec<usize> = if flag == "--stress-long" {
            vec![
                513, 514, 519, 520, 575, 576, 577, 639, 640, 641, 767, 768, 769, 895, 896, 897,
                1023, 1024, 1025, 1151, 1152, 1153, 1279, 1280, 1281, 1359, 1360, 1361, 1362, 1363,
                1364, 1365,
            ]
        } else {
            if sweep {
                (2..=512).collect()
            } else {
                vec![
                    2, 3, 7, 8, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 287, 288, 511,
                    512,
                ]
            }
        };
        for n in lengths {
            for pattern in 0..if sweep { 1 } else { 3 } {
                let mut seed = 0x9e3779b9u32 ^ n as u32;
                let mut ids: Vec<u32> = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        match pattern {
                            0 => {
                                let token = seed % 50368;
                                if token == 50283 { 42 } else { token }
                            }
                            1 => 0,
                            _ => 50367,
                        }
                    })
                    .collect();
                ids[0] = 50281;
                ids[n - 1] = 50282;
                let mut next = 0u32;
                let mut state = serde_bytes::ByteBuf::new();
                let mut instructions = 0u64;
                let mut max_instructions = 0u64;
                let mut queries = 0;
                let embedding = loop {
                    let (method, payload) = if next == 0 {
                        ("startEmbedding", Encode!(&ids).unwrap())
                    } else {
                        (
                            "continueEmbedding",
                            Encode!(&(n as u32), &next, &state).unwrap(),
                        )
                    };
                    let response = pic
                        .query_call(id, Principal::anonymous(), method, payload)
                        .unwrap();
                    let r = Decode!(&response, Result<EmbeddingStep,String>)
                        .unwrap()
                        .unwrap();
                    assert!(
                        r.instructions < 5_000_000_000 && r.next_layer > next && r.next_layer <= 12
                    );
                    instructions += r.instructions;
                    max_instructions = max_instructions.max(r.instructions);
                    queries += 1;
                    if r.next_layer == 12 {
                        break r.embedding;
                    }
                    next = r.next_layer;
                    state = r.state;
                };
                let expected = granite_core::query_plan(n).unwrap().len();
                assert_eq!(queries, expected);
                assert_eq!(embedding.len(), H);
                assert!(embedding.iter().all(|v| v.is_finite()));
                assert!((embedding.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-4);
                cases.push(json!({"tokens":n,"pattern":pattern,"ids":ids,"embedding":embedding,
                    "instructions":instructions,"max_instructions":max_instructions,"queries":queries}));
                println!("stress tokens={n} pattern={pattern} instructions={instructions}");
            }
        }
        fs::write(
            cache.join(if flag == "--stress-long" {
                "stress-long.json"
            } else {
                if sweep { "sweep.json" } else { "stress.json" }
            }),
            serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"cases":cases})).unwrap(),
        )
        .unwrap();
    }
    if args
        .iter()
        .any(|s| s.starts_with("--sweep") || s.starts_with("--stress"))
        && !args
            .iter()
            .any(|s| s == "--validate-regression" || s == "--validate-long")
    {
        return;
    }
    if args.iter().any(|s| {
        s == "--probe"
            || s == "--probe-single"
            || s == "--validate-regression"
            || s == "--validate-long"
    }) {
        if args.iter().any(|s| s == "--probe") {
            let mut output = Vec::new();
            let mut trace = Vec::new();
            let mut total = 0;
            for layer in 0..12 {
                let body = if layer == 0 {
                    ids.iter().flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    float_bytes(&output)
                };
                let payload = request(
                    [
                        if layer == 0 { 7 } else { 6 },
                        layer,
                        0,
                        length,
                        1,
                        0,
                        0,
                        length,
                    ],
                    &body,
                );
                let response = pic
                    .query_call(
                        id,
                        Principal::anonymous(),
                        "profileStep",
                        Encode!(&serde_bytes::ByteBuf::from(payload)).unwrap(),
                    )
                    .unwrap();
                let p = Decode!(&response,Result<ProfileMeasurement,String>)
                    .unwrap()
                    .unwrap();
                total += p.instructions;
                output = decode_floats(&p.data).unwrap();
                trace.push(json!({"layer":layer,"instructions":p.instructions,"phases":p.phases}));
                println!("profile layer={layer} instructions={}", p.instructions);
            }
            fs::write(cache.join("probe-profile.json"),serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"tokens":length,"stats":{"instructions":total},"trace":trace,"embedding":output})).unwrap()).unwrap();
        }
        let mut results = Vec::new();
        for name in ["256", "space-256", "cooking-256", "weather-256"] {
            let ids: Vec<u32> = serde_json::from_slice(
                &fs::read(cache.join(format!("tokens-{name}.json"))).unwrap(),
            )
            .unwrap();
            let payload = request(
                [7, 0, 0, ids.len(), 12, 0, 0, ids.len()],
                &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            );
            let now = Instant::now();
            match pic.query_call(
                id,
                Principal::anonymous(),
                "step",
                Encode!(&serde_bytes::ByteBuf::from(payload)).unwrap(),
            ) {
                Err(e) => {
                    println!("single {name} rejected: {e:?}");
                    results.push(json!({"case":name,"ok":false,"error":format!("{e:?}")}));
                }
                Ok(response) => {
                    let m = Decode!(&response,Result<Measurement,String>)
                        .unwrap()
                        .unwrap();
                    let output = decode_floats(&m.data).unwrap();
                    assert_eq!(output.len(), 384);
                    assert!(m.instructions < 5_000_000_000);
                    let reference = decode_floats(
                        &fs::read(cache.join(format!("reference-{name}.f32"))).unwrap(),
                    )
                    .unwrap();
                    let previous_file = if name == "256" {
                        root.join("benchmarks/single-query/api-256.json")
                    } else {
                        root.join(format!("benchmarks/single-query/api-{}.json", name))
                    };
                    let previous: serde_json::Value =
                        serde_json::from_slice(&fs::read(previous_file).unwrap()).unwrap();
                    let old: Vec<f32> =
                        serde_json::from_value(previous["embedding"].clone()).unwrap();
                    let cosine = similarity(&output, &reference);
                    let cosine_previous = similarity(&output, &old);
                    println!(
                        "single {name} instructions={} cosine={cosine} previous={cosine_previous}",
                        m.instructions
                    );
                    results.push(json!({"case":name,"ok":true,"instructions":m.instructions,"seconds":now.elapsed().as_secs_f64(),"cosine_to_f32":cosine,"cosine_to_previous":cosine_previous,"embedding":output}));
                }
            }
        }
        fs::write(
            cache.join("probe-single.json"),
            serde_json::to_vec_pretty(&json!({"wasm_sha256":wasm_hash,"cases":results})).unwrap(),
        )
        .unwrap();
        fs::copy(
            cache.join("probe-single.json"),
            cache.join(format!("probe-single-{wasm_hash}.json")),
        )
        .unwrap();
        if args.iter().any(|s| s == "--repeat-single") {
            let mut repeats = Vec::new();
            for name in ["256", "space-256", "cooking-256", "weather-256"] {
                let ids: Vec<u32> = serde_json::from_slice(
                    &fs::read(cache.join(format!("tokens-{name}.json"))).unwrap(),
                )
                .unwrap();
                let body: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
                let payload = request([7, 0, 0, 256, 12, 0, 0, 256], &body);
                let expected: Vec<f32> = serde_json::from_value(
                    results.iter().find(|r| r["case"] == name).unwrap()["embedding"].clone(),
                )
                .unwrap();
                let repeat_count: usize = args
                    .iter()
                    .find_map(|s| s.strip_prefix("--repeat-count="))
                    .map(|s| s.parse().unwrap())
                    .unwrap_or(5);
                assert!((1..=1000).contains(&repeat_count));
                for iteration in 0..repeat_count {
                    // Distinct request bytes defeat the replica query cache. kstart is
                    // unused by fused op=7; token IDs and arithmetic stay identical.
                    let mut nonce_payload = payload.clone();
                    nonce_payload[24..28].copy_from_slice(&(iteration as u32 + 1).to_le_bytes());
                    let began = Instant::now();
                    let response = pic
                        .query_call(
                            id,
                            Principal::anonymous(),
                            "step",
                            Encode!(&serde_bytes::ByteBuf::from(nonce_payload)).unwrap(),
                        )
                        .unwrap();
                    let seconds = began.elapsed().as_secs_f64();
                    let m = Decode!(&response,Result<Measurement,String>)
                        .unwrap()
                        .unwrap();
                    assert_eq!(m.data, float_bytes(&expected));
                    repeats.push(json!({"case":name,"iteration":iteration,"seconds":seconds,"instructions":m.instructions}));
                }
            }
            fs::write(cache.join("repeat-single.json"), serde_json::to_vec_pretty(
                &json!({"wasm_sha256":wasm_hash,"cache_bypass":"unused fused-request kstart nonce","install_seconds":install_seconds,"setup_seconds":setup_seconds,"cases":repeats})
            ).unwrap()).unwrap();
        }
        if args.iter().any(|s| s == "--validate-regression") {
            assert!(
                results.iter().all(|r| r["ok"] == true),
                "single-query goal failed"
            );
            let mut keys: Vec<String> = [32, 64, 128, 160, 176, 256, 512]
                .into_iter()
                .map(|n| n.to_string())
                .collect();
            for topic in ["space", "cooking", "weather"] {
                for n in [160, 256, 512] {
                    keys.push(format!("{topic}-{n}"));
                }
            }
            for name in keys {
                let ids: Vec<u32> = serde_json::from_slice(
                    &fs::read(cache.join(format!("tokens-{name}.json"))).unwrap(),
                )
                .unwrap();
                let reference =
                    decode_floats(&fs::read(cache.join(format!("reference-{name}.f32"))).unwrap())
                        .unwrap();
                let quant_reference = decode_floats(
                    &fs::read(cache.join(format!("reference-int8-{name}.f32"))).unwrap(),
                )
                .unwrap();
                run_api(
                    &pic,
                    id,
                    &ids,
                    &reference,
                    &quant_reference,
                    &cache,
                    &name,
                    &wasm_hash,
                );
                let result: serde_json::Value = serde_json::from_slice(
                    &fs::read(cache.join(format!("api-{name}.json"))).unwrap(),
                )
                .unwrap();
                if ids.len() <= 256 {
                    assert_eq!(result["stats"]["queries"], 1);
                }
                if let Some(raw) = results.iter().find(|r| r["case"] == name) {
                    let a: Vec<f32> = serde_json::from_value(raw["embedding"].clone()).unwrap();
                    let b: Vec<f32> = serde_json::from_value(result["embedding"].clone()).unwrap();
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "raw/typed F32 output mismatch"
                    );
                }
                println!("API suite {name} passed");
            }
        }
        if !args.iter().any(|s| s == "--validate-long") {
            return;
        }
    }
    if args.iter().any(|s| s == "--validate-long") {
        let mut keys: Vec<String> = [640, 768, 1024, 1280, 1365]
            .into_iter()
            .map(|n| n.to_string())
            .collect();
        for name in ["space", "cooking", "weather"] {
            for n in [1280, 1365] {
                keys.push(format!("{name}-{n}"));
            }
        }
        for key in keys {
            let ids: Vec<u32> = serde_json::from_slice(
                &fs::read(cache.join(format!("tokens-{key}.json"))).unwrap(),
            )
            .unwrap();
            let f32ref =
                decode_floats(&fs::read(cache.join(format!("reference-{key}.f32"))).unwrap())
                    .unwrap();
            let intref =
                decode_floats(&fs::read(cache.join(format!("reference-int8-{key}.f32"))).unwrap())
                    .unwrap();
            run_api(&pic, id, &ids, &f32ref, &intref, &cache, &key, &wasm_hash);
        }
        return;
    }
    if args.iter().any(|s| s == "--micro-gelu") {
        let model = Model::load(&fs::read(cache.join("model.pack")).unwrap()).unwrap();
        let mut results = Vec::new();
        for rows in [1, 8, 64, 128] {
            let mut wide = vec![0.; rows * 3072];
            for (r, row) in wide.chunks_exact_mut(3072).enumerate() {
                for i in 0..1536 {
                    row[i] = (((r * 1536 + i) * 73) % 120001) as f32 / 5000. - 12.;
                    row[1536 + i] = 1. + (i % 17) as f32 / 16.;
                }
            }
            let payload = request([9, 0, 0, rows, 0, 0, 0, rows], &float_bytes(&wide));
            let expected = decode_floats(&model.run(&payload).unwrap()).unwrap();
            let response = pic
                .query_call(
                    id,
                    Principal::anonymous(),
                    "profileStep",
                    Encode!(&serde_bytes::ByteBuf::from(payload)).unwrap(),
                )
                .unwrap();
            let measured = Decode!(&response,Result<ProfileMeasurement,String>)
                .unwrap()
                .unwrap();
            let actual = decode_floats(&measured.data).unwrap();
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let mean_abs = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .sum::<f64>()
                / actual.len() as f64;
            results.push(json!({"rows":rows,"values":actual.len(),"max_abs_error":max_abs,"mean_abs_error":mean_abs,"instructions":measured.instructions,"phases":measured.phases}));
        }
        let report = json!({"kind":"gelu-micro","wasm_sha256":wasm_hash,"cases":results});
        fs::write(
            cache.join("micro-gelu.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("GELU microbenchmark completed.");
        return;
    }
    run_api(
        &pic,
        id,
        &ids,
        &reference,
        &int8_reference,
        &cache,
        &key,
        &wasm_hash,
    );
}
