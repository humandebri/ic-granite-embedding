//! Resumable fixed-pack loader. icp handles identity and network selection.
use candid::{Decode, Deserialize, Encode};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command, time::Duration};

#[derive(Deserialize, candid::CandidType)]
struct Info {
    model_id: String,
    revision: String,
    embedding_space: String,
    weight_pack_sha256: String,
    uploaded_bytes: u64,
    weights_verified: bool,
    active: bool,
    last_error: Option<String>,
}

struct Client {
    environment: String,
    canister: String,
    identity: Option<String>,
    file: PathBuf,
    icp: PathBuf,
}
impl Drop for Client {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.file);
    }
}
impl Client {
    fn call(&self, method: &str, args: Vec<u8>, query: bool) -> Vec<u8> {
        fs::write(&self.file, args).expect("write binary Candid arguments");
        for attempt in 0..3 {
            let mut command = Command::new(&self.icp);
            command
                .args([
                    "canister",
                    "call",
                    &self.canister,
                    method,
                    "-e",
                    &self.environment,
                    "--candid",
                    "crates/bekko25-canister/bekko25.did",
                    "--output",
                    "hex",
                    "--args-format",
                    "bin",
                    "--args-file",
                ])
                .arg(&self.file);
            if query {
                command.arg("--query");
            }
            if let Some(identity) = &self.identity {
                command.args(["--identity", identity]);
            }
            let output = command.output().expect("run icp");
            if output.status.success() {
                let hex = String::from_utf8(output.stdout).expect("hex response");
                let hex = hex.trim().trim_start_matches("0x");
                assert!(hex.len() % 2 == 0);
                return (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex reply bytes"))
                    .collect();
            }
            if attempt == 2 {
                panic!("{method}: {}", String::from_utf8_lossy(&output.stderr));
            }
            // Duplicate upload calls verify already-stored bytes without
            // advancing the digest again; activate is idempotent.
            std::thread::sleep(Duration::from_millis(500 * (1 << attempt)));
        }
        unreachable!()
    }
    fn info(&self) -> Info {
        // A replicated query observes preceding completed updates even when
        // the gateway chooses another replica. Upload decisions need this.
        Decode!(&self.call("modelInfo", Encode!().unwrap(), false), Info).unwrap()
    }
    fn update(&self, method: &str) {
        Decode!(&self.call(method,Encode!().unwrap(),false),Result<(),String>)
            .unwrap()
            .unwrap();
    }
}
fn upload_pack(client: &Client, pack: &[u8], uploaded_bytes: u64, restart: bool) {
    let mut offset = uploaded_bytes as usize;
    if restart {
        client.update("beginUpload");
        // Acknowledged reset establishes the offset, without another read.
        offset = 0;
    }
    assert!(offset <= pack.len());
    while offset < pack.len() {
        let end = (offset + 1_800_000).min(pack.len());
        let reply = client.call(
            "upload",
            Encode!(
                &(offset as u64),
                &serde_bytes::ByteBuf::from(pack[offset..end].to_vec())
            )
            .unwrap(),
            false,
        );
        let next = Decode!(&reply,Result<u64,String>).unwrap().unwrap() as usize;
        assert!(next >= end && next <= pack.len());
        offset = next;
        if offset % 90_000_000 < 1_800_000 || offset == pack.len() {
            println!("uploaded {offset}/{}", pack.len());
        }
    }
}
fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_current_dir(&root).unwrap();
    let mut environment = "local".to_string();
    let mut canister = "bekko25".to_string();
    let mut identity = None;
    let mut pack_path = root.join(".cache/bekko25/model.pack");
    let mut restart = false;
    let mut activate_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--environment" => environment = args.next().expect("environment"),
            "--canister" => canister = args.next().expect("canister name or ID"),
            "--identity" => identity = Some(args.next().expect("identity")),
            "--pack" => pack_path = PathBuf::from(args.next().expect("pack path")),
            "--restart" => restart = true,
            "--activate-only" => activate_only = true,
            "--help" => {
                println!(
                    "bekko25-load [--environment local] [--canister bekko25] [--identity NAME] [--pack PATH] [--restart | --activate-only]"
                );
                return;
            }
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(
        !(restart && activate_only),
        "--restart conflicts with --activate-only"
    );
    fs::create_dir_all(root.join(".cache/bekko25")).unwrap();
    let file = root.join(format!(".cache/bekko25/upload-{}.bin", std::process::id()));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&file)
        .expect("unique argument file");
    let client = Client {
        environment,
        canister,
        identity,
        file,
        icp: PathBuf::from("icp"),
    };
    let info = client.info();
    assert_eq!(info.model_id, bekko25_core::MODEL_ID);
    assert_eq!(info.revision, bekko25_core::REVISION);
    assert_eq!(info.embedding_space, bekko25_core::EMBEDDING_SPACE);
    assert_eq!(info.weight_pack_sha256, bekko25_core::PACK_SHA256);
    assert!(
        info.last_error
            .as_deref()
            .is_none_or(|e| e.contains("incomplete upload discarded")
                || e.contains("upload the complete verified")),
        "canister storage error: {:?}",
        info.last_error
    );
    if !activate_only {
        let pack = fs::read(&pack_path).expect("read fixed weight pack");
        assert_eq!(pack.len(), bekko25_core::PACK_BYTES);
        assert_eq!(
            format!("{:x}", Sha256::digest(&pack)),
            bekko25_core::PACK_SHA256
        );
        upload_pack(&client, &pack, info.uploaded_bytes, restart);
    }
    client.update("activate");
    let ready = client.info();
    assert!(ready.active && ready.weights_verified);
    println!(
        "{} / {} ready: {}",
        client.environment, client.canister, ready.embedding_space
    );
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn hex(bytes: Vec<u8>) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
    fn fixture(name: &str, script: &str) -> (PathBuf, Client) {
        let dir =
            std::env::temp_dir().join(format!("bekko25-loader-{}-{name}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let icp = dir.join("icp");
        fs::write(&icp, format!("#!/usr/bin/env python3\n{script}")).unwrap();
        fs::set_permissions(&icp, fs::Permissions::from_mode(0o700)).unwrap();
        let client = Client {
            environment: "local".into(),
            canister: "test".into(),
            identity: None,
            file: dir.join("args.bin"),
            icp,
        };
        (dir, client)
    }

    #[test]
    fn state_reads_use_replicated_calls_even_if_query_state_is_stale() {
        let info = |active| Info {
            model_id: "test".into(),
            revision: "test".into(),
            embedding_space: "test".into(),
            weight_pack_sha256: "test".into(),
            uploaded_bytes: 3,
            weights_verified: active,
            active,
            last_error: None,
        };
        let (dir, client) = fixture(
            "replicated",
            &format!(
                "import sys\nprint('{}' if '--query' in sys.argv else '{}')\n",
                hex(Encode!(&info(false)).unwrap()),
                hex(Encode!(&info(true)).unwrap())
            ),
        );
        assert!(client.info().active);
        drop(client);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn restart_uploads_from_zero_despite_previous_complete_offset() {
        let ok = hex(Encode!(&Ok::<(), String>(())).unwrap());
        let uploaded = hex(Encode!(&Ok::<u64, String>(3)).unwrap());
        let (dir, client) = fixture(
            "restart",
            &format!(
                "import sys\nfrom pathlib import Path\np = Path(__file__).parent\nmethod = sys.argv[4]\nwith (p / 'calls').open('a') as f: f.write(method + '\\n')\nif method == 'beginUpload':\n print('{ok}')\nelif method == 'upload':\n (p / 'uploaded').write_bytes(Path(sys.argv[sys.argv.index('--args-file') + 1]).read_bytes())\n print('{uploaded}')\nelse:\n raise RuntimeError('unexpected state read after reset')\n"
            ),
        );
        upload_pack(&client, &[7, 8, 9], 3, true);
        assert_eq!(
            fs::read_to_string(dir.join("calls")).unwrap(),
            "beginUpload\nupload\n"
        );
        let args = fs::read(dir.join("uploaded")).unwrap();
        let (offset, bytes) = Decode!(&args, u64, serde_bytes::ByteBuf).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(bytes.as_ref(), &[7, 8, 9]);
        drop(client);
        fs::remove_dir_all(dir).unwrap();
    }
}
