"""Compare a selected canister's live status to the verified recipe Wasm."""
import argparse
import contextlib
import io
import json
import subprocess

import report_bekko25_compact as compact_report


def check_status(status, expected_id, expected_hash):
    if not isinstance(status, dict) or status.get("id") != expected_id:
        raise ValueError("status belongs to another canister")
    if status.get("module_hash") != "0x" + expected_hash:
        raise ValueError("canister module hash differs from verified recipe Wasm")


def read_live_status(canister_id, environment, identity=None):
    command = ["icp", "canister", "status", canister_id, "-e", environment, "--json"]
    if identity:
        command += ["--identity", identity]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canister-id", required=True, help="expected canister principal")
    parser.add_argument("--environment", default="local")
    parser.add_argument("--identity")
    args = parser.parse_args()
    # Hold the same lock throughout, and recheck the build after reading status.
    with compact_report.result_lock():
        with contextlib.redirect_stdout(io.StringIO()) as output:
            compact_report._report_locked()
        report = json.loads(output.getvalue())
        status = read_live_status(args.canister_id, args.environment, args.identity)
        check_status(status, args.canister_id, report["compact_wasm_sha256"])
        with contextlib.redirect_stdout(io.StringIO()) as final:
            compact_report._report_locked()
        if json.loads(final.getvalue()) != report:
            raise ValueError("verification changed while checking deployment")
    print(f"{args.canister_id}: module hash matches verified compact recipe Wasm")


if __name__ == "__main__":
    main()
