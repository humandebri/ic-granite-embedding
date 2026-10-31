"""Negative tests for compact release identity and retrieval gates."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bekko25_provenance as provenance
import report_bekko25_compact as report
import build_bekko25_compact_recipe as recipe_builder
import verify_bekko25_compact_deployment as deployment


class CompactReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bekko25-compact-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.fixture_path = self.root / ".cache/bekko-memory-study/retrieval-ja-hard.json"
        self.result_path = self.root / ".cache/bekko25/compact-verification.json"
        self.result_path.parent.mkdir(parents=True)
        self.write("scripts/bekko25-build-inputs.json", json.dumps({
            "runtime_roots": ["runtime"], "runtime_files": [provenance.CONFIG, "icp.yaml"],
            "validation_roots": ["validation"], "validation_files": []}))
        self.write("runtime/lib.rs", "runtime v1")
        self.write("validation/main.rs", "validation v1")
        self.write("icp.yaml", "recipe v1")
        self.write(".cache/bekko25/cases.json", "cases v1")
        names = ["wasm32-unknown-unknown/release/bekko25_canister.wasm",
                 "wasm32-unknown-unknown/release/bekko25_compact_canister.wasm",
                 "wasm32-unknown-unknown/release/bekko_canister.wasm",
                 "release/bekko25-bench", "release/bekko25-load", "release/bekko25-live",
                 "release/bekko25-compact-verify"]
        for name in names:
            self.write("target/" + name, name)
        self.record = {"schema": 1, "rustc": "rustc test", "cargo": "cargo test",
                       "rustflags": {"RUSTFLAGS": None, "CARGO_ENCODED_RUSTFLAGS": None},
                       **{f"{k}_source_sha256": provenance.snapshot(self.root, k)
                                        for k in ("runtime", "validation")},
                       "artifacts": {name: provenance.digest(self.target / name) for name in names}}
        self.write("target/build-record.json", json.dumps(self.record))
        queries = [f"q{i}" for i in range(50)]
        documents = [f"d{i}" for i in range(50)]
        fixture = [{"query_ids": queries, "document_ids": documents,
                    "qrels": {q: [d] for q, d in zip(queries, documents)}}]
        self.fixture_path.parent.mkdir(parents=True)
        self.fixture_path.write_text(json.dumps(fixture))
        vectors = [[1.0 if j == i else 0.0 for j in range(384)] for i in range(50)]
        vectors += vectors.copy()
        rows = [{"before": vectors[i % 100], "after": vectors[i % 100],
                 "after_instructions": 1} for i in range(475)]
        self.write("target/" + provenance.COMPACT_RECIPE, "recipe wasm")
        self.recipe = {"schema": 1, "build_record_sha256": provenance.digest(self.target / "build-record.json"),
                       "icp_yaml_sha256": provenance.digest(self.root / "icp.yaml"),
                       "raw_cargo_sha256": self.record["artifacts"][provenance.COMPACT_RAW],
                       "recipe_wasm_sha256": provenance.digest(self.target / provenance.COMPACT_RECIPE),
                       "icp_version": "test"}
        self.write("target/" + provenance.COMPACT_RECIPE_RECORD, json.dumps(self.recipe))
        self.result = {
            "build_provenance": {"build_record": self.record,
                                 "build_record_sha256": provenance.digest(self.target / "build-record.json"),
                                 "compact_recipe": self.recipe,
                                 "compact_recipe_record_sha256": provenance.digest(self.target / provenance.COMPACT_RECIPE_RECORD),
                                 "input_sha256": {name: provenance.digest(self.root / name) for name in provenance.COMPACT_INPUTS}},
            "baseline_wasm_sha256": "baseline",
            "int8_wasm_sha256": self.recipe["recipe_wasm_sha256"],
            "original_pack_sha256": report.BASELINE_PACK_SHA256,
            "int8_pack_sha256": report.PACK_SHA256,
            "comparisons": rows}
        self.save()
        for name, value in (("ROOT", self.root), ("FIXTURE", self.fixture_path),
                            ("RESULT", self.result_path),
                            ("FIXTURE_SHA256", provenance.digest(self.fixture_path))):
            patcher = patch.object(report, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"CARGO_TARGET_DIR": str(self.target)})
        env.start()
        self.addCleanup(env.stop)

    def write(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def save(self):
        self.result_path.write_text(json.dumps(self.result))

    def run_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report.report()

    def test_valid_record_is_accepted(self):
        self.run_report()

    def test_other_wasm_is_rejected(self):
        self.result["int8_wasm_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "another compact recipe Wasm"):
            self.run_report()

    def test_source_change_is_rejected(self):
        self.write("runtime/lib.rs", "runtime v2")
        with self.assertRaisesRegex(ValueError, "sources changed"):
            self.run_report()

    def test_stale_build_record_is_rejected(self):
        self.write("target/build-record.json", "{}")
        with self.assertRaisesRegex(ValueError, "build record changed"):
            self.run_report()

    def test_changed_case_input_is_rejected(self):
        self.write(".cache/bekko25/cases.json", "changed cases")
        with self.assertRaisesRegex(ValueError, "validation input changed"):
            self.run_report()

    def test_missing_input_hash_is_rejected(self):
        del self.result["build_provenance"]["input_sha256"][".cache/bekko25/cases.json"]
        self.save()
        with self.assertRaisesRegex(ValueError, "input record is incomplete"):
            self.run_report()

    def test_recipe_configuration_change_is_rejected(self):
        self.write("icp.yaml", "recipe v2")
        with self.assertRaisesRegex(ValueError, "runtime sources changed"):
            self.run_report()

    def test_recipe_artifact_change_is_rejected(self):
        self.write("target/" + provenance.COMPACT_RECIPE, "different recipe wasm")
        with self.assertRaisesRegex(ValueError, "recipe Wasm changed"):
            self.run_report()

    def test_recipe_record_change_is_rejected(self):
        self.recipe["icp_version"] = "another version"
        self.write("target/" + provenance.COMPACT_RECIPE_RECORD, json.dumps(self.recipe))
        with self.assertRaisesRegex(ValueError, "recipe record changed"):
            self.run_report()

    def test_collapsed_vectors_are_rejected(self):
        for row in self.result["comparisons"][:100]:
            row["after"] = [1.0] + [0.0] * 383
        self.save()
        with self.assertRaisesRegex(ValueError, "retrieval quality"):
            self.run_report()

    def test_running_verification_blocks_reporting(self):
        self.write(".cache/bekko25/compact-verification.lock", "")
        with self.assertRaisesRegex(ValueError, "still running"):
            self.run_report()

    def test_deployed_canister_identity_and_hash_must_both_match(self):
        sha = self.recipe["recipe_wasm_sha256"]
        status = {"id": "test-canister", "module_hash": "0x" + sha}
        deployment.check_status(status, "test-canister", sha)
        with self.assertRaisesRegex(ValueError, "another canister"):
            deployment.check_status(status, "other-canister", sha)
        with self.assertRaisesRegex(ValueError, "module hash differs"):
            deployment.check_status({**status, "module_hash": "0x" + "0" * 64}, "test-canister", sha)

    def test_deployment_check_reads_live_status_for_exact_target(self):
        from types import SimpleNamespace
        with patch.object(deployment.subprocess, "run", return_value=SimpleNamespace(stdout='{"id":"test-canister"}')) as run:
            self.assertEqual(deployment.read_live_status("test-canister", "local", "alice")["id"], "test-canister")
        self.assertEqual(run.call_args.args[0], ["icp", "canister", "status", "test-canister", "-e", "local", "--json", "--identity", "alice"])

    def run_recipe_build(self, build_step):
        def version(args, **kwargs):
            return {"rustc": b"rustc test\n", "cargo": b"cargo test\n", "icp": b"icp test\n"}[args[0]]
        with (patch.object(recipe_builder, "ROOT", self.root),
              patch("sys.argv", ["build_bekko25_compact_recipe.py", "--target-dir", str(self.target)]),
              patch.object(recipe_builder.subprocess, "check_output", side_effect=version),
              patch.object(recipe_builder.subprocess, "run", side_effect=build_step),
              contextlib.redirect_stdout(io.StringIO())):
            recipe_builder.main()

    def test_recipe_build_records_exact_produced_artifact(self):
        self.run_recipe_build(lambda *args, **kwargs: self.write(".icp/cache/artifacts/bekko25_compact", "new recipe wasm"))
        recipe, _ = provenance.validate_compact_recipe(self.root, self.target, self.record)
        self.assertEqual(recipe["recipe_wasm_sha256"], provenance.digest(self.target / provenance.COMPACT_RECIPE))
        self.assertEqual((self.target / provenance.COMPACT_RECIPE).read_text(), "new recipe wasm")

    def test_raw_wasm_change_during_recipe_build_invalidates_attestation(self):
        def change(*args, **kwargs):
            self.write("target/" + provenance.COMPACT_RAW, "changed raw wasm")
        with self.assertRaisesRegex(ValueError, "build artifact changed"):
            self.run_recipe_build(change)
        self.assertFalse((self.target / provenance.COMPACT_RECIPE_RECORD).exists())

    def test_config_change_during_recipe_build_invalidates_attestation(self):
        def change(*args, **kwargs):
            self.write("icp.yaml", "changed recipe")
        with self.assertRaisesRegex(ValueError, "runtime sources changed"):
            self.run_recipe_build(change)
        self.assertFalse((self.target / provenance.COMPACT_RECIPE_RECORD).exists())

    def test_recipe_build_does_not_replace_artifact_during_verification(self):
        self.write(".cache/bekko25/compact-verification.lock", "")
        old = (self.target / provenance.COMPACT_RECIPE_RECORD).read_bytes()
        with self.assertRaisesRegex(ValueError, "still running"):
            self.run_recipe_build(lambda *args, **kwargs: self.fail("must not build while locked"))
        self.assertEqual((self.target / provenance.COMPACT_RECIPE_RECORD).read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
