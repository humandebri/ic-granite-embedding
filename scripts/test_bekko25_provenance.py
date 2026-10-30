"""Regression tests for early prerequisites and stale-release rejection."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bekko25_provenance as provenance
import build_bekko25 as builder
import report_bekko25_operations as reporter


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bekko25-provenance-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "target"
        self.out = self.root / "results"
        self.out.mkdir()
        self.write(
            "scripts/bekko25-build-inputs.json",
            json.dumps(
                {
                    "runtime_roots": ["runtime"],
                    "runtime_files": [provenance.CONFIG, "icp.yaml"],
                    "validation_roots": ["harness"],
                    "validation_files": [],
                }
            ),
        )
        self.write("runtime/src/lib.rs", "runtime v1")
        self.write("icp.yaml", "recipe v1")
        self.write("harness/src/main.rs", "harness v1")
        for name in ["model.pack", "cases.json", "retrieval.json"]:
            self.write(f".cache/bekko25/{name}", "[]")
        self.artifacts = [
            "wasm32-unknown-unknown/release/bekko25_canister.wasm",
            "wasm32-unknown-unknown/release/bekko25_compact_canister.wasm",
            "wasm32-unknown-unknown/release/bekko_canister.wasm",
            "release/bekko25-bench",
            "release/bekko25-load",
            "release/bekko25-live",
            "release/bekko25-compact-verify",
        ]
        for name in self.artifacts:
            self.write(f"target/{name}", name)
        self.record = {
            "schema": 1,
            "runtime_source_sha256": provenance.snapshot(self.root, "runtime"),
            "validation_source_sha256": provenance.snapshot(self.root, "validation"),
            "artifacts": {
                name: provenance.digest(self.target / name) for name in self.artifacts
            },
        }
        self.write("target/build-record.json", json.dumps(self.record))
        self.report = {
            **self.preflight(),
            "wasm_sha256": self.record["artifacts"][self.artifacts[0]],
            "evidence_sha256": {},
        }
        self.write("results/validation.json", json.dumps(self.report))
        self.write(
            "results/recipe-artifact.json",
            json.dumps(
                {
                    "raw_cargo_sha256": self.report["wasm_sha256"],
                    "deployed_sha256": "deployed",
                }
            ),
        )
        self.verified = {
            "validation_sha256": provenance.digest(self.out / "validation.json"),
            "build_record_sha256": self.report["build_record_sha256"],
            "wasm_sha256": self.report["wasm_sha256"],
            "deployed_sha256": "deployed",
            "evidence_sha256": {
                "recipe-artifact.json": provenance.digest(
                    self.out / "recipe-artifact.json"
                )
            },
        }
        self.write("results/verification.json", json.dumps(self.verified))

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def preflight(self):
        return provenance.preflight(
            self.root, self.target, self.target / "release/bekko25-bench", True
        )

    def test_matching_build_and_evidence_pass(self):
        report, record, verified = provenance.validate_release_inputs(
            self.root, self.out, self.target
        )
        self.assertEqual(report, self.report)
        self.assertEqual(record, self.record)
        self.assertEqual(verified, self.verified)

    def test_missing_a8_fails_before_build_record_or_network_is_needed(self):
        (self.target / self.artifacts[2]).unlink()
        (self.target / "build-record.json").unlink()
        with self.assertRaisesRegex(
            ValueError, "missing validation prerequisite: .*bekko_canister.wasm"
        ):
            self.preflight()

    def test_missing_retrieval_fails_early(self):
        (self.root / ".cache/bekko25/retrieval.json").unlink()
        with self.assertRaisesRegex(
            ValueError, "missing validation prerequisite: .*retrieval.json"
        ):
            self.preflight()

    def test_runtime_edit_rejected(self):
        self.write("runtime/src/lib.rs", "runtime v2")
        with self.assertRaisesRegex(ValueError, "runtime sources changed"):
            self.preflight()

    def test_recipe_config_edit_rejected(self):
        self.write("icp.yaml", "recipe v2")
        with self.assertRaisesRegex(ValueError, "runtime sources changed"):
            self.preflight()

    def test_new_source_file_rejected(self):
        self.write("runtime/src/attention.rs", "new module")
        with self.assertRaisesRegex(ValueError, "runtime sources changed"):
            self.preflight()

    def test_harness_edit_rejected(self):
        self.write("harness/src/main.rs", "harness v2")
        with self.assertRaisesRegex(ValueError, "validation sources changed"):
            self.preflight()

    def test_changed_wasm_rejected(self):
        self.write("target/" + self.artifacts[0], "other Wasm")
        with self.assertRaisesRegex(ValueError, "build artifact changed"):
            self.preflight()

    def test_other_running_executable_rejected(self):
        self.write("other-harness", "another executable")
        with self.assertRaisesRegex(ValueError, "running harness differs"):
            provenance.preflight(self.root, self.target, self.root / "other-harness")

    def test_legacy_report_requires_rerun(self):
        report = copy.deepcopy(self.report)
        del report["build_record"]
        self.write("results/validation.json", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "no build provenance"):
            provenance.validate_release_inputs(self.root, self.out, self.target)

    def test_mixed_validation_and_verification_rejected(self):
        self.write(
            "results/validation.json", json.dumps({**self.report, "another_run": True})
        )
        with self.assertRaisesRegex(ValueError, "another validation run"):
            provenance.validate_release_inputs(self.root, self.out, self.target)

    def test_changed_evidence_rejected(self):
        self.write("results/recipe-artifact.json", "{}")
        with self.assertRaisesRegex(ValueError, "verification evidence changed"):
            provenance.validate_release_inputs(self.root, self.out, self.target)

    def test_changed_retrieval_input_rejected(self):
        self.write(".cache/bekko25/retrieval.json", "[1]")
        with self.assertRaisesRegex(ValueError, "validation input changed"):
            provenance.validate_release_inputs(self.root, self.out, self.target)

    def test_reporter_leaves_existing_release_untouched_on_stale_sources(self):
        self.write("results/release.json", "previous release")
        self.write("results/RESULTS.md", "previous results")
        self.write("runtime/src/lib.rs", "runtime v2")
        with (
            patch.object(reporter, "ROOT", self.root),
            patch.object(reporter, "OUT", self.out),
            patch.dict("os.environ", {"CARGO_TARGET_DIR": str(self.target)}),
            self.assertRaisesRegex(ValueError, "runtime sources changed"),
        ):
            reporter.main()
        self.assertEqual((self.out / "release.json").read_text(), "previous release")
        self.assertEqual((self.out / "RESULTS.md").read_text(), "previous results")

    def test_new_report_labels_old_shared_runtime_checks_as_historical(self):
        # Exercise the renderer with small synthetic data. Validation is
        # stubbed because this test checks wording, not model inference.
        report = {
            **self.report,
            "model": {
                "runtime_id": "test",
                "embedding_space": "test",
                "weight_pack_sha256": "test",
            },
            "cases": [
                {"tokens": n, "instructions": 1}
                for n in [2, 31, 63, 64, 65, 128, 192, 255, 256]
            ],
            "checks": ["check"],
            "final_checks": ["final check"],
            "wasm_memory_bytes": 1,
            "stable_memory_bytes": 1,
        }
        verified = {
            **self.verified,
            "torch_references_checked": True,
            "languages": {
                "en": {
                    "reference_recall5": 0.5,
                    "qk13_reference_recall5": 0.5,
                    "wasm_recall5": 0.5,
                }
            },
            "maximum_instructions": 1,
            "margin_percent": 99.0,
            "retrieval_embeddings": 1,
        }
        build = {**self.record, "rustc": "test rustc", "cargo": "test cargo"}
        report["build_record_sha256"] = "new-build"
        self.write(
            "results/live.json",
            json.dumps({"status": {"id": "test"}, "network": {"api_url": "test"}}),
        )
        self.write("results/sweep.json", "[]")
        self.write("results/stress.json", "[]")
        self.write("results/live-sweep.json", '{"results": []}')
        # Stale/unparseable mutable records must never become current success
        # claims. The historical section links to immutable published records.
        self.write("results/a8-regression.json", "old invalid record")
        self.write("results/generator-regression.json", "old invalid record")
        with (
            patch.object(reporter, "ROOT", self.root),
            patch.object(reporter, "OUT", self.out),
            patch.object(
                reporter,
                "validate_release_inputs",
                return_value=(report, build, verified),
            ),
        ):
            reporter.main()
        text = (self.out / "RESULTS.md").read_text()
        self.assertIn("2026-10-29 release records", text)
        self.assertIn("Historical shared-runtime regression records", text)
        self.assertIn("do not establish regression success for the current build", text)
        self.assertIn(f"/blob/{reporter.HISTORICAL_REVISION}/", text)
        self.assertNotIn("Its 15 fixtures and lifecycle/retrieval smoke pass", text)
        self.assertNotIn("[regression](a8-regression.json)", text)
        published = json.loads((self.out / "release.json").read_text())
        self.assertEqual(published["build_record"], build)

    def test_failed_build_removes_previous_attestation(self):
        with (
            patch.object(builder, "ROOT", self.root),
            patch("sys.argv", ["build_bekko25.py", "--target-dir", str(self.target)]),
            patch.object(
                builder.subprocess, "run", side_effect=RuntimeError("compiler failed")
            ),
            self.assertRaisesRegex(RuntimeError, "compiler failed"),
        ):
            builder.main()
        self.assertFalse((self.target / "build-record.json").exists())

    def test_source_edit_during_build_is_not_attested(self):
        def compile_step(*args, **kwargs):
            self.write("runtime/src/lib.rs", "changed during compile")

        with (
            patch.object(builder, "ROOT", self.root),
            patch("sys.argv", ["build_bekko25.py", "--target-dir", str(self.target)]),
            patch.object(builder.subprocess, "run", side_effect=compile_step),
            self.assertRaisesRegex(ValueError, "sources changed during build"),
        ):
            builder.main()
        self.assertFalse((self.target / "build-record.json").exists())


if __name__ == "__main__":
    unittest.main()
