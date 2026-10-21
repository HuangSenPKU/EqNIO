"""Result layout, input protection, and publication checks for the batch runner."""
import csv
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_baselines import (
    alignment_directory, commands, parse_args, preflight, publish_results, result_directory,
)


class ResultLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            data_root=self.root / "PedROAM", out_root=self.root / "PedROAM/results",
            work_root=self.root / "work", checkpoint_root=self.root / "models",
            test_name="baseline-v1", session="session-a", person="P02", max_seconds=180,
            cpu=True, workers=1, initialization_distance_m=10.,
        )
        self.record = {"session": "session-a", "person": "P02"}
        for relative in ["recordings/session-a/phones/P02/sensorlogger/Gyroscope.csv",
                         "recordings/session-a/phones/P02/sensorlogger/TotalAcceleration.csv",
                         "recordings/session-a/reference/P02/trajectory-gt.csv"]:
            path = self.args.data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("input must remain unchanged")
        for relative in ["Ronin_o2/checkpoint_38.pt", "Ronin_o2/config.json",
                         "TLIO_o2/checkpoint_best.pt", "TLIO_o2/parameters.json"]:
            path = self.args.checkpoint_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("checkpoint")

    def test_default_results_and_single_component_test_name(self):
        argv = ["run_baselines.py", "--data-root", str(self.args.data_root), "--test-name", "trial"]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.out_root, self.args.data_root / "results")
        self.assertEqual(result_directory(args, self.record), args.out_root / "session-a/P02/trial")
        with patch("sys.argv", argv[:-1] + ["../escape"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_existing_results_base_allowed_but_existing_experiment_refused(self):
        old = self.args.out_root / "session-a/P02/rtklib-spp"
        old.mkdir(parents=True)
        self.assertEqual(preflight(self.args, ["ronin", "tlio"]), [self.record])
        result_directory(self.args, self.record).mkdir()
        with self.assertRaisesRegex(ValueError, "Experiment already exists"):
            preflight(self.args, ["ronin"])
        self.assertTrue(old.is_dir())

    def test_raw_input_and_symlink_escape_rejected(self):
        original = self.args.out_root
        self.args.out_root = self.args.data_root / "recordings"
        with self.assertRaisesRegex(ValueError, "input data"):
            preflight(self.args, ["ronin"])
        self.args.out_root = original
        original.mkdir(parents=True)
        (original / "session-a").symlink_to(self.args.data_root / "recordings/session-a", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            preflight(self.args, ["ronin"])

    def test_work_paths_and_nondefault_alignment_distance(self):
        self.args.initialization_distance_m = 5.
        steps = commands(self.args, ["ronin", "tlio"])
        for name, command in steps:
            target = Path(command[command.index("--out-root") + 1])
            self.assertTrue(target.is_relative_to(self.args.work_root))
            if name != "reference-audit":
                self.assertEqual(command[command.index("--initialization-distance-m") + 1], "5.0")
            if name.endswith("-alignment"):
                self.assertEqual(target.name, "alignment-5m-continuous")

    def test_publication_is_per_record_self_contained_and_never_overwrites(self):
        models = ["ronin", "tlio"]
        work = self.args.work_root
        alignment = alignment_directory(10.)
        for model in models:
            inference = work / model / "inference"
            common = inference / "session-a/P02"
            source = common / "model-output"
            source.mkdir(parents=True)
            (inference / "run.json").write_text(json.dumps({"method": "test", "trajectory_subdir": "model-output"}))
            (source / "trajectory_local.csv").write_text("raw bytes\n")
            (source / "trajectory_initial_aligned.csv").write_text("aligned bytes\n")
            (common / "input_quality.json").write_text("{}")
            report = work / model / alignment / ("Madgwick" if model == "ronin" else "TLIO-EKF") / "session-a/P02"
            report.mkdir(parents=True)
            (report / "summary.json").write_text(json.dumps({"source": str(source)}))
            (report / "evaluation.csv").write_text("initial errors\n")
            (work / model / alignment / "run.json").write_text("{}")
        audit = work / "reference-audit"
        (audit / "session-a/P02").mkdir(parents=True)
        (audit / "session-a/P02/reference_quality.csv").write_text("quality\n")
        (audit / "audit.json").write_text(json.dumps({"sources": {"session-a/P02": {}, "other/P01": {}}}))
        (audit / "per_sequence.csv").write_text("session,person\nsession-a,P02\nother,P01\n")
        (audit / "discontinuities.csv").write_text("\n")
        (work / "per_sequence.csv").write_text(
            "baseline,session,person,initial_post_rmse_m\nronin,session-a,P02,1\ntlio,session-a,P02,2\nronin,other,P01,99\n")
        manifest = {"published_results": [], "status": "running"}
        publish_results(self.args, models, [self.record], manifest)
        target = result_directory(self.args, self.record)
        self.assertEqual(json.loads((target / "run.json").read_text())["status"], "completed")
        with (target / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["baseline"] for row in rows}, set(models))
        self.assertEqual(manifest["published_results"], [str(target)])
        self.assertFalse((target / ".cache").exists())
        for model in models:
            self.assertEqual((target / model / "trajectory_local.csv").read_text(), "raw bytes\n")
            self.assertTrue((target / model / "input_quality.json").exists())
            summary = json.loads((target / model / "evaluation/summary.json").read_text())
            self.assertEqual(summary["source"], str(target / model))
        with self.assertRaises(FileExistsError):
            publish_results(self.args, models, [self.record], manifest)
        self.args.test_name = "copy-failure"
        failed_target = result_directory(self.args, self.record)
        real_copytree = shutil.copytree

        def fail_archive_copy(source, destination, *args, **kwargs):
            if Path(destination).is_relative_to(failed_target):
                raise OSError("simulated copy failure")
            return real_copytree(source, destination, *args, **kwargs)

        with patch("run_baselines.shutil.copytree", side_effect=fail_archive_copy):
            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                publish_results(self.args, models, [self.record], manifest)
        self.assertEqual(json.loads((failed_target / "run.json").read_text())["status"], "publishing")
        self.assertNotIn(str(failed_target), manifest["published_results"])


if __name__ == "__main__":
    unittest.main()
