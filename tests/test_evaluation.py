"""Checks for misleading metrics and damaged historical records."""
import json
import os
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from image_error_eval.evaluation import evaluate, load_records, compare
from image_error_eval.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_cli_reports_are_identical_across_hash_seeds(self):
        summaries = []
        comparisons = []
        with tempfile.TemporaryDirectory() as folder:
            for seed in ("1", "2", "3", "17"):
                output = Path(folder) / seed
                subprocess.run(
                    [sys.executable, "-m", "image_error_eval", "compare", "--output-dir", str(output)],
                    cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=seed),
                    check=True, capture_output=True, timeout=20,
                )
                summaries.append((output / "summary.json").read_bytes())
                comparisons.append((output / "comparison.md").read_bytes())
        self.assertTrue(all(value == summaries[0] for value in summaries))
        self.assertTrue(all(value == comparisons[0] for value in comparisons))

    def test_cli_writes_canonical_lf_files(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = main([
                "compare", "--annotations", str(ROOT / "data/annotations.json"),
                "--predictions-dir", str(ROOT / "results/predictions"),
                "--output-dir", folder,
            ])
            self.assertEqual(result, 0)
            for name in ("summary.json", "comparison.md"):
                contents = (Path(folder) / name).read_bytes()
                self.assertNotIn(b"\r\n", contents)
                self.assertTrue(contents.endswith(b"\n"))

    def test_metrics_on_hand_calculated_multilabel_example(self):
        truth = {"a.jpg": {"wrong", "weird"}, "b.jpg": {"other"}, "c.jpg": {"unrealistic"}}
        pred = {"c.jpg": set(), "b.jpg": {"other"}, "a.jpg": {"wrong"}}
        result = evaluate(truth, pred)
        self.assertAlmostEqual(result["exact_match"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertAlmostEqual(result["hamming_loss"], 2 / 12)
        self.assertEqual(result["empty_predictions"], 1)
        self.assertEqual(result["per_label"]["weird"]["precision"], 0)
        self.assertEqual(result["per_label"]["weird"]["fn"], 1)

    def test_missing_and_extra_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Filename mismatch"):
            evaluate({"a.jpg": {"wrong"}}, {"b.jpg": {"wrong"}})

    def test_empty_annotations_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({}, {})

    def test_conflicting_historical_predictions_are_preserved_and_counted(self):
        result = evaluate({"a.jpg": {"wrong"}}, {"a.jpg": {"wrong", "other"}})
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(result["other_rule_violations"], 1)
        self.assertEqual(result["per_label"]["other"]["fp"], 1)

    def test_duplicate_and_unknown_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.json"
            for rows, message in [
                ([{"filename": "a.jpg", "labels": ["wrong"]}] * 2, "duplicate filename"),
                ([{"filename": "a.jpg", "labels": ["unknown"]}], "invalid labels"),
                ([{"filename": "a.jpg", "labels": ["wrong", "wrong"]}], "duplicate labels"),
            ]:
                path.write_text(json.dumps(rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_records(path)

    def test_reference_annotation_rules_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.json"
            for labels in ([], ["other", "wrong"]):
                path.write_text(json.dumps([{"filename": "a.jpg", "labels": labels}]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "annotation label rules"):
                    load_records(path, annotations=True)

    def test_archives_match_independently_checked_historical_counts(self):
        report = compare(ROOT / "data/annotations.json", ROOT / "results/predictions")
        self.assertEqual(len(report["historical_runs"]), 8)
        for name, exact, empty, wrong_f1 in (
            ("test_07", 23, 14, 0.3783783783783784),
            ("test_17", 27, 0, 0.0),
            ("test_19", 25, 0, 0.20512820512820512),
        ):
            metrics = report["historical_runs"][name]
            self.assertEqual(metrics["n"], 90)
            self.assertEqual(metrics["exact_count"], exact)
            self.assertEqual(metrics["empty_predictions"], empty)
            self.assertAlmostEqual(metrics["per_label"]["wrong"]["f1"], wrong_f1)
        baseline = report["baselines"]["always_other"]
        self.assertEqual(baseline["exact_count"], 34)
        self.assertEqual(baseline["per_label"]["wrong"]["f1"], 0)


if __name__ == "__main__":
    unittest.main()
