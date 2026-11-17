"""Strict alignment and dependency-free multi-label evaluation."""
from __future__ import annotations

import json
from math import fsum
from pathlib import Path

LABELS = ("wrong", "weird", "unrealistic", "other")


def load_records(path: str | Path, *, annotations: bool = False) -> dict[str, set[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    records = {}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} must be an object")
        filename, labels = row.get("filename"), row.get("labels")
        if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
            raise ValueError(f"{path}: row {index} has an invalid filename")
        if filename in records:
            raise ValueError(f"{path}: duplicate filename {filename}")
        if not isinstance(labels, list) or any(not isinstance(label, str) or label not in LABELS for label in labels):
            raise ValueError(f"{path}: row {index} has invalid labels")
        if len(labels) != len(set(labels)):
            raise ValueError(f"{path}: row {index} has duplicate labels")
        if annotations and (not labels or ("other" in labels and len(labels) != 1)):
            raise ValueError(f"{path}: row {index} violates annotation label rules")
        records[filename] = set(labels)
    return records


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(truth: dict[str, set[str]], predictions: dict[str, set[str]]) -> dict:
    """Require a complete one-to-one match; keep historical invalid combinations."""
    if not truth:
        raise ValueError("Annotations must not be empty")
    missing, extra = set(truth) - set(predictions), set(predictions) - set(truth)
    if missing or extra:
        raise ValueError(f"Filename mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, records in (("annotations", truth), ("predictions", predictions)):
        if any(not isinstance(labels, set) or not labels <= set(LABELS) for labels in records.values()):
            raise ValueError(f"Invalid label sets in {name}")
    if any(not labels or ("other" in labels and len(labels) != 1) for labels in truth.values()):
        raise ValueError("Annotations violate nonempty / exclusive-other label rules")
    per_label = {}
    for label in LABELS:
        tp = sum(label in truth[k] and label in predictions[k] for k in truth)
        fp = sum(label not in truth[k] and label in predictions[k] for k in truth)
        fn = sum(label in truth[k] and label not in predictions[k] for k in truth)
        tn = len(truth) - tp - fp - fn
        per_label[label] = dict(
            support=tp + fn, predicted=tp + fp, tp=tp, fp=fp, fn=fn, tn=tn,
            precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
            f1=_ratio(2 * tp, 2 * tp + fp + fn),
        )
    totals = {key: sum(row[key] for row in per_label.values()) for key in ("tp", "fp", "fn", "tn")}
    exact_count = sum(truth[k] == predictions[k] for k in truth)
    return dict(
        n=len(truth), exact_count=exact_count, exact_match=exact_count / len(truth),
        macro_f1=fsum(row["f1"] for row in per_label.values()) / len(LABELS),
        micro_precision=_ratio(totals["tp"], totals["tp"] + totals["fp"]),
        micro_recall=_ratio(totals["tp"], totals["tp"] + totals["fn"]),
        micro_f1=_ratio(2 * totals["tp"], 2 * totals["tp"] + totals["fp"] + totals["fn"]),
        hamming_loss=(totals["fp"] + totals["fn"]) / (len(truth) * len(LABELS)),
        empty_predictions=sum(not labels for labels in predictions.values()),
        other_rule_violations=sum("other" in labels and len(labels) > 1 for labels in predictions.values()),
        per_label=per_label,
    )


def compare(annotation_path: str | Path, prediction_dir: str | Path) -> dict:
    truth = load_records(annotation_path, annotations=True)
    files = sorted(Path(prediction_dir).glob("test_*.json"))
    if not files:
        raise ValueError(f"No test_*.json predictions found in {prediction_dir}")
    runs = {path.stem: evaluate(truth, load_records(path)) for path in files}
    baselines = {f"always_{label}": evaluate(truth, {k: {label} for k in truth}) for label in LABELS}
    return dict(
        evaluation_policy=dict(undefined_precision_recall_f1=0, macro_labels=list(LABELS),
                               historical_predictions="preserved; no automatic label repair"),
        dataset=dict(n=len(truth), label_counts={label: sum(label in labels for labels in truth.values()) for label in LABELS}),
        baselines=baselines, historical_runs=runs,
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# Recomputed historical results", "",
        "All results use the same 90-image exploratory dataset. No new model calls were made.",
        "Undefined precision/recall/F1 = 0. Historical parsed predictions are unchanged.", "",
        "| Run | Exact match | Macro-F1 | Micro-F1 | wrong F1 | Empty | Other rule violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in {**report["baselines"], **report["historical_runs"]}.items():
        lines.append(f"| {name} | {metrics['exact_match']:.2%} | {metrics['macro_f1']:.4f} | "
                     f"{metrics['micro_f1']:.4f} | {metrics['per_label']['wrong']['f1']:.4f} | "
                     f"{metrics['empty_predictions']} | {metrics['other_rule_violations']} |")
    lines += ["", "Constant-label baselines are descriptive comparisons on this dataset, not held-out results.",
              "See summary.json for per-label precision, recall, F1 and TP/FP/FN/TN counts.", ""]
    return "\n".join(lines)
