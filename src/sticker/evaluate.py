"""用人工标签评估贴片报告的车牌级与字符槽级指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Annotation:
    image_id: str
    decision: str
    suspicious_slots: frozenset[int]
    slot_count: int | None = None


def _parse_slots(value: str) -> frozenset[int]:
    stripped = value.strip()
    if not stripped:
        return frozenset()
    return frozenset(int(item) for item in re.split(r"[;,\s]+", stripped) if item)


def read_annotations(path: str | Path) -> dict[str, Annotation]:
    annotations: dict[str, Annotation] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "decision", "suspicious_slots"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"标注 CSV 必须包含列：{sorted(required)}")
        for row in reader:
            image_id = row["image_id"].strip()
            decision = row["decision"].strip()
            if not image_id:
                continue
            if decision not in {"suspicious", "clear", "unassessable"}:
                raise ValueError(f"{image_id} 的 decision 非法：{decision}")
            slots = _parse_slots(row["suspicious_slots"])
            raw_slot_count = (row.get("slot_count") or "").strip()
            slot_count = int(raw_slot_count) if raw_slot_count else None
            if decision == "clear" and slots:
                raise ValueError(f"{image_id} 标为 clear 但仍给出了 suspicious_slots")
            if decision == "unassessable" and slots:
                raise ValueError(f"{image_id} 标为 unassessable 但仍给出了 suspicious_slots")
            if slot_count is not None and (slot_count not in {7, 8} or any(slot > slot_count or slot < 1 for slot in slots)):
                raise ValueError(f"{image_id} 的槽位编号超出 1..{slot_count}")
            annotations[image_id] = Annotation(image_id, decision, slots, slot_count)
    return annotations


def read_reports(root: str | Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for path in sorted(Path(root).rglob("report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports[path.parent.name] = payload
    for path in sorted(Path(root).rglob("batch_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample in payload.get("samples", []):
            if sample.get("status") != "success" or not sample.get("sample_id"):
                continue
            suspected = []
            for item in sample.get("tampered_characters", []):
                if item.get("slot") is not None:
                    suspected.append(
                        {
                            "slot": int(item["slot"]),
                            "confidence_level": "high",
                        }
                    )
            reports[str(sample["sample_id"])] = {
                "decision": sample.get("decision", "unassessable"),
                "suspected_characters": suspected,
                "source": str(path),
            }
    return reports


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate_reports(annotations: dict[str, Annotation], reports: dict[str, dict]) -> dict:
    missing_reports = sorted(set(annotations) - set(reports))
    unlabelled_reports = sorted(set(reports) - set(annotations))
    plate_tp = plate_fp = plate_fn = plate_tn = 0
    slot_tp = slot_fp = slot_fn = 0
    slot_tn = slot_total = 0
    predicted_unassessable = 0
    truth_positive_count = truth_negative_count = 0
    answered_plate_tp = answered_plate_fp = answered_plate_fn = answered_plate_tn = 0
    evaluated = 0
    per_image = []
    for image_id in sorted(set(annotations) & set(reports)):
        annotation = annotations[image_id]
        report = reports[image_id]
        predicted_decision = str(report.get("decision", "unassessable"))
        predicted_slots = frozenset(
            int(item.get("slot"))
            for item in report.get("suspected_characters", [])
            if item.get("confidence_level") == "high" and item.get("slot") is not None
        )
        if predicted_decision == "unassessable":
            predicted_unassessable += 1
        if annotation.decision == "unassessable":
            per_image.append({"image_id": image_id, "excluded": "ground_truth_unassessable"})
            continue
        evaluated += 1
        truth_positive = annotation.decision == "suspicious"
        truth_positive_count += int(truth_positive)
        truth_negative_count += int(not truth_positive)
        predicted_positive = predicted_decision == "suspicious"
        if truth_positive and predicted_positive:
            plate_tp += 1
        elif not truth_positive and predicted_positive:
            plate_fp += 1
        elif truth_positive and not predicted_positive:
            plate_fn += 1
        else:
            plate_tn += 1
        if predicted_decision != "unassessable":
            if truth_positive and predicted_positive:
                answered_plate_tp += 1
            elif not truth_positive and predicted_positive:
                answered_plate_fp += 1
            elif truth_positive:
                answered_plate_fn += 1
            else:
                answered_plate_tn += 1
        slot_tp += len(predicted_slots & annotation.suspicious_slots)
        slot_fp += len(predicted_slots - annotation.suspicious_slots)
        slot_fn += len(annotation.suspicious_slots - predicted_slots)
        if annotation.slot_count is not None:
            slot_total += annotation.slot_count
            slot_tn += annotation.slot_count - len(predicted_slots | annotation.suspicious_slots)
        per_image.append(
            {
                "image_id": image_id,
                "ground_truth_decision": annotation.decision,
                "predicted_decision": predicted_decision,
                "ground_truth_slots": sorted(annotation.suspicious_slots),
                "predicted_slots": sorted(predicted_slots),
            }
        )
    plate_precision = _safe_div(plate_tp, plate_tp + plate_fp)
    plate_recall = _safe_div(plate_tp, plate_tp + plate_fn)
    slot_precision = _safe_div(slot_tp, slot_tp + slot_fp)
    slot_recall = _safe_div(slot_tp, slot_tp + slot_fn)
    plate_accuracy = _safe_div(plate_tp + plate_tn, evaluated)
    plate_specificity = _safe_div(plate_tn, plate_tn + plate_fp)
    answered_total = answered_plate_tp + answered_plate_fp + answered_plate_fn + answered_plate_tn
    answered_precision = _safe_div(answered_plate_tp, answered_plate_tp + answered_plate_fp)
    answered_recall = _safe_div(answered_plate_tp, answered_plate_tp + answered_plate_fn)
    answered_accuracy = _safe_div(answered_plate_tp + answered_plate_tn, answered_total)
    return {
        "schema_version": 1,
        "counts": {
            "annotations": len(annotations),
            "reports": len(reports),
            "evaluated_assessable_ground_truth": evaluated,
            "ground_truth_positive": truth_positive_count,
            "ground_truth_negative": truth_negative_count,
            "predicted_unassessable": predicted_unassessable,
            "missing_reports": len(missing_reports),
            "unlabelled_reports": len(unlabelled_reports),
        },
        "plate_level": {
            "tp": plate_tp,
            "fp": plate_fp,
            "fn": plate_fn,
            "tn": plate_tn,
            "precision": plate_precision,
            "recall": plate_recall,
            "f1": _f1(plate_precision, plate_recall),
            "accuracy": plate_accuracy,
            "specificity": plate_specificity,
            "precision_ci95_wilson": _wilson_interval(plate_tp, plate_tp + plate_fp),
            "recall_ci95_wilson": _wilson_interval(plate_tp, plate_tp + plate_fn),
            "accuracy_ci95_wilson": _wilson_interval(plate_tp + plate_tn, evaluated),
            "unassessable_rate": _safe_div(predicted_unassessable, len(set(annotations) & set(reports))),
        },
        "plate_level_answered_only": {
            "evaluated": answered_total,
            "coverage": _safe_div(answered_total, evaluated),
            "tp": answered_plate_tp,
            "fp": answered_plate_fp,
            "fn": answered_plate_fn,
            "tn": answered_plate_tn,
            "precision": answered_precision,
            "recall": answered_recall,
            "f1": _f1(answered_precision, answered_recall),
            "accuracy": answered_accuracy,
            "accuracy_ci95_wilson": _wilson_interval(
                answered_plate_tp + answered_plate_tn, answered_total
            ),
        },
        "character_slot_level": {
            "tp": slot_tp,
            "fp": slot_fp,
            "fn": slot_fn,
            "precision": slot_precision,
            "recall": slot_recall,
            "f1": _f1(slot_precision, slot_recall),
            "tn": slot_tn if slot_total else None,
            "total_slots": slot_total if slot_total else None,
            "accuracy": _safe_div(slot_tp + slot_tn, slot_total) if slot_total else None,
        },
        "missing_report_ids": missing_reports,
        "unlabelled_report_ids": unlabelled_reports,
        "per_image": per_image,
        "metric_note": (
            "仅使用人工标注计算；uncertain_candidates 不计为阳性字符槽。"
            "plate_level 将 unassessable 视为非阳性，需同时查看 answered_only coverage；"
            "30张仅适合作为流程试标，区间可能很宽。"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="人工标注 CSV")
    parser.add_argument("reports", type=Path, help="包含 report.json 的输出根目录")
    parser.add_argument("--output", type=Path, help="可选的指标 JSON 输出路径")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = evaluate_reports(read_annotations(args.annotations), read_reports(args.reports))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
