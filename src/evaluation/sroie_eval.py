"""
SROIE 2019 evaluator: precision / recall / F1 per entity type.

Entities scored: company, date, address, total
(matches the official SROIE 2019 Task 3 entities).

Matching rules
--------------
For each (predicted, ground-truth) field, we consider it a match if any of:
  - exact equality after normalization (whitespace collapsed, lowercased)
  - the predicted string is a substring of GT (or vice versa) for `address`
  - numeric equality (within `total_tolerance`) for `total`
  - normalized date equality for `date` (best-effort across common formats)

Per-image result feeds into corpus-level micro/macro metrics.

Ground-truth loader expects the SROIE structure: one JSON file per image
under `<root>/<split>/<entities>/<image_id>.txt` (or .json) where each file
contains a single JSON object with the four entity keys.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


PathLike = Union[str, Path]
ENTITIES = ("company", "date", "address", "total")


# ---------- normalization helpers ----------
def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_total(s: Any) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        cleaned = re.sub(r"[^\d.,-]", "", str(s))
        # If both . and , are present, assume , is thousands sep.
        if "." in cleaned and "," in cleaned:
            cleaned = cleaned.replace(",", "")
        else:
            # Lone comma -> decimal sep (European style).
            cleaned = cleaned.replace(",", ".")
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _norm_date(s: Any) -> Optional[str]:
    if not s:
        return None
    raw = str(s).strip()
    formats = (
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y",
        "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    # Last-ditch: extract digits and try DDMMYYYY
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 8:
        for fmt in ("%d%m%Y", "%Y%m%d"):
            try:
                return datetime.strptime(digits, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _match(pred: Any, gt: Any, entity: str, total_tolerance: float = 0.01) -> bool:
    if entity == "total":
        p, g = _norm_total(pred), _norm_total(gt)
        if p is None or g is None:
            return False
        return abs(p - g) <= max(total_tolerance, 0.01 * abs(g))

    if entity == "date":
        p, g = _norm_date(pred), _norm_date(gt)
        if p and g:
            return p == g
        return _norm_text(pred) == _norm_text(gt) and bool(_norm_text(gt))

    p, g = _norm_text(pred), _norm_text(gt)
    if not g:
        return False
    if p == g:
        return True
    if entity == "address":
        # Partial match accepted on either side; helps when address has
        # extra trailing tokens like phone numbers or state codes.
        return (p in g or g in p) and len(p) >= max(8, len(g) // 2)
    return False


# ---------- result types ----------
@dataclass
class EntityScore:
    entity: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvaluationReport:
    per_entity: Dict[str, EntityScore]
    n_documents: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_document: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "per_entity": {k: v.to_dict() for k, v in self.per_entity.items()},
            "micro": {
                "precision": round(self.micro_precision, 4),
                "recall": round(self.micro_recall, 4),
                "f1": round(self.micro_f1, 4),
            },
            "macro": {
                "precision": round(self.macro_precision, 4),
                "recall": round(self.macro_recall, 4),
                "f1": round(self.macro_f1, 4),
            },
            "per_document": self.per_document,
        }


# ---------- ground-truth loader ----------
def load_sroie_groundtruth(gt_dir: PathLike) -> Dict[str, Dict[str, str]]:
    """Read SROIE ground-truth files into a {image_id: {entity: value}} dict.

    Accepts both `.txt` and `.json` extensions. SROIE GT files are JSON
    objects, but the official release uses the `.txt` extension.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth dir not found: {gt_dir}")

    out: Dict[str, Dict[str, str]] = {}
    for path in sorted(gt_dir.iterdir()):
        if path.suffix.lower() not in {".txt", ".json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        out[path.stem] = {k.lower(): str(v) for k, v in data.items()}
    return out


# ---------- evaluator ----------
class SROIEEvaluator:
    def __init__(self, total_tolerance: float = 0.01, entities: Iterable[str] = ENTITIES):
        self.total_tolerance = total_tolerance
        self.entities = tuple(entities)

    def evaluate(
        self,
        predictions: Mapping[str, Mapping[str, Any]],
        ground_truth: Mapping[str, Mapping[str, Any]],
    ) -> EvaluationReport:
        """Score predictions against ground truth.

        `predictions` and `ground_truth` are both keyed by image_id (the
        SROIE filename stem). Missing predictions count as false negatives.
        """
        scores: Dict[str, EntityScore] = {e: EntityScore(entity=e) for e in self.entities}
        per_doc: List[Dict[str, Any]] = []

        all_ids = set(ground_truth.keys()) | set(predictions.keys())
        for doc_id in sorted(all_ids):
            gt = ground_truth.get(doc_id, {}) or {}
            pred = predictions.get(doc_id, {}) or {}
            doc_record: Dict[str, Any] = {"document": doc_id, "fields": {}}
            for ent in self.entities:
                gt_v = gt.get(ent, "")
                pred_v = pred.get(ent, "")
                gt_present = bool(_norm_text(gt_v))
                pred_present = bool(_norm_text(pred_v))
                matched = _match(pred_v, gt_v, ent, self.total_tolerance) if (gt_present and pred_present) else False

                if matched:
                    scores[ent].tp += 1
                    outcome = "tp"
                elif pred_present and not matched:
                    scores[ent].fp += 1
                    if gt_present:
                        scores[ent].fn += 1
                    outcome = "fp"
                elif gt_present and not pred_present:
                    scores[ent].fn += 1
                    outcome = "fn"
                else:
                    outcome = "tn"

                doc_record["fields"][ent] = {
                    "gt": str(gt_v),
                    "pred": str(pred_v),
                    "outcome": outcome,
                }
            per_doc.append(doc_record)

        # Micro: pool TP/FP/FN across entities.
        total_tp = sum(s.tp for s in scores.values())
        total_fp = sum(s.fp for s in scores.values())
        total_fn = sum(s.fn for s in scores.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

        # Macro: average per-entity metrics (unweighted).
        macro_p = sum(s.precision for s in scores.values()) / len(scores)
        macro_r = sum(s.recall for s in scores.values()) / len(scores)
        macro_f1 = sum(s.f1 for s in scores.values()) / len(scores)

        return EvaluationReport(
            per_entity=scores,
            n_documents=len(all_ids),
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f1,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            per_document=per_doc,
        )
