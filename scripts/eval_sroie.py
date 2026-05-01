"""
Evaluate the end-to-end pipeline on the SROIE 2019 benchmark.

Usage:
    python scripts/eval_sroie.py \
        --images   data/sroie/images \
        --gt       data/sroie/entities \
        --out      output/sroie_report.json \
        --mode     vision   # or "ocr" | "hybrid"
        --limit    50       # optional: cap dataset size for a quick run

The script writes a per-entity precision/recall/F1 report (JSON) and prints
a human-readable summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as a script from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.evaluation import SROIEEvaluator, load_sroie_groundtruth  # noqa: E402
from src.gemini_extractor.structured import StructuredGeminiExtractor  # noqa: E402
from src.pipeline import InvoicePipeline  # noqa: E402
from src.preprocessing import OpenCVPreprocessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SROIE evaluation runner")
    p.add_argument("--images", required=True, help="Directory of receipt images")
    p.add_argument("--gt", required=True, help="Directory of ground-truth JSON entity files")
    p.add_argument("--out", default="output/sroie_report.json", help="Path to write JSON report")
    p.add_argument("--mode", default="vision", choices=["vision", "ocr", "hybrid"])
    p.add_argument("--limit", type=int, default=None, help="Cap dataset size")
    p.add_argument("--ocr-lang", default="en")
    p.add_argument("--model", default="gemini-flash-latest")
    return p.parse_args()


def collect_images(image_dir: Path, ids: List[str]) -> Dict[str, Path]:
    """Match GT ids to actual image files (any common extension)."""
    by_stem: Dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            by_stem[path.stem] = path
    return {i: by_stem[i] for i in ids if i in by_stem}


def main() -> int:
    args = parse_args()
    load_dotenv()

    image_dir = Path(args.images)
    gt_dir = Path(args.gt)

    print(f"Loading ground truth from {gt_dir} ...")
    gt = load_sroie_groundtruth(gt_dir)
    print(f"  -> {len(gt)} ground-truth records")

    image_paths = collect_images(image_dir, list(gt.keys()))
    print(f"Found {len(image_paths)} matching images in {image_dir}")
    if args.limit:
        image_paths = dict(list(image_paths.items())[: args.limit])
        gt = {k: gt[k] for k in image_paths}
        print(f"  -> capped to {len(image_paths)} images")

    # Build pipeline
    gemini = StructuredGeminiExtractor(model=args.model)
    ocr_runner = None
    if args.mode in {"ocr", "hybrid"}:
        from src.paddle_ocr import PaddleOCRRunner
        ocr_runner = PaddleOCRRunner(lang=args.ocr_lang)

    pipeline = InvoicePipeline(
        gemini_extractor=gemini,
        preprocessor=OpenCVPreprocessor(),
        ocr_runner=ocr_runner,
    )

    # Run extractions
    predictions: Dict[str, Dict[str, str]] = {}
    for i, (doc_id, path) in enumerate(image_paths.items(), 1):
        print(f"[{i}/{len(image_paths)}] {doc_id}")
        try:
            result = pipeline.extract(path, mode=args.mode)
            predictions[doc_id] = {k: str(v) for k, v in result.invoice.items()}
        except Exception as e:
            print(f"  ! failed: {e}")
            predictions[doc_id] = {}

    # Score
    evaluator = SROIEEvaluator()
    report = evaluator.evaluate(predictions, gt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    # Pretty print
    print("\n=== Per-entity metrics ===")
    print(f"{'entity':<12} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    for ent, score in report.per_entity.items():
        print(f"{ent:<12} {score.precision:>8.4f} {score.recall:>8.4f} {score.f1:>8.4f} "
              f"{score.tp:>6} {score.fp:>6} {score.fn:>6}")
    print(f"\nMicro F1: {report.micro_f1:.4f}    Macro F1: {report.macro_f1:.4f}    "
          f"({report.n_documents} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
