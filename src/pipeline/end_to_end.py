"""
End-to-end invoice extraction pipeline.

Stages:
  1. OpenCV preprocessing (resize, deskew, binarize)
  2. PaddleOCR detection + recognition  (optional; vision-only mode skips it)
  3. Gemini structured extraction       (Pydantic-validated)

Two extraction modes:
  - "ocr"    : preprocess -> PaddleOCR -> Gemini text mode -> validated invoice
  - "vision" : preprocess -> Gemini vision mode -> validated invoice
  - "hybrid" : run both, prefer OCR text but fall back to vision if validation fails
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger
from pydantic import BaseModel

from src.preprocessing import OpenCVPreprocessor
from src.schemas.invoice import SROIEInvoice


PathLike = Union[str, Path]


@dataclass
class ExtractionOutput:
    """One image's extraction result."""
    image_path: str
    invoice: Dict[str, Any]            # validated dict
    ocr_text: Optional[str] = None
    deskew_angle: float = 0.0
    mode: str = "vision"
    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InvoicePipeline:
    """End-to-end invoice extractor.

    The OCR runner and Gemini extractor are passed in (dependency injection)
    so callers can swap them out for tests, or skip Paddle entirely in
    vision-only mode.
    """

    def __init__(
        self,
        gemini_extractor,                          # StructuredGeminiExtractor
        preprocessor: Optional[OpenCVPreprocessor] = None,
        ocr_runner=None,                            # PaddleOCRRunner | None
        schema: Type[BaseModel] = SROIEInvoice,
    ):
        self.gemini = gemini_extractor
        self.preprocessor = preprocessor or OpenCVPreprocessor()
        self.ocr = ocr_runner
        self.schema = schema

    def extract(
        self,
        image: PathLike,
        mode: str = "vision",
    ) -> ExtractionOutput:
        if mode not in {"vision", "ocr", "hybrid"}:
            raise ValueError(f"mode must be vision|ocr|hybrid, got {mode}")
        if mode in {"ocr", "hybrid"} and self.ocr is None:
            raise ValueError("ocr_runner is required for mode='ocr' or 'hybrid'")

        start = time.perf_counter()
        path_str = str(image)
        errors: List[str] = []

        # Stage 1: preprocess
        pre = self.preprocessor.process(path_str)
        pil = self.preprocessor.to_pil(pre.image)

        # Stage 2 + 3
        invoice_obj = None
        ocr_text = None

        if mode in {"ocr", "hybrid"}:
            try:
                # Run OCR on the preprocessed image (use grayscale or binary;
                # PaddleOCR is happy with BGR/RGB so we feed the deskewed binary as RGB).
                ocr_result = self.ocr.run(pil)
                ocr_text = ocr_result.text()
                if not ocr_text.strip():
                    raise ValueError("PaddleOCR returned no text")
                invoice_obj = self.gemini.extract_from_text(ocr_text)
            except Exception as e:
                logger.warning(f"OCR-mode extraction failed: {e}")
                errors.append(f"ocr_mode: {e}")
                if mode == "ocr":
                    elapsed = time.perf_counter() - start
                    return ExtractionOutput(
                        image_path=path_str,
                        invoice={},
                        ocr_text=ocr_text,
                        deskew_angle=pre.deskew_angle,
                        mode=mode,
                        elapsed_seconds=elapsed,
                        errors=errors,
                    )

        if invoice_obj is None:  # vision or hybrid-fallback
            try:
                invoice_obj = self.gemini.extract_from_image(pil)
            except Exception as e:
                logger.error(f"Vision-mode extraction failed: {e}")
                errors.append(f"vision_mode: {e}")
                elapsed = time.perf_counter() - start
                return ExtractionOutput(
                    image_path=path_str,
                    invoice={},
                    ocr_text=ocr_text,
                    deskew_angle=pre.deskew_angle,
                    mode=mode,
                    elapsed_seconds=elapsed,
                    errors=errors,
                )

        elapsed = time.perf_counter() - start
        return ExtractionOutput(
            image_path=path_str,
            invoice=invoice_obj.model_dump(),
            ocr_text=ocr_text,
            deskew_angle=pre.deskew_angle,
            mode=mode,
            elapsed_seconds=elapsed,
            errors=errors,
        )

    def extract_batch(
        self,
        images: List[PathLike],
        mode: str = "vision",
        continue_on_error: bool = True,
    ) -> List[ExtractionOutput]:
        out: List[ExtractionOutput] = []
        for i, img in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}] {img}")
            try:
                out.append(self.extract(img, mode=mode))
            except Exception as e:
                if not continue_on_error:
                    raise
                logger.error(f"Failed on {img}: {e}")
                out.append(ExtractionOutput(
                    image_path=str(img),
                    invoice={},
                    mode=mode,
                    errors=[str(e)],
                ))
        return out
