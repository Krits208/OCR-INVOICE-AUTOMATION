"""
PaddleOCR wrapper for invoice text detection + recognition.

Wraps the official `paddleocr.PaddleOCR` class so the rest of the pipeline
works with a small typed surface (`OCRLine`, `OCRResult`) and never touches
the raw nested-list output that PaddleOCR returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]
BBox = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: BBox  # 4 corners (TL, TR, BR, BL)

    @property
    def y_center(self) -> float:
        return float(np.mean([p[1] for p in self.bbox]))

    @property
    def x_left(self) -> float:
        return float(min(p[0] for p in self.bbox))


@dataclass
class OCRResult:
    lines: List[OCRLine] = field(default_factory=list)

    def text(self, sep: str = "\n") -> str:
        """Return all text joined top-to-bottom, left-to-right."""
        ordered = sorted(self.lines, key=lambda l: (round(l.y_center / 10), l.x_left))
        return sep.join(l.text for l in ordered if l.text.strip())

    def __len__(self) -> int:
        return len(self.lines)


class PaddleOCRRunner:
    """Thin wrapper around `paddleocr.PaddleOCR`.

    The PaddleOCR import is lazy so the rest of the system (preprocessing,
    schemas, Gemini extractor) keeps working in environments where Paddle
    isn't installed (e.g. CI for the API layer only).
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_db_box_thresh: float = 0.5,
        drop_score: float = 0.5,
        show_log: bool = False,
    ):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "paddleocr is not installed. Install with `pip install paddleocr paddlepaddle` "
                "(CPU) or `pip install paddleocr paddlepaddle-gpu` (GPU)."
            ) from e

        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            det_db_box_thresh=det_db_box_thresh,
            drop_score=drop_score,
            show_log=show_log,
        )
        self.drop_score = drop_score

    def run(
        self,
        image: Union[PathLike, np.ndarray, Image.Image],
        cls: bool = True,
    ) -> OCRResult:
        arr = self._as_ndarray(image)
        raw = self._ocr.ocr(arr, cls=cls)
        return self._parse(raw)

    # ------------- helpers -------------
    def _as_ndarray(self, image: Union[PathLike, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        return np.array(Image.open(str(image)).convert("RGB"))

    def _parse(self, raw) -> OCRResult:
        """PaddleOCR returns either:
            - [[line, line, ...]]              (single-image mode)
            - [None]                            (empty image)
        Where each `line` is `[bbox, (text, conf)]` with bbox a 4-point list.
        """
        result = OCRResult()
        if not raw:
            return result
        page = raw[0]
        if not page:
            return result
        for det in page:
            try:
                bbox_pts, (text, conf) = det
                if conf < self.drop_score or not text:
                    continue
                bbox: BBox = tuple(tuple(int(round(v)) for v in pt) for pt in bbox_pts)  # type: ignore[assignment]
                result.lines.append(OCRLine(text=str(text), confidence=float(conf), bbox=bbox))
            except (ValueError, TypeError):
                # Skip malformed detection rows rather than failing the page.
                continue
        return result
