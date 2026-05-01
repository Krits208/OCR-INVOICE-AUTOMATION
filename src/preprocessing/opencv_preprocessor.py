"""
OpenCV-based image preprocessing for invoice OCR.

Pipeline stages:
    grayscale -> denoise -> deskew (Hough / projection-profile) -> binarize (Otsu/adaptive) -> morphology

The output is a clean binary image suitable for PaddleOCR text detection
and recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


PathLike = Union[str, Path]


@dataclass
class PreprocessResult:
    image: np.ndarray            # final preprocessed image (BGR or single-channel)
    gray: np.ndarray             # intermediate grayscale
    binary: np.ndarray           # binarized image
    deskew_angle: float          # angle (deg) used to deskew; 0.0 if no rotation
    original_shape: Tuple[int, int]


class OpenCVPreprocessor:
    """OpenCV preprocessor: deskew + binarize for invoice OCR."""

    def __init__(
        self,
        max_dim: int = 2000,
        binarize: str = "adaptive",   # "otsu" | "adaptive" | "none"
        denoise: bool = True,
        deskew: bool = True,
        deskew_method: str = "hough", # "hough" | "projection"
        adaptive_block_size: int = 35,
        adaptive_C: int = 11,
    ):
        if binarize not in {"otsu", "adaptive", "none"}:
            raise ValueError(f"binarize must be otsu|adaptive|none, got {binarize}")
        if deskew_method not in {"hough", "projection"}:
            raise ValueError(f"deskew_method must be hough|projection, got {deskew_method}")
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1  # must be odd

        self.max_dim = max_dim
        self.binarize = binarize
        self.denoise = denoise
        self.deskew = deskew
        self.deskew_method = deskew_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_C = adaptive_C

    # -------- public API --------
    def process(self, image: Union[PathLike, np.ndarray, Image.Image]) -> PreprocessResult:
        bgr = self._load(image)
        original_shape = bgr.shape[:2]
        bgr = self._resize_if_large(bgr)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        angle = 0.0
        if self.deskew:
            angle = self._estimate_skew(gray)
            if abs(angle) > 0.1:
                gray = self._rotate(gray, angle)
                bgr = self._rotate(bgr, angle)

        binary = self._binarize(gray)

        return PreprocessResult(
            image=binary if self.binarize != "none" else gray,
            gray=gray,
            binary=binary,
            deskew_angle=float(angle),
            original_shape=original_shape,
        )

    def to_pil(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    # -------- internals --------
    def _load(self, image: Union[PathLike, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        path = str(image)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return bgr

    def _resize_if_large(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m <= self.max_dim:
            return bgr
        scale = self.max_dim / m
        return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _estimate_skew(self, gray: np.ndarray) -> float:
        if self.deskew_method == "hough":
            return self._estimate_skew_hough(gray)
        return self._estimate_skew_projection(gray)

    def _estimate_skew_hough(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 60, 180, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120, minLineLength=gray.shape[1] // 4, maxLineGap=10)
        if lines is None:
            return 0.0

        angles = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            ang = np.degrees(np.arctan2(dy, dx))
            # Keep near-horizontal lines (text baselines)
            if -45 < ang < 45:
                angles.append(ang)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def _estimate_skew_projection(self, gray: np.ndarray) -> float:
        # Try a small angle search; pick the one that maximizes row-wise variance.
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        best_angle, best_score = 0.0, -1.0
        for ang in np.arange(-5.0, 5.5, 0.5):
            rotated = self._rotate(thresh, ang)
            hist = rotated.sum(axis=1, dtype=np.float32)
            score = float(((hist[1:] - hist[:-1]) ** 2).sum())
            if score > best_score:
                best_score, best_angle = score, float(ang)
        return best_angle

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        border = 255 if img.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        if self.binarize == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        if self.binarize == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_C,
            )
        return gray  # "none"


def preprocess_for_ocr(
    image: Union[PathLike, np.ndarray, Image.Image],
    **kwargs,
) -> PreprocessResult:
    """Convenience wrapper around OpenCVPreprocessor.process."""
    return OpenCVPreprocessor(**kwargs).process(image)
