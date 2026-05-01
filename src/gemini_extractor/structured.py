"""
Gemini-based structured invoice extraction with Pydantic schema enforcement.

Two entry points:
  - `extract_from_image(image)` for the vision pipeline (image -> Gemini)
  - `extract_from_text(text)`   for the OCR-then-LLM pipeline (PaddleOCR -> Gemini)

The model is asked for JSON-only output, parsed, and validated against a
Pydantic schema. Default schema is `SROIEInvoice`, but any subclass works.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

from PIL import Image
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.schemas.invoice import SROIEInvoice


T = TypeVar("T", bound=BaseModel)
PathLike = Union[str, Path]

_VISION_PROMPT = (
    "You are an information extraction system for retail receipts and invoices.\n"
    "Look at the image and extract these fields: company, date, address, total.\n"
    "Rules:\n"
    "  - Return ONLY a single JSON object. No prose, no markdown fences.\n"
    "  - Use empty string \"\" for fields that are clearly missing.\n"
    "  - `total` is the final paid amount; strip currency symbols where possible.\n"
    "  - `date` should preserve the printed format.\n"
)

_TEXT_PROMPT = (
    "You are an information extraction system for retail receipts and invoices.\n"
    "Below is the OCR transcription of a receipt. Extract these fields: "
    "company, date, address, total.\n"
    "Rules:\n"
    "  - Return ONLY a single JSON object. No prose, no markdown fences.\n"
    "  - Use empty string \"\" for fields that are clearly missing.\n"
    "  - `total` is the final paid amount; strip currency symbols where possible.\n\n"
    "OCR_TEXT:\n{ocr_text}\n"
)


class StructuredGeminiExtractor:
    """Calls Gemini and validates the response against a Pydantic schema."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-flash-latest",
        schema: Type[BaseModel] = SROIEInvoice,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError("google-generativeai is required: pip install google-generativeai") from e

        self._genai = genai
        self.schema = schema
        self.model_name = model

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set; pass api_key= or export the env var.")
        genai.configure(api_key=api_key)

        self._gen_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )
        self._model = genai.GenerativeModel(model, generation_config=self._gen_config)

    # ------------- public API -------------
    def extract_from_image(
        self,
        image: Union[PathLike, Image.Image],
        extra_prompt: Optional[str] = None,
    ) -> BaseModel:
        pil = self._as_pil(image)
        prompt = _VISION_PROMPT + self._schema_hint()
        if extra_prompt:
            prompt += f"\n\nADDITIONAL_INSTRUCTIONS:\n{extra_prompt}"
        response = self._model.generate_content([prompt, pil])
        return self._parse_and_validate(response.text)

    def extract_from_text(self, ocr_text: str, extra_prompt: Optional[str] = None) -> BaseModel:
        prompt = _TEXT_PROMPT.format(ocr_text=ocr_text) + self._schema_hint()
        if extra_prompt:
            prompt += f"\n\nADDITIONAL_INSTRUCTIONS:\n{extra_prompt}"
        response = self._model.generate_content(prompt)
        return self._parse_and_validate(response.text)

    # ------------- internals -------------
    def _schema_hint(self) -> str:
        # Hand the model the JSON schema so it knows the exact field names.
        try:
            schema_dict = self.schema.model_json_schema()
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            schema_dict = self.schema.schema()
        return f"\n\nJSON_SCHEMA:\n{json.dumps(schema_dict, indent=2)}\n"

    def _parse_and_validate(self, raw_text: str) -> BaseModel:
        payload = self._extract_json(raw_text)
        try:
            return self.schema.model_validate(payload)
        except ValidationError as e:
            logger.warning(f"Schema validation failed; raw payload: {payload}")
            raise ValueError(f"Gemini response failed schema validation: {e}") from e

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Pull a JSON object out of Gemini's response.

        Handles:
          - clean JSON
          - markdown-fenced JSON (```json ... ```)
          - text with leading/trailing prose around a JSON block
        """
        if not text:
            raise ValueError("Empty response from Gemini")
        text = text.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find first { ... last }
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"No JSON object found in Gemini response: {text[:200]}")
            return json.loads(text[start : end + 1])

    @staticmethod
    def _as_pil(image: Union[PathLike, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(str(image)).convert("RGB")
