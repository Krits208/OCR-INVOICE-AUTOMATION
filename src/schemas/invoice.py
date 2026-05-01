"""
Pydantic schemas for invoice extraction.

`SROIEInvoice` matches the four entities of the SROIE 2019 challenge
(company, date, address, total) and is what the evaluator scores against.
`RichInvoice` is the full extraction schema (line items + tax info).
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


SROIE_FIELDS = ("company", "date", "address", "total")


class SROIEInvoice(BaseModel):
    """SROIE 2019 entity-level schema (company / date / address / total)."""

    model_config = {"extra": "ignore", "str_strip_whitespace": True}

    company: str = Field(..., description="Seller / merchant name as printed on the receipt.")
    date: str = Field(..., description="Receipt date. Prefer DD/MM/YYYY, but echo what's printed.")
    address: str = Field(..., description="Seller address.")
    total: str = Field(..., description="Final amount paid (post-tax). String, currency-free if possible.")

    @field_validator("company", "date", "address", "total", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if v is None:
            return ""
        return str(v).strip()


class InvoiceLineItem(BaseModel):
    model_config = {"extra": "ignore", "str_strip_whitespace": True}

    description: str = Field(..., description="Product or service description.")
    quantity: float = Field(default=1.0, ge=0)
    unit_price: Optional[float] = Field(default=None, ge=0)
    line_total: Optional[float] = Field(default=None, ge=0)


class RichInvoice(SROIEInvoice):
    """Extended schema that adds line items, tax, and currency."""

    currency: Optional[str] = Field(default=None, description="ISO-4217 code if present (e.g. USD, MYR, VND).")
    subtotal: Optional[float] = Field(default=None, ge=0)
    tax: Optional[float] = Field(default=None, ge=0)
    line_items: List[InvoiceLineItem] = Field(default_factory=list)
    parsed_date: Optional[date] = Field(default=None, description="ISO-parsed date when unambiguous.")
