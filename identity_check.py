from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency at runtime
    pytesseract = None
    Image = None


@dataclass
class IdentityCheckResult:
    customer_id: str
    name_match: bool
    id_match: bool
    dob_match: bool
    ocr_available: bool
    ocr_fields: Dict[str, str]
    confidence_score: float
    reasons: list[str]


def load_customers(customers_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(customers_csv, dtype=str).fillna("")
    expected_cols = {"customer_id", "name", "id_number", "dob", "country"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in customers CSV: {sorted(missing)}")
    return df


def extract_text(document_path: str | Path) -> str:
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    # Support both real images and text files for easier demo.
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if pytesseract is None or Image is None:
        raise RuntimeError(
            "pytesseract/Pillow is not available. Install dependencies or use a .txt sample file."
        )
    return pytesseract.image_to_string(Image.open(path))


def parse_identity_fields(raw_text: str) -> Dict[str, str]:
    fields = {"name": "", "id_number": "", "dob": ""}

    # Parse line-by-line first to avoid false matches like "IDENTITY".
    for line in raw_text.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if re.match(r"^name\s*:", normalized, re.I):
            fields["name"] = normalized.split(":", 1)[1].strip().upper()
        elif re.match(r"^(?:id(?:\s*number)?|nric|fin|passport)\s*:", normalized, re.I):
            fields["id_number"] = normalized.split(":", 1)[1].strip().upper()
        elif re.match(r"^(?:dob|date of birth)\s*:", normalized, re.I):
            fields["dob"] = normalized.split(":", 1)[1].strip()

    if all(fields.values()):
        return fields

    cleaned = " ".join(raw_text.split())
    # Fallback patterns for OCR outputs without clear line labels.
    id_match = re.search(
        r"(?:\bID(?:\s*Number)?\b|\bNRIC\b|\bFIN\b|\bPassport\b)\s*[:\-]?\s*([A-Z0-9 -]{5,25})",
        cleaned,
        re.I,
    )
    dob_match = re.search(r"(?:DOB|Date of Birth)\s*[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", cleaned, re.I)
    name_match = re.search(r"(?:Name)\s*[:\-]?\s*([A-Za-z\s]{3,60})", cleaned, re.I)

    if id_match and not fields["id_number"]:
        fields["id_number"] = id_match.group(1).strip().upper()
    if dob_match and not fields["dob"]:
        fields["dob"] = dob_match.group(1).strip()
    if name_match and not fields["name"]:
        fields["name"] = " ".join(name_match.group(1).split()).upper()

    return fields


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().upper().split())


def _normalize_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.strip().upper())


def _normalize_dob(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = raw.replace(",", "")
    formats = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d/%m/%Y")
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def run_identity_check(
    customer_id: str,
    customers_df: pd.DataFrame,
    doc_path: str | Path,
) -> IdentityCheckResult:
    row = customers_df.loc[customers_df["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"Unknown customer_id: {customer_id}")
    record = row.iloc[0]

    extracted_text = extract_text(doc_path)
    ocr_fields = parse_identity_fields(extracted_text)

    expected_name = _normalize_name(str(record["name"]))
    expected_id = _normalize_id(str(record["id_number"]))
    expected_dob = _normalize_dob(str(record["dob"]))

    parsed_name = _normalize_name(ocr_fields["name"])
    parsed_id = _normalize_id(ocr_fields["id_number"])
    parsed_dob = _normalize_dob(ocr_fields["dob"])

    name_match = parsed_name == expected_name if parsed_name else False
    id_match = parsed_id == expected_id if parsed_id else False
    dob_match = parsed_dob == expected_dob if parsed_dob else False

    score_components = [name_match, id_match, dob_match]
    confidence_score = round(sum(bool(x) for x in score_components) / 3 * 100, 2)

    reasons: list[str] = []
    if not name_match:
        reasons.append("Name does not match trusted customer register.")
    if not id_match:
        reasons.append("ID number does not match trusted customer register.")
    if not dob_match:
        reasons.append("DOB does not match trusted customer register.")

    return IdentityCheckResult(
        customer_id=customer_id,
        name_match=name_match,
        id_match=id_match,
        dob_match=dob_match,
        ocr_available=True,
        ocr_fields=ocr_fields,
        confidence_score=confidence_score,
        reasons=reasons,
    )
