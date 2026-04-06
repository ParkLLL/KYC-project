from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


@dataclass
class NewsCheckResult:
    customer_id: str
    customer_name: str
    matched_alerts: list[dict[str, Any]]
    news_risk_score: float
    reasons: list[str]


SEVERITY_WEIGHTS = {
    "low": 20,
    "medium": 50,
    "high": 80,
}

REPORT_RISK_KEYWORDS = {
    "financial loss due to": 8,
    "lawsuit": 8,
    "litigation": 8,
    "penalty": 7,
    "fine": 7,
    "regulatory action": 8,
    "non-compliance": 9,
    "compliance breach": 9,
    "money laundering": 10,
    "fraud": 9,
    "sanction": 9,
    "investigation": 7,
    "bribery": 9,
    "corruption": 9,
    "whistleblowing": 7,
    "breach": 6,
}


def load_adverse_media_db(path: str | Path) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Adverse media file must contain a list of records.")
    return records


def _extract_pdf_text(pdf_path: str | Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Please install requirements first.")
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks)


def screen_entity_from_reports(
    entity_name: str,
    report_files: list[str | Path],
) -> tuple[float, list[str], list[dict[str, Any]]]:
    if not report_files:
        return 0.0, ["No annual reports provided for report screening."], []

    entity = entity_name.strip().lower()
    entity_flat = re.sub(r"[^a-z0-9]", "", entity)
    if not entity:
        return 0.0, ["Entity name is empty, skipped report screening."], []

    total_weighted_hits = 0
    findings: list[str] = []
    report_alerts: list[dict[str, Any]] = []

    for report in report_files:
        report_path = Path(report)
        if not report_path.exists():
            findings.append(f"Report not found: {report_path.name}")
            continue
        try:
            text = _extract_pdf_text(report_path).lower()
        except Exception as exc:
            findings.append(f"Failed to parse report {report_path.name}: {exc}")
            continue

        file_name_hit = entity_flat and entity_flat in re.sub(r"[^a-z0-9]", "", report_path.stem.lower())
        text_hit = entity in text if entity else False
        if not file_name_hit and not text_hit:
            # If entity match is weak, still continue scanning because user-provided
            # annual reports are often already filtered to the target company.
            findings.append(f"{report_path.name}: entity mention not explicit, scanned as provided source.")

        per_report_hits: list[tuple[str, int]] = []
        for keyword, weight in REPORT_RISK_KEYWORDS.items():
            count = len(re.findall(re.escape(keyword), text))
            if count > 0:
                per_report_hits.append((keyword, count))
                # Use presence-based scoring to avoid inflated counts from repetitive report sections.
                total_weighted_hits += weight + min(3, int(math.log2(count + 1)))

        if per_report_hits:
            top_hits = sorted(per_report_hits, key=lambda x: x[1], reverse=True)[:4]
            hit_text = ", ".join([f"{k} x{v}" for k, v in top_hits])
            findings.append(f"{report_path.name}: {hit_text}")
            report_alerts.append(
                {
                    "source": report_path.name,
                    "severity": "medium" if total_weighted_hits < 60 else "high",
                    "summary": f"Keyword risk signals found for entity '{entity_name}'.",
                }
            )

    if total_weighted_hits == 0:
        if findings:
            return 0.0, findings, report_alerts
        return 0.0, ["No high-risk signals found in annual reports."], report_alerts

    # Normalize weighted hits into an intuitive 0-100 scale.
    report_risk_score = float(min(70, round((total_weighted_hits / max(1, len(report_files))) * 1.2, 2)))
    return report_risk_score, findings, report_alerts


def run_news_check(
    customer_id: str,
    customer_name: str,
    adverse_media_db: list[dict[str, Any]],
    entity_name: str | None = None,
    report_files: list[str | Path] | None = None,
) -> NewsCheckResult:
    normalized_name = customer_name.strip().upper()
    hits: list[dict[str, Any]] = []

    for item in adverse_media_db:
        item_name = str(item.get("name", "")).strip().upper()
        item_customer_id = str(item.get("customer_id", "")).strip()
        if item_customer_id == customer_id or (item_name and item_name == normalized_name):
            hits.append(item)

    reasons: list[str] = []
    base_news_score = 0.0
    if not hits:
        reasons.append("No adverse media hit found.")
    else:
        weighted_scores = []
        for hit in hits:
            sev = str(hit.get("severity", "low")).lower()
            score = SEVERITY_WEIGHTS.get(sev, 20)
            weighted_scores.append(score)
            reasons.append(f"Hit: {hit.get('source', 'unknown')} ({sev}) - {hit.get('summary', '')}")
        # Cap score at 100 to keep scale interpretable.
        base_news_score = float(min(100, round(sum(weighted_scores) / len(weighted_scores), 2)))

    report_score = 0.0
    report_reasons: list[str] = []
    report_alerts: list[dict[str, Any]] = []
    if entity_name and report_files:
        report_score, report_reasons, report_alerts = screen_entity_from_reports(entity_name, report_files)
        reasons.extend(report_reasons)

    combined_score = float(round(max(base_news_score, report_score), 2))
    return NewsCheckResult(
        customer_id=customer_id,
        customer_name=customer_name,
        matched_alerts=hits + report_alerts,
        news_risk_score=combined_score,
        reasons=reasons,
    )
