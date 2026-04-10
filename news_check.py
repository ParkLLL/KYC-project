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


def screen_entity_from_reports_rag(
    entity_name: str,
    vector_store,
) -> tuple[float, list[str], list[dict[str, Any]]]:
    """RAG-based screening: semantically retrieve relevant chunks, then score risk keywords."""

    # Multiple risk-focused queries to retrieve relevant sections
    risk_queries = [
        f"{entity_name} fraud investigation penalty fine",
        f"{entity_name} regulatory sanction non-compliance",
        f"{entity_name} corruption bribery whistleblowing",
        f"{entity_name} money laundering litigation breach",
    ]

    seen_chunks: set[str] = set()
    retrieved_chunks: list[str] = []
    source_names: set[str] = set()

    for query in risk_queries:
        try:
            docs = vector_store.similarity_search(query, k=3)
        except Exception:
            continue
        for doc in docs:
            text = doc.page_content
            if text not in seen_chunks:
                seen_chunks.add(text)
                retrieved_chunks.append(text)
                src = doc.metadata.get("source", "")
                if src:
                    source_names.add(Path(src).name)

    if not retrieved_chunks:
        return 0.0, [f"RAG: no relevant chunks retrieved for '{entity_name}'."], []

    combined_text = " ".join(retrieved_chunks).lower()
    total_weighted_hits = 0
    per_keyword_hits: list[tuple[str, int]] = []

    for keyword, weight in REPORT_RISK_KEYWORDS.items():
        count = len(re.findall(re.escape(keyword), combined_text))
        if count > 0:
            per_keyword_hits.append((keyword, count))
            total_weighted_hits += weight + min(3, int(math.log2(count + 1)))

    findings: list[str] = []
    report_alerts: list[dict[str, Any]] = []
    sources_label = ", ".join(sorted(source_names)) or "annual reports"

    if total_weighted_hits == 0:
        findings.append(
            f"RAG retrieved {len(retrieved_chunks)} chunks from {sources_label} "
            f"for '{entity_name}' but found no risk keywords."
        )
        return 0.0, findings, report_alerts

    top_hits = sorted(per_keyword_hits, key=lambda x: x[1], reverse=True)[:5]
    hit_text = ", ".join(f"{k} x{v}" for k, v in top_hits)
    findings.append(f"RAG ({sources_label}): {hit_text}")
    severity = "high" if total_weighted_hits >= 60 else "medium"
    report_alerts.append({
        "source": f"RAG({sources_label})",
        "severity": severity,
        "summary": (
            f"Semantic retrieval found risk signals for entity '{entity_name}' "
            f"in annual reports."
        ),
    })

    report_risk_score = float(min(70, round(total_weighted_hits * 0.8, 2)))
    return report_risk_score, findings, report_alerts


def run_news_check(
    customer_id: str,
    customer_name: str,
    adverse_media_db: list[dict[str, Any]],
    entity_name: str | None = None,
    report_files: list[str | Path] | None = None,
    vector_store=None,
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
    if entity_name and vector_store is not None:
        # RAG path: use pre-built vector store for semantic retrieval
        report_score, report_reasons, report_alerts = screen_entity_from_reports_rag(
            entity_name, vector_store
        )
        reasons.extend(report_reasons)
    elif entity_name and report_files:
        # Fallback: keyword scan when no vector store available
        report_score, report_reasons, report_alerts = screen_entity_from_reports(
            entity_name, report_files
        )
        reasons.extend(report_reasons)

    combined_score = float(round(max(base_news_score, report_score), 2))
    return NewsCheckResult(
        customer_id=customer_id,
        customer_name=customer_name,
        matched_alerts=hits + report_alerts,
        news_risk_score=combined_score,
        reasons=reasons,
    )
