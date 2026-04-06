from __future__ import annotations

from dataclasses import dataclass

from identity_check import IdentityCheckResult
from news_check import NewsCheckResult


@dataclass
class RiskScoringResult:
    final_risk_score: float
    risk_level: str
    decision: str
    reasons: list[str]


def run_risk_scoring(
    identity_result: IdentityCheckResult,
    news_result: NewsCheckResult,
) -> RiskScoringResult:
    identity_risk = 100.0 - identity_result.confidence_score
    news_risk = news_result.news_risk_score

    # Identity has slightly higher weight in onboarding.
    final_score = round(0.6 * identity_risk + 0.4 * news_risk, 2)

    reasons = [
        f"Identity risk={identity_risk:.2f} (from confidence {identity_result.confidence_score:.2f}).",
        f"News risk={news_risk:.2f} (from adverse media screening).",
    ]
    reasons.extend(identity_result.reasons)
    reasons.extend(news_result.reasons)

    if final_score >= 60:
        return RiskScoringResult(
            final_risk_score=final_score,
            risk_level="High",
            decision="REJECT",
            reasons=reasons,
        )
    if final_score >= 30:
        return RiskScoringResult(
            final_risk_score=final_score,
            risk_level="Medium",
            decision="MANUAL_REVIEW",
            reasons=reasons,
        )
    return RiskScoringResult(
        final_risk_score=final_score,
        risk_level="Low",
        decision="APPROVE",
        reasons=reasons,
    )
