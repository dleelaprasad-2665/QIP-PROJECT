"""Operational recommendation engine — separate from visual classification labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from material_intelligence.app.domain.enums import HIGH_RISK_CONDITIONS, InspectionStatus
from material_intelligence.app.domain.schemas import ClassificationResult, InspectionPrediction


# Preliminary recommendation categories (not structural certification).
REC_PROCEED = "proceed_to_normal_quality_verification"
REC_ADDITIONAL = "additional_inspection_required"
REC_QUARANTINE = "quarantine_pending_review"
REC_REUSE = "candidate_for_approved_nonstructural_reuse_or_recycling"
REC_REJECT = "rejection_proposed_for_engineering_review"

RECOMMENDATION_DISPLAY = {
    REC_PROCEED: "Proceed to normal quality verification",
    REC_ADDITIONAL: "Additional inspection required",
    REC_QUARANTINE: "Quarantine pending review",
    REC_REUSE: "Candidate for approved nonstructural reuse/recycling",
    REC_REJECT: "Rejection proposed for engineering review",
}


@dataclass
class RecommendationOutcome:
    recommendation: str
    visual_condition: str
    reason: str
    confidence: float | None
    requires_engineer: bool


def visual_condition_from_classifier(result: ClassificationResult | None) -> str:
    """Binary visual screen: defect identified vs not."""
    if result is None or not result.condition:
        return "unknown"
    if result.condition == "no_visible_damage":
        return "no_visible_defect_identified"
    return "visible_defect_identified"


def recommend_from_prediction(
    prediction: InspectionPrediction,
    *,
    unsupported_material: bool = False,
    missing_documents: bool = False,
) -> RecommendationOutcome:
    primary = prediction.classifications[0] if prediction.classifications else None
    conf = primary.confidence if primary else None
    condition = primary.condition if primary else None
    visual = visual_condition_from_classifier(primary)

    if unsupported_material:
        return RecommendationOutcome(
            recommendation=REC_ADDITIONAL,
            visual_condition=visual,
            reason="Unsupported or unknown material — independent inspection required.",
            confidence=conf,
            requires_engineer=True,
        )
    if missing_documents:
        return RecommendationOutcome(
            recommendation=REC_ADDITIONAL,
            visual_condition=visual,
            reason="Missing testing documents — hold for documentation review.",
            confidence=conf,
            requires_engineer=True,
        )
    if primary is None or primary.model_status.value != "ready":
        return RecommendationOutcome(
            recommendation=REC_ADDITIONAL,
            visual_condition=visual,
            reason="Classifier unavailable or not ready — no fabricated prediction.",
            confidence=conf,
            requires_engineer=True,
        )
    if conf is not None and conf < 0.55:
        return RecommendationOutcome(
            recommendation=REC_ADDITIONAL,
            visual_condition=visual,
            reason="Low classifier confidence — human inspection required.",
            confidence=conf,
            requires_engineer=True,
        )
    if condition in HIGH_RISK_CONDITIONS:
        return RecommendationOutcome(
            recommendation=REC_QUARANTINE,
            visual_condition=visual,
            reason=f"High-risk visual condition '{condition}' — quarantine pending engineer review.",
            confidence=conf,
            requires_engineer=True,
        )
    if condition == "no_visible_damage":
        return RecommendationOutcome(
            recommendation=REC_PROCEED,
            visual_condition=visual,
            reason="No visible defect identified at current confidence — proceed to normal QA (not a structural certification).",
            confidence=conf,
            requires_engineer=False,
        )
    return RecommendationOutcome(
        recommendation=REC_ADDITIONAL,
        visual_condition=visual,
        reason=f"Visible condition '{condition}' requires additional inspection.",
        confidence=conf,
        requires_engineer=True,
    )


def recommendation_from_status(status: str | InspectionStatus) -> str:
    value = status.value if isinstance(status, InspectionStatus) else str(status)
    mapping = {
        InspectionStatus.SCREENING_COMPLETED.value: REC_PROCEED,
        InspectionStatus.HUMAN_INSPECTION_REQUIRED.value: REC_ADDITIONAL,
        InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value: REC_ADDITIONAL,
        InspectionStatus.QUARANTINED.value: REC_QUARANTINE,
        InspectionStatus.REJECTED.value: REC_REJECT,
        InspectionStatus.APPROVED.value: REC_PROCEED,
    }
    return mapping.get(value, REC_ADDITIONAL)


def evaluate_recommendations(
    predicted: list[str],
    reference: list[str],
) -> dict[str, Any]:
    """Compare preliminary recommendations to engineer reference decisions."""
    if not predicted or len(predicted) != len(reference):
        return {
            "status": "not_evaluated",
            "message": "Predicted and reference recommendation lists must be non-empty and aligned.",
        }
    n = len(predicted)
    agree = sum(1 for p, r in zip(predicted, reference, strict=True) if p == r)
    # False clearance: model says proceed, reference says quarantine/reject/additional
    risky_refs = {REC_QUARANTINE, REC_REJECT, REC_ADDITIONAL}
    false_clearance = sum(
        1
        for p, r in zip(predicted, reference, strict=True)
        if p == REC_PROCEED and r in risky_refs
    )
    unnecessary_hold = sum(
        1
        for p, r in zip(predicted, reference, strict=True)
        if p in {REC_QUARANTINE, REC_ADDITIONAL} and r == REC_PROCEED
    )
    human_referral = sum(1 for p in predicted if p == REC_ADDITIONAL)
    labels = sorted(set(predicted) | set(reference))
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for p, r in zip(predicted, reference, strict=True):
        matrix[r][p] += 1
    return {
        "status": "evaluated",
        "n": n,
        "agreement_rate": agree / n,
        "false_clearance_rate": false_clearance / n,
        "unnecessary_hold_rate": unnecessary_hold / n,
        "human_review_referral_rate": human_referral / n,
        "confusion": matrix,
        "definitions": {
            "false_clearance_rate": "Share of cases where AI recommended proceed but reference required hold/reject/additional review.",
            "unnecessary_hold_rate": "Share of cases where AI held/quarantined but reference approved proceed.",
            "disclaimer": "Recommendations are preliminary screening aids — not structural safety certifications.",
        },
    }
