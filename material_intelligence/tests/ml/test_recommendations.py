"""Recommendation engine tests."""

from __future__ import annotations

from material_intelligence.app.domain.enums import ModelStatus
from material_intelligence.app.domain.schemas import (
    ClassificationResult,
    DetectionResult,
    InspectionPrediction,
    InspectionStatus,
)
from material_intelligence.app.ml.evaluation.recommendations import (
    REC_ADDITIONAL,
    REC_PROCEED,
    REC_QUARANTINE,
    evaluate_recommendations,
    recommend_from_prediction,
)


def _pred(condition: str, conf: float) -> InspectionPrediction:
    return InspectionPrediction(
        detection=DetectionResult(model_status=ModelStatus.READY),
        classifications=[
            ClassificationResult(
                condition=condition,
                confidence=conf,
                model_status=ModelStatus.READY,
            )
        ],
        recommended_status=InspectionStatus.SCREENING_COMPLETED,
        processing_ms=1.0,
    )


def test_high_risk_quarantine():
    out = recommend_from_prediction(_pred("cracked", 0.9))
    assert out.recommendation == REC_QUARANTINE
    assert out.requires_engineer is True


def test_good_high_conf_proceed():
    out = recommend_from_prediction(_pred("no_visible_damage", 0.9))
    assert out.recommendation == REC_PROCEED


def test_low_conf_additional():
    out = recommend_from_prediction(_pred("no_visible_damage", 0.3))
    assert out.recommendation == REC_ADDITIONAL


def test_recommendation_agreement_metrics():
    predicted = [REC_PROCEED, REC_PROCEED, REC_QUARANTINE]
    reference = [REC_PROCEED, REC_QUARANTINE, REC_QUARANTINE]
    stats = evaluate_recommendations(predicted, reference)
    assert stats["status"] == "evaluated"
    assert abs(stats["agreement_rate"] - 2 / 3) < 1e-9
    assert abs(stats["false_clearance_rate"] - 1 / 3) < 1e-9
