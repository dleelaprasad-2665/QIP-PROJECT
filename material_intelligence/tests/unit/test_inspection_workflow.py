"""Inspection status recommendation tests."""

from __future__ import annotations

from material_intelligence.app.domain.enums import InspectionStatus, ModelStatus
from material_intelligence.app.domain.schemas import ClassificationResult
from material_intelligence.app.ml.inference_pipeline import recommend_status


def test_high_risk_requires_additional_testing():
    result = ClassificationResult(
        condition="cracked",
        confidence=0.9,
        model_status=ModelStatus.READY,
    )
    assert recommend_status([result]) == InspectionStatus.ADDITIONAL_TESTING_REQUIRED


def test_uncertain_requires_human_review():
    result = ClassificationResult(
        condition="no_visible_damage",
        confidence=0.3,
        model_status=ModelStatus.READY,
        message="Uncertain prediction — human inspection recommended.",
    )
    assert recommend_status([result]) == InspectionStatus.HUMAN_INSPECTION_REQUIRED


def test_empty_classifications_require_human():
    assert recommend_status([]) == InspectionStatus.HUMAN_INSPECTION_REQUIRED
