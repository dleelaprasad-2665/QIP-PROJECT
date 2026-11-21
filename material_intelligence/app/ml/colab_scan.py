"""Notebook / Colab-friendly scan — no NiceGUI, no database required.

Uses visual heuristics (+ optional YOLO / EfficientNet weights if present).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from material_intelligence.app.domain.enums import MATERIAL_DISPLAY_NAMES
from material_intelligence.app.domain.schemas import PreprocessProfile
from material_intelligence.app.ml.brick_heuristic import suggest_brick_condition
from material_intelligence.app.ml.cement_bag_heuristic import suggest_cement_condition
from material_intelligence.app.ml.material_visual import guess_material_from_image
from material_intelligence.app.ml.preprocessing import preprocess_image
from material_intelligence.app.ml.rust_heuristic import suggest_steel_condition

MATERIAL_KEYS = {
    "brick": "clay_brick",
    "clay_brick": "clay_brick",
    "steel": "steel_rebar",
    "steel_rebar": "steel_rebar",
    "rebar": "steel_rebar",
    "cement": "cement_bag",
    "cement_bag": "cement_bag",
    "wood": "wood",
    "auto": None,
    "auto-detect": None,
}


@dataclass
class ColabScanResult:
    material: str
    material_key: str
    condition: str  # Good | Damaged | Inspection Required
    recommendation: str
    detail: str
    confidence_pct: int
    preprocess: str
    ai_notes: str


def _to_condition(cond_key: str, conf: float) -> tuple[str, str, str]:
    """Map model condition → (label, recommendation, detail)."""
    details = {
        "no_visible_damage": "No clear surface damage seen in the photo.",
        "surface_corrosion": "Surface rust / corrosion visible on the steel.",
        "extensive_visible_corrosion": "Heavy rust / corrosion — needs engineer review.",
        "deformation": "Bars look bent or deformed.",
        "cracked": "Cracks are visible on the material.",
        "chipped": "Chips or broken edges are visible.",
        "broken": "Material appears broken or uneven.",
        "torn_packaging": "Torn packaging on the cement bag (label area).",
        "visible_moisture_staining": "Moisture staining is visible.",
        "damaged_packaging": "Damaged packaging on the cement bag.",
    }
    detail = details.get(cond_key, cond_key.replace("_", " ").title())
    if conf < 0.45:
        return "Inspection Required", "Further inspection", detail
    if cond_key == "no_visible_damage":
        return "Good", "Routine checks", detail
    if cond_key in {
        "extensive_visible_corrosion",
        "broken",
        "torn_packaging",
        "deformation",
    }:
        return "Damaged", "Hold for review", detail
    if cond_key in {
        "surface_corrosion",
        "cracked",
        "chipped",
        "damaged_packaging",
        "visible_moisture_staining",
    }:
        return "Damaged", "Further inspection", detail
    return "Inspection Required", "Further inspection", detail


def _resolve_material(
    image_bgr: np.ndarray,
    material_hint: str | None,
) -> tuple[str, str, str]:
    if material_hint:
        key = MATERIAL_KEYS.get(material_hint.lower().strip(), material_hint)
        if key in MATERIAL_DISPLAY_NAMES:
            return key, MATERIAL_DISPLAY_NAMES[key], "user selection"
    visual_key, visual_conf, _ = guess_material_from_image(image_bgr)
    if visual_key:
        return (
            visual_key,
            MATERIAL_DISPLAY_NAMES.get(visual_key, visual_key),
            f"visual ({visual_conf:.0%})",
        )
    return "steel_rebar", "Steel", "fallback"


def _condition_for_material(material_key: str, image_bgr: np.ndarray) -> tuple[str, float, str]:
    if material_key == "steel_rebar":
        return suggest_steel_condition(image_bgr)
    if material_key == "clay_brick":
        return suggest_brick_condition(image_bgr)
    if material_key == "cement_bag":
        return suggest_cement_condition(image_bgr)
    # Wood / unknown: light moisture / edge check via brick-like outline
    cond, conf, note = suggest_brick_condition(image_bgr)
    return cond, conf, note.replace("Brick", "Wood")


def load_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def scan_image(
    image_bgr: np.ndarray,
    *,
    material: str | None = "auto",
) -> ColabScanResult:
    """
    Run the 6-step screening flow in-process (for Colab / scripts).

    1 Upload (array) → 2 Preprocess → 3 Heuristic AI → 4 Condition → 5 Rec → 6 caller stores
    """
    profile = PreprocessProfile()
    profile.resize_to = None
    processed = preprocess_image(image_bgr, profile).processed_bgr
    scan = processed if processed.size else image_bgr

    hint = None if not material or str(material).lower() in {"auto", "auto-detect", "none"} else material
    material_key, material_name, source = _resolve_material(scan, hint)
    cond_key, conf, note = _condition_for_material(material_key, scan)
    label, rec, detail = _to_condition(cond_key, conf)

    return ColabScanResult(
        material=material_name,
        material_key=material_key,
        condition=label,
        recommendation=rec,
        detail=detail,
        confidence_pct=int(round(conf * 100)),
        preprocess="CLAHE · Resize · Normalize",
        ai_notes=f"Material via {source}. {note}",
    )


def scan_path(path: str | Path, *, material: str | None = "auto") -> ColabScanResult:
    return scan_image(load_bgr(path), material=material)


def scan_folder(
    folder: str | Path,
    *,
    material: str | None = "auto",
) -> list[dict]:
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows: list[dict] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            result = scan_path(p, material=material)
            row = asdict(result)
            row["file"] = p.name
            row["path"] = str(p)
            rows.append(row)
    return rows
