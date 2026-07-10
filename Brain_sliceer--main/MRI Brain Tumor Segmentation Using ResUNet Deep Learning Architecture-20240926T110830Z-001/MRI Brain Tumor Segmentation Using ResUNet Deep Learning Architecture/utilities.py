from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src" / "brain_tumor_pipeline").exists():
            return parent
    raise RuntimeError("Could not find project root containing src/brain_tumor_pipeline")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_pipeline.data import DataGenerator  # noqa: E402
from brain_tumor_pipeline.metrics import (  # noqa: E402
    dice_coefficient,
    focal_tversky,
    iou_coefficient,
    tversky,
    tversky_loss,
)
from brain_tumor_pipeline.predict import prediction  # noqa: E402

__all__ = [
    "DataGenerator",
    "dice_coefficient",
    "focal_tversky",
    "iou_coefficient",
    "prediction",
    "tversky",
    "tversky_loss",
]
