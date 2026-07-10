from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "kaggle_3m"
DEFAULT_METADATA_CSV = (
    PROJECT_ROOT
    / "MRI Brain Tumor Segmentation Using ResUNet Deep Learning Architecture-20240926T110830Z-001"
    / "MRI Brain Tumor Segmentation Using ResUNet Deep Learning Architecture"
    / "data_mask.csv"
)


@dataclass(slots=True)
class TrainingConfig:
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = DEFAULT_DATASET_DIR
    metadata_csv: Path = DEFAULT_METADATA_CSV
    output_dir: Path = PROJECT_ROOT / "artifacts"
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 16
    random_seed: int = 42
    initial_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clipnorm: float = 1.0
    label_smoothing: float = 0.03
    classifier_dropout: float = 0.45
    classifier_l2: float = 1e-4
    classifier_epochs: int = 100
    classifier_fine_tune_epochs: int = 30
    classifier_fine_tune_layers: int = 75
    segmenter_epochs: int = 100
    folds: int = 5
    validation_size: float = 0.15
    test_size: float = 0.15
    segmenter_positive_only: bool = True
    use_mlflow: bool = False
    mlflow_experiment: str = "brain-tumor-pipeline"
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 4
    min_learning_rate: float = 1e-7
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.dataset_dir = Path(self.dataset_dir)
        self.metadata_csv = Path(self.metadata_csv)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values
