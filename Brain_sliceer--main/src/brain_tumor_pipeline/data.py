from __future__ import annotations

import math
import warnings
from pathlib import Path, PurePosixPath
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from brain_tumor_pipeline.config import DEFAULT_DATASET_DIR, DEFAULT_METADATA_CSV


def patient_id_from_image_path(image_path: str) -> str:
    normalized = str(image_path).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not parts:
        raise ValueError(f"Cannot derive patient_id from empty image_path: {image_path!r}")
    return parts[0]


def load_metadata(
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    prefer_path_patient_id: bool = True,
) -> pd.DataFrame:
    metadata_csv = Path(metadata_csv)
    dataset_dir = Path(dataset_dir)
    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
    else:
        df = build_metadata_from_dataset(dataset_dir)

    required = {"image_path", "mask_path", "mask"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    df["mask_path"] = df["mask_path"].astype(str).str.replace("\\", "/", regex=False)
    df["mask"] = df["mask"].astype(int)
    df["label"] = df["mask"].astype(str)

    path_patient_id = df["image_path"].map(patient_id_from_image_path)
    if "patient_id" in df.columns:
        csv_patient_id = df["patient_id"].astype(str)
        mismatch = (csv_patient_id != path_patient_id).sum()
        if mismatch:
            warnings.warn(
                f"{mismatch} metadata rows have patient_id values that do not match "
                "the image_path folder. Using image_path-derived patient_id for leakage-safe splits.",
                RuntimeWarning,
                stacklevel=2,
            )
    if prefer_path_patient_id or "patient_id" not in df.columns:
        df["patient_id"] = path_patient_id

    return df


def build_metadata_from_dataset(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    rows: list[dict[str, object]] = []
    for image_path in sorted(dataset_dir.glob("TCGA_*/*.tif")):
        if image_path.stem.endswith("_mask"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rows.append(
            {
                "patient_id": image_path.parent.name,
                "image_path": image_path.relative_to(dataset_dir).as_posix(),
                "mask_path": mask_path.relative_to(dataset_dir).as_posix(),
                "mask": int(mask is not None and np.any(mask > 0)),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No image/mask pairs found under {dataset_dir}")
    return pd.DataFrame(rows)


def assert_no_patient_overlap(*frames: pd.DataFrame, group_col: str = "patient_id") -> None:
    groups = [set(frame[group_col].astype(str)) for frame in frames]
    for left_idx, left in enumerate(groups):
        for right_idx, right in enumerate(groups[left_idx + 1 :], start=left_idx + 1):
            overlap = left.intersection(right)
            if overlap:
                preview = sorted(overlap)[:5]
                raise AssertionError(
                    f"Patient leakage between split {left_idx} and split {right_idx}: {preview}"
                )


def group_train_val_test_split(
    df: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    group_col: str = "patient_id",
    label_col: str = "mask",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    groups = df[group_col].astype(str)
    labels = df[label_col].astype(int)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, labels, groups))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    relative_val_size = validation_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state + 1,
    )
    train_idx, val_idx = next(
        val_splitter.split(
            train_val,
            train_val[label_col].astype(int),
            train_val[group_col].astype(str),
        )
    )
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()
    assert_no_patient_overlap(train, val, test, group_col=group_col)
    return train, val, test


def patient_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "patient_id",
    label_col: str = "mask",
    random_state: int = 42,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = df[group_col].astype(str)
    labels = df[label_col].astype(int)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, labels, groups), start=1):
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        assert_no_patient_overlap(train, val, group_col=group_col)
        yield fold, train, val


def class_weights(df: pd.DataFrame, label_col: str = "mask") -> dict[int, float]:
    labels = df[label_col].astype(int).to_numpy()
    classes = np.sort(np.unique(labels))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def read_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return np.clip(image, 0.0, 1.0)


def read_mask(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    mask = cv2.resize(mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)[..., np.newaxis]


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        ids,
        mask,
        image_dir: str | Path = "./",
        batch_size: int = 16,
        img_h: int = 256,
        img_w: int = 256,
        shuffle: bool = True,
        augment: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ids = np.asarray(ids)
        self.mask = np.asarray(mask)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.image_size = (img_h, img_w)
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        return self._data_generation(list_ids, list_mask)

    def on_epoch_end(self) -> None:
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _resolve(self, path: str | Path) -> Path:
        path = Path(str(path))
        return path if path.is_absolute() else self.image_dir / path

    def _data_generation(self, list_ids, list_mask) -> tuple[np.ndarray, np.ndarray]:
        batch_len = len(list_ids)
        x = np.empty((batch_len, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        y = np.empty((batch_len, self.image_size[0], self.image_size[1], 1), dtype=np.float32)

        for i, (image_id, mask_id) in enumerate(zip(list_ids, list_mask)):
            image = read_image(self._resolve(image_id), self.image_size)
            mask = read_mask(self._resolve(mask_id), self.image_size)
            if self.augment:
                image, mask = self._augment_pair(image, mask)
            x[i] = image
            y[i] = mask
        return x, y

    def _augment_pair(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if self.rng.random() < 0.25:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        height, width = image.shape[:2]
        angle = float(self.rng.uniform(-15.0, 15.0))
        scale = float(self.rng.uniform(0.9, 1.1))
        shift_x = float(self.rng.uniform(-0.05, 0.05) * width)
        shift_y = float(self.rng.uniform(-0.05, 0.05) * height)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[:, 2] += (shift_x, shift_y)

        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask_2d = cv2.warpAffine(
            mask[..., 0],
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        brightness = float(self.rng.uniform(0.8, 1.2))
        image = np.clip(image * brightness, 0.0, 1.0)
        mask = (mask_2d > 0.5).astype(np.float32)[..., np.newaxis]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


# Backward-compatible name used by the original notebook.
DataGenerator = SegmentationDataGenerator


def classification_generators(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    dataset_dir: str | Path,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 16,
) -> tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    flow_kwargs = {
        "directory": str(dataset_dir),
        "x_col": "image_path",
        "y_col": "label",
        "class_mode": "categorical",
        "target_size": image_size,
        "batch_size": batch_size,
    }
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        shuffle=True,
        seed=42,
        **flow_kwargs,
    )
    val_gen = eval_datagen.flow_from_dataframe(
        dataframe=val,
        shuffle=False,
        **flow_kwargs,
    )
    test_gen = eval_datagen.flow_from_dataframe(
        dataframe=test,
        shuffle=False,
        **flow_kwargs,
    )
    return train_gen, val_gen, test_gen


def segmentation_generators(
    train: pd.DataFrame,
    val: pd.DataFrame,
    dataset_dir: str | Path,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[SegmentationDataGenerator, SegmentationDataGenerator]:
    train_gen = SegmentationDataGenerator(
        train["image_path"].to_numpy(),
        train["mask_path"].to_numpy(),
        image_dir=dataset_dir,
        batch_size=batch_size,
        img_h=image_size[0],
        img_w=image_size[1],
        shuffle=True,
        augment=True,
        seed=seed,
    )
    val_gen = SegmentationDataGenerator(
        val["image_path"].to_numpy(),
        val["mask_path"].to_numpy(),
        image_dir=dataset_dir,
        batch_size=batch_size,
        img_h=image_size[0],
        img_w=image_size[1],
        shuffle=False,
        augment=False,
        seed=seed,
    )
    return train_gen, val_gen
