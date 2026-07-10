from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from brain_tumor_pipeline.data import read_image
from brain_tumor_pipeline.metrics import CUSTOM_OBJECTS


def load_keras_model(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)


def predict_image(
    image_path: str | Path,
    classifier: tf.keras.Model,
    segmenter: tf.keras.Model,
    image_size: tuple[int, int] = (256, 256),
    classifier_threshold: float = 0.5,
    mask_threshold: float = 0.5,
):
    image = read_image(image_path, image_size)
    batch = image[np.newaxis, ...]
    class_pred = classifier.predict(batch, verbose=0)
    if class_pred.shape[-1] == 1:
        tumor_probability = float(class_pred[0, 0])
    else:
        tumor_probability = float(class_pred[0, 1])
    if tumor_probability < classifier_threshold:
        return {"has_mask": 0, "tumor_probability": tumor_probability, "mask": None}

    pred_mask = segmenter.predict(batch, verbose=0)[0, ..., 0]
    has_mask = int((pred_mask >= mask_threshold).sum() > 0)
    return {
        "has_mask": has_mask,
        "tumor_probability": tumor_probability,
        "mask": pred_mask,
    }


def prediction(
    test: pd.DataFrame,
    model: tf.keras.Model,
    model_seg: tf.keras.Model,
    image_dir: str | Path = "./",
):
    image_ids: list[str] = []
    masks: list[object] = []
    has_masks: list[int] = []
    image_dir = Path(image_dir)

    for image_path in test.image_path:
        resolved_path = Path(str(image_path))
        if not resolved_path.is_absolute():
            resolved_path = image_dir / resolved_path
        result = predict_image(resolved_path, model, model_seg)
        image_ids.append(str(image_path))
        has_masks.append(int(result["has_mask"]))
        masks.append("No mask" if result["mask"] is None or result["has_mask"] == 0 else result["mask"])
    return image_ids, masks, has_masks
