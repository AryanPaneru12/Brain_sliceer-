from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import tensorflow as tf

from brain_tumor_pipeline.config import TrainingConfig
from brain_tumor_pipeline.data import (
    assert_no_patient_overlap,
    class_weights,
    classification_generators,
    group_train_val_test_split,
    load_metadata,
    patient_folds,
    segmentation_generators,
)
from brain_tumor_pipeline.metrics import dice_numpy, hausdorff95, hausdorff_distance
from brain_tumor_pipeline.models import (
    build_resnet50_classifier,
    build_resunet,
    compile_classifier,
    compile_segmenter,
    unfreeze_resnet_top_layers,
)


def callbacks(output_dir: Path, name: str) -> list[tf.keras.callbacks.Callback]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{name}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / f"{name}_history.csv")),
    ]


def mlflow_run(config: TrainingConfig, run_name: str):
    if not config.use_mlflow:
        return nullcontext(None)
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; continuing with CSV logs only.")
        return nullcontext(None)
    mlflow.set_experiment(config.mlflow_experiment)
    return mlflow.start_run(run_name=run_name)


def log_run_config(config: TrainingConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def train_classifier(config: TrainingConfig, df: pd.DataFrame | None = None) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(config.random_seed)
    df = load_metadata(config.metadata_csv, config.dataset_dir) if df is None else df.copy()
    train, val, test = group_train_val_test_split(
        df,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    assert_no_patient_overlap(train, val, test)
    train_gen, val_gen, test_gen = classification_generators(
        train,
        val,
        test,
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )

    output_dir = config.output_dir / "classifier"
    log_run_config(config, output_dir)
    weights = class_weights(train)
    model = compile_classifier(
        build_resnet50_classifier(input_shape=(*config.image_size, 3)),
        learning_rate=config.initial_learning_rate,
    )

    with mlflow_run(config, "classifier") as run:
        if run is not None:
            import mlflow

            mlflow.log_params(config.to_dict())
            mlflow.log_param("class_weight", weights)
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.classifier_epochs,
            callbacks=callbacks(output_dir, "resnet50_head"),
            class_weight=weights,
        )

        if config.classifier_fine_tune_epochs > 0:
            unfreeze_resnet_top_layers(model, n_layers=config.classifier_fine_tune_layers)
            compile_classifier(model, learning_rate=config.initial_learning_rate * 0.1)
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=config.classifier_fine_tune_epochs,
                callbacks=callbacks(output_dir, "resnet50_finetuned"),
                class_weight=weights,
            )

        metrics = model.evaluate(test_gen, verbose=1, return_dict=True)
        (output_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        if run is not None:
            import mlflow

            mlflow.log_metrics({f"test_{key}": float(value) for key, value in metrics.items()})
    return model


def cross_validate_classifier(config: TrainingConfig) -> list[dict[str, float]]:
    df = load_metadata(config.metadata_csv, config.dataset_dir)
    results: list[dict[str, float]] = []
    for fold, train, val in patient_folds(
        df,
        n_splits=config.folds,
        random_state=config.random_seed,
    ):
        train_gen, val_gen, _ = classification_generators(
            train,
            val,
            val,
            dataset_dir=config.dataset_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
        )
        output_dir = config.output_dir / f"classifier_fold_{fold}"
        log_run_config(config, output_dir)
        weights = class_weights(train)
        model = compile_classifier(
            build_resnet50_classifier(input_shape=(*config.image_size, 3)),
            learning_rate=config.initial_learning_rate,
        )
        with mlflow_run(config, f"classifier_fold_{fold}") as run:
            if run is not None:
                import mlflow

                mlflow.log_params(config.to_dict())
                mlflow.log_param("fold", fold)
            model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=config.classifier_epochs,
                callbacks=callbacks(output_dir, "resnet50_head"),
                class_weight=weights,
            )
            if config.classifier_fine_tune_epochs > 0:
                unfreeze_resnet_top_layers(model, n_layers=config.classifier_fine_tune_layers)
                compile_classifier(model, learning_rate=config.initial_learning_rate * 0.1)
                model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=config.classifier_fine_tune_epochs,
                    callbacks=callbacks(output_dir, "resnet50_finetuned"),
                    class_weight=weights,
                )
            metrics = model.evaluate(val_gen, verbose=1, return_dict=True)
            fold_metrics = {"fold": fold, **{key: float(value) for key, value in metrics.items()}}
            results.append(fold_metrics)
            (output_dir / "validation_metrics.json").write_text(
                json.dumps(fold_metrics, indent=2), encoding="utf-8"
            )
            if run is not None:
                import mlflow

                mlflow.log_metrics({f"val_{key}": value for key, value in fold_metrics.items()})

    summary = {}
    metric_keys = [key for key in results[0].keys() if key != "fold"] if results else []
    for key in metric_keys:
        values = [result[key] for result in results]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    output_dir = config.output_dir / "classifier_cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fold_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def evaluate_segmentation(model: tf.keras.Model, generator) -> dict[str, float]:
    dices: list[float] = []
    hausdorffs: list[float] = []
    hausdorff95s: list[float] = []
    for index in range(len(generator)):
        x_batch, y_batch = generator[index]
        pred_batch = model.predict(x_batch, verbose=0)
        for y_true, y_pred in zip(y_batch, pred_batch):
            dices.append(dice_numpy(y_true[..., 0], y_pred[..., 0]))
            hausdorffs.append(hausdorff_distance(y_true[..., 0], y_pred[..., 0]))
            hausdorff95s.append(hausdorff95(y_true[..., 0], y_pred[..., 0]))
    return {
        "dice": float(np.mean(dices)),
        "hausdorff": float(np.mean(hausdorffs)),
        "hausdorff95": float(np.mean(hausdorff95s)),
    }


def train_segmenter(
    config: TrainingConfig,
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    output_name: str = "segmenter",
) -> tuple[tf.keras.Model, dict[str, float]]:
    tf.keras.utils.set_random_seed(config.random_seed)
    if train is None or val is None:
        df = load_metadata(config.metadata_csv, config.dataset_dir)
        if config.segmenter_positive_only:
            df = df[df["mask"] == 1].copy()
        train, val, _ = group_train_val_test_split(
            df,
            validation_size=config.validation_size,
            test_size=config.test_size,
            random_state=config.random_seed,
        )
    assert_no_patient_overlap(train, val)
    train_gen, val_gen = segmentation_generators(
        train,
        val,
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.random_seed,
    )

    output_dir = config.output_dir / output_name
    log_run_config(config, output_dir)
    model = compile_segmenter(
        build_resunet(input_shape=(*config.image_size, 3)),
        learning_rate=config.initial_learning_rate,
    )

    with mlflow_run(config, output_name) as run:
        if run is not None:
            import mlflow

            mlflow.log_params(config.to_dict())
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.segmenter_epochs,
            callbacks=callbacks(output_dir, "resunet"),
        )
        metrics = evaluate_segmentation(model, val_gen)
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        if run is not None:
            import mlflow

            mlflow.log_metrics(metrics)
    return model, metrics


def cross_validate_segmenter(config: TrainingConfig) -> list[dict[str, float]]:
    df = load_metadata(config.metadata_csv, config.dataset_dir)
    if config.segmenter_positive_only:
        df = df[df["mask"] == 1].copy()
    results: list[dict[str, float]] = []
    for fold, train, val in patient_folds(
        df,
        n_splits=config.folds,
        random_state=config.random_seed,
    ):
        _, metrics = train_segmenter(
            config,
            train=train,
            val=val,
            output_name=f"segmenter_fold_{fold}",
        )
        metrics["fold"] = fold
        results.append(metrics)

    summary = {}
    for key in ("dice", "hausdorff", "hausdorff95"):
        values = [result[key] for result in results]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    output_dir = config.output_dir / "segmenter_cross_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fold_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-stage brain tumor pipeline.")
    parser.add_argument("--stage", choices=["classifier", "segmenter", "all"], default="all")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    if args.metadata_csv is not None:
        config.metadata_csv = args.metadata_csv
    if args.dataset_dir is not None:
        config.dataset_dir = args.dataset_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.classifier_epochs = args.epochs
        config.segmenter_epochs = args.epochs
    config.use_mlflow = args.use_mlflow

    if args.cross_validate:
        if args.stage in {"classifier", "all"}:
            cross_validate_classifier(config)
        if args.stage in {"segmenter", "all"}:
            cross_validate_segmenter(config)
        return
    if args.stage in {"classifier", "all"}:
        train_classifier(config)
    if args.stage in {"segmenter", "all"}:
        train_segmenter(config)


if __name__ == "__main__":
    main()
