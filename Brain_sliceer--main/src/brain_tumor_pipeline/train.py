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


def log_progress(message: str) -> None:
    print(f"[brain-tumor-train] {message}", flush=True)


def callbacks(
    output_dir: Path,
    name: str,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 12,
    reduce_lr_patience: int = 4,
    min_lr: float = 1e-7,
) -> list[tf.keras.callbacks.Callback]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{name}.keras"),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
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


def configure_runtime(config: TrainingConfig) -> None:
    log_progress("Configuring TensorFlow runtime")
    tf.keras.utils.set_random_seed(config.random_seed)
    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log_progress("Mixed precision enabled")


def write_dataset_report(df: pd.DataFrame, output_dir: Path, split_name: str = "all") -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_summary = df.groupby("patient_id")["mask"].agg(["count", "sum", "mean"])
    class_counts = df["mask"].value_counts().sort_index()
    report: dict[str, object] = {
        "split": split_name,
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()),
        "negative_images": int(class_counts.get(0, 0)),
        "positive_images": int(class_counts.get(1, 0)),
        "positive_image_fraction": float(df["mask"].mean()) if len(df) else 0.0,
        "positive_patients": int((patient_summary["sum"] > 0).sum()),
        "negative_only_patients": int((patient_summary["sum"] == 0).sum()),
        "slices_per_patient_min": int(patient_summary["count"].min()) if len(patient_summary) else 0,
        "slices_per_patient_median": float(patient_summary["count"].median()) if len(patient_summary) else 0.0,
        "slices_per_patient_max": int(patient_summary["count"].max()) if len(patient_summary) else 0,
    }
    (output_dir / f"dataset_report_{split_name}.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def write_history_report(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    name: str,
    primary_metric: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = history.history
    report: dict[str, object] = {"run": name, "epochs_completed": len(values.get("loss", []))}
    for key, series in values.items():
        if not series:
            continue
        report[f"last_{key}"] = float(series[-1])
        if key.startswith("val_") or key in {"loss", "val_loss"}:
            report[f"best_{key}"] = float(min(series) if "loss" in key else max(series))

    train_key = primary_metric
    val_key = f"val_{primary_metric}"
    if train_key in values and val_key in values and values[train_key] and values[val_key]:
        train_last = float(values[train_key][-1])
        val_last = float(values[val_key][-1])
        report[f"{primary_metric}_train_val_gap"] = train_last - val_last
        if train_last - val_last > 0.12:
            report["overfitting_warning"] = (
                f"Last training {primary_metric} is more than 0.12 above validation. "
                "Use stronger augmentation, lower fine-tune layers, more dropout, or fewer epochs."
            )
    (output_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_classifier(config: TrainingConfig, df: pd.DataFrame | None = None) -> tf.keras.Model:
    configure_runtime(config)
    log_progress(f"Loading classifier metadata from {config.dataset_dir}")
    df = load_metadata(config.metadata_csv, config.dataset_dir) if df is None else df.copy()
    log_progress(f"Loaded {len(df)} image rows across {df['patient_id'].nunique()} patients")
    log_progress("Creating patient-safe train/validation/test split")
    train, val, test = group_train_val_test_split(
        df,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    assert_no_patient_overlap(train, val, test)
    log_progress(f"Split rows: train={len(train)}, val={len(val)}, test={len(test)}")
    log_progress("Creating classifier data generators")
    train_gen, val_gen, test_gen = classification_generators(
        train,
        val,
        test,
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )
    log_progress(f"Generator batches: train={len(train_gen)}, val={len(val_gen)}, test={len(test_gen)}")

    output_dir = config.output_dir / "classifier"
    log_run_config(config, output_dir)
    write_dataset_report(df, output_dir, "all")
    write_dataset_report(train, output_dir, "train")
    write_dataset_report(val, output_dir, "val")
    write_dataset_report(test, output_dir, "test")
    weights = class_weights(train)
    log_progress("Building ResNet50 classifier. This may download ImageNet weights on the first run.")
    model = compile_classifier(
        build_resnet50_classifier(
            input_shape=(*config.image_size, 3),
            dropout_rate=config.classifier_dropout,
            l2_regularization=config.classifier_l2,
        ),
        learning_rate=config.initial_learning_rate,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        clipnorm=config.gradient_clipnorm,
    )

    with mlflow_run(config, "classifier") as run:
        if run is not None:
            import mlflow

            mlflow.log_params(config.to_dict())
            mlflow.log_param("class_weight", weights)
        log_progress(f"Starting classifier head training for {config.classifier_epochs} epochs")
        head_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.classifier_epochs,
            callbacks=callbacks(
                output_dir,
                "resnet50_head",
                monitor="val_auc",
                mode="max",
                patience=config.early_stopping_patience,
                reduce_lr_patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
            ),
            class_weight=weights,
        )
        write_history_report(head_history, output_dir, "resnet50_head", primary_metric="auc")

        if config.classifier_fine_tune_epochs > 0:
            log_progress(
                f"Starting classifier fine-tuning for {config.classifier_fine_tune_epochs} epochs"
            )
            unfreeze_resnet_top_layers(model, n_layers=config.classifier_fine_tune_layers)
            compile_classifier(
                model,
                learning_rate=config.initial_learning_rate * 0.1,
                weight_decay=config.weight_decay,
                label_smoothing=config.label_smoothing,
                clipnorm=config.gradient_clipnorm,
            )
            fine_tune_history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=config.classifier_fine_tune_epochs,
                callbacks=callbacks(
                    output_dir,
                    "resnet50_finetuned",
                    monitor="val_auc",
                    mode="max",
                    patience=config.early_stopping_patience,
                    reduce_lr_patience=config.reduce_lr_patience,
                    min_lr=config.min_learning_rate,
                ),
                class_weight=weights,
            )
            write_history_report(
                fine_tune_history,
                output_dir,
                "resnet50_finetuned",
                primary_metric="auc",
            )

        log_progress("Evaluating classifier on the test split")
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
            build_resnet50_classifier(
                input_shape=(*config.image_size, 3),
                dropout_rate=config.classifier_dropout,
                l2_regularization=config.classifier_l2,
            ),
            learning_rate=config.initial_learning_rate,
            weight_decay=config.weight_decay,
            label_smoothing=config.label_smoothing,
            clipnorm=config.gradient_clipnorm,
        )
        with mlflow_run(config, f"classifier_fold_{fold}") as run:
            if run is not None:
                import mlflow

                mlflow.log_params(config.to_dict())
                mlflow.log_param("fold", fold)
            head_history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=config.classifier_epochs,
                callbacks=callbacks(
                    output_dir,
                    "resnet50_head",
                    monitor="val_auc",
                    mode="max",
                    patience=config.early_stopping_patience,
                    reduce_lr_patience=config.reduce_lr_patience,
                    min_lr=config.min_learning_rate,
                ),
                class_weight=weights,
            )
            write_history_report(head_history, output_dir, "resnet50_head", primary_metric="auc")
            if config.classifier_fine_tune_epochs > 0:
                unfreeze_resnet_top_layers(model, n_layers=config.classifier_fine_tune_layers)
                compile_classifier(
                    model,
                    learning_rate=config.initial_learning_rate * 0.1,
                    weight_decay=config.weight_decay,
                    label_smoothing=config.label_smoothing,
                    clipnorm=config.gradient_clipnorm,
                )
                fine_tune_history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=config.classifier_fine_tune_epochs,
                    callbacks=callbacks(
                        output_dir,
                        "resnet50_finetuned",
                        monitor="val_auc",
                        mode="max",
                        patience=config.early_stopping_patience,
                        reduce_lr_patience=config.reduce_lr_patience,
                        min_lr=config.min_learning_rate,
                    ),
                    class_weight=weights,
                )
                write_history_report(
                    fine_tune_history,
                    output_dir,
                    "resnet50_finetuned",
                    primary_metric="auc",
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
    configure_runtime(config)
    if train is None or val is None:
        log_progress(f"Loading segmenter metadata from {config.dataset_dir}")
        df = load_metadata(config.metadata_csv, config.dataset_dir)
        if config.segmenter_positive_only:
            df = df[df["mask"] == 1].copy()
            log_progress(f"Using {len(df)} positive-mask rows for segmentation")
        log_progress("Creating segmenter train/validation split")
        train, val, _ = group_train_val_test_split(
            df,
            validation_size=config.validation_size,
            test_size=config.test_size,
            random_state=config.random_seed,
        )
    assert_no_patient_overlap(train, val)
    log_progress(f"Segmenter split rows: train={len(train)}, val={len(val)}")
    log_progress("Creating segmenter data generators")
    train_gen, val_gen = segmentation_generators(
        train,
        val,
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.random_seed,
    )
    log_progress(f"Segmenter batches: train={len(train_gen)}, val={len(val_gen)}")

    output_dir = config.output_dir / output_name
    log_run_config(config, output_dir)
    write_dataset_report(train, output_dir, "train")
    write_dataset_report(val, output_dir, "val")
    log_progress("Building ResUNet segmenter")
    model = compile_segmenter(
        build_resunet(input_shape=(*config.image_size, 3)),
        learning_rate=config.initial_learning_rate,
        weight_decay=config.weight_decay,
        clipnorm=config.gradient_clipnorm,
    )

    with mlflow_run(config, output_name) as run:
        if run is not None:
            import mlflow

            mlflow.log_params(config.to_dict())
        log_progress(f"Starting segmenter training for {config.segmenter_epochs} epochs")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.segmenter_epochs,
            callbacks=callbacks(
                output_dir,
                "resunet",
                monitor="val_dice_coefficient",
                mode="max",
                patience=config.early_stopping_patience,
                reduce_lr_patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
            ),
        )
        write_history_report(history, output_dir, "resunet", primary_metric="dice_coefficient")
        log_progress("Evaluating segmenter")
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
    parser.add_argument("--classifier-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--segmenter-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
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
        if args.fine_tune_epochs is None:
            config.classifier_fine_tune_epochs = 0
    if args.classifier_epochs is not None:
        config.classifier_epochs = args.classifier_epochs
    if args.fine_tune_epochs is not None:
        config.classifier_fine_tune_epochs = args.fine_tune_epochs
    if args.segmenter_epochs is not None:
        config.segmenter_epochs = args.segmenter_epochs
    if args.fine_tune_layers is not None:
        config.classifier_fine_tune_layers = args.fine_tune_layers
    if args.learning_rate is not None:
        config.initial_learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.dropout is not None:
        config.classifier_dropout = args.dropout
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
    if args.mixed_precision:
        config.mixed_precision = True
    config.use_mlflow = args.use_mlflow

    log_progress(
        "Starting run with "
        f"stage={args.stage}, dataset_dir={config.dataset_dir}, output_dir={config.output_dir}, "
        f"batch_size={config.batch_size}, classifier_epochs={config.classifier_epochs}, "
        f"fine_tune_epochs={config.classifier_fine_tune_epochs}, "
        f"segmenter_epochs={config.segmenter_epochs}"
    )

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
