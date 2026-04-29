import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import rasterio
from patchify import patchify
from sklearn.model_selection import train_test_split

from dagster import asset, AssetExecutionContext, Output, MetadataValue

from gis_pipeline.resources import PipelineConfig
from gis_pipeline.assets.training import _load_and_patch


# ── Dagster asset ─────────────────────────────────────────────────────────────

@asset(
    group_name="model_evaluation",
    deps=["train_unet"],       
    description=(
        "Loads the saved model, runs predictions on the test split, "
        "and saves the confusion matrix + classification report."
    ),
)
def evaluate_model(
    context: AssetExecutionContext,
    config: PipelineConfig,
) -> Output[dict]:

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(config.model_output_path):
        raise FileNotFoundError(
            f"Model not found at {config.model_output_path}. "
            "Run the 'train_unet' asset first."
        )

    context.log.info(f"Loading model from {config.model_output_path}")
    model = tf.keras.models.load_model(config.model_output_path)

    # ── Recreate test split ───────────────────────────────────────────────
    context.log.info("Loading data for evaluation split...")
    X, Y = _load_and_patch(config)
    _, X_test, _, y_test = train_test_split(
        X, Y, test_size=config.test_size, random_state=42  
    )
    context.log.info(f"Test patches: {len(X_test)}")

    # ── Predict ───────────────────────────────────────────────────────────
    context.log.info("Running predictions...")
    y_pred      = model.predict(X_test, batch_size=config.batch_size, verbose=0)
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
    y_test_flat = y_test.flatten()

    # ── Classification report ─────────────────────────────────────────────
    report_dict = classification_report(
        y_test_flat,
        y_pred_flat,
        target_names=config.class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_test_flat,
        y_pred_flat,
        target_names=config.class_names,
        zero_division=0,
    )
    context.log.info(f"\n{report_str}")

    with open(config.metrics_output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # ── Confusion matrix plot ─────────────────────────────────────────────
    cm      = confusion_matrix(y_test_flat, y_pred_flat)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=config.class_names,
        yticklabels=config.class_names,
        ax=ax,
    )
    ax.set_title("Matriz de Confusión Normalizada (Píxel a Píxel)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Realidad")
    plt.tight_layout()

    cm_path = os.path.join(config.plots_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    context.log.info(f"Confusion matrix saved → {cm_path}")

    # ── Training history plot (if history JSON exists) ────────────────────
    history_path = os.path.join(config.base_path, "training_history_v4.json")
    history_plot_path = None

    if os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(hist["accuracy"],     label="Entrenamiento")
        ax1.plot(hist["val_accuracy"], label="Validación")
        ax1.set_title("Precisión (Accuracy)")
        ax1.set_xlabel("Época")
        ax1.set_ylabel("Precisión")
        ax1.legend()

        ax2.plot(hist["loss"],     label="Entrenamiento")
        ax2.plot(hist["val_loss"], label="Validación")
        ax2.set_title("Pérdida (Loss)")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Pérdida")
        ax2.legend()

        plt.tight_layout()
        history_plot_path = os.path.join(config.plots_dir, "training_history.png")
        fig2.savefig(history_plot_path, dpi=150)
        plt.close(fig2)
        context.log.info(f"Training history plot saved → {history_plot_path}")

    # ── Per-class accuracy  ────────────────────────────────────────
    per_class = {
        name: round(report_dict[name]["f1-score"], 4)
        for name in config.class_names
        if name in report_dict
    }
    overall_acc = round(report_dict["accuracy"], 4)

    context.log.info(f"Overall pixel accuracy: {overall_acc:.4f}")
    for cls, f1 in per_class.items():
        context.log.info(f"  {cls:<22} F1: {f1:.4f}")

    result = {
        "overall_accuracy":    overall_acc,
        "per_class_f1":        per_class,
        "metrics_path":        config.metrics_output_path,
        "confusion_matrix_plot": cm_path,
        "history_plot":        history_plot_path,
    }

    metadata = {
        "overall_accuracy":   MetadataValue.float(overall_acc),
        "metrics_saved_to":   MetadataValue.path(config.metrics_output_path),
        "confusion_matrix":   MetadataValue.path(cm_path),
    }
    for cls, f1 in per_class.items():
        metadata[f"f1_{cls}"] = MetadataValue.float(f1)

    return Output(value=result, metadata=metadata)