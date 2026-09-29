"""
Two-Stage Training Pipeline for EfficientNetB0 Waste Classification
Stage 1: Train Top Classifier Head (Frozen Backbone)
Stage 2: Fine-tune Top Convolutional Blocks (Unfrozen Backbone)
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from config import (
    MODELS_DIR, INITIAL_LR, FINETUNE_LR, PHASE1_EPOCHS, PHASE2_EPOCHS
)
from dataset_generator import get_data_pipelines
from model import build_efficientnet_model, unfreeze_efficientnet_layers

def train_model():
    """Executes full training and fine-tuning pipeline."""
    print("=" * 60)
    print("      EFFICIENTNETB0 WASTE MATERIAL CLASSIFICATION TRAINING      ")
    print("=" * 60)
    
    # 1. Load Data Pipelines
    train_ds, val_ds, test_ds = get_data_pipelines()
    
    # 2. Build Model
    model = build_efficientnet_model(trainable_backbone=False)
    
    # Compile for Phase 1
    optimizer_phase1 = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer_phase1,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_acc")]
    )
    
    print("\n[Phase 1] Training Top Classification Head (Frozen Backbone)...")
    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_phase1
    )
    
    # 3. Phase 2: Unfreeze & Fine-tune
    print("\n[Phase 2] Fine-tuning Unfrozen EfficientNetB0 Layers...")
    unfreeze_efficientnet_layers(model, num_unfreeze_layers=30)
    
    optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
    model.compile(
        optimizer=optimizer_phase2,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_acc")]
    )
    
    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1)
    ]
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_phase2
    )
    
    # 4. Save Final Trained Models
    h5_path = MODELS_DIR / "waste_classifier_efficientnetb0.h5"
    keras_path = MODELS_DIR / "waste_classifier_efficientnetb0.keras"
    
    model.save(keras_path)
    try:
        model.save(h5_path)
    except Exception as e:
        print(f"[Warning] H5 save notice: {e}")
        
    print(f"\n[Model Saved] Trained weights saved successfully to {keras_path}")
    
    # 5. Plot & Save Combined Training History
    _plot_training_history(history1, history2)
    return model, history1, history2


def _plot_training_history(h1, h2):
    """Combines and plots training curves from Phase 1 & Phase 2."""
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    
    epochs_range = range(1, len(acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#2E7D32", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#1565C0", linewidth=2, linestyle="--")
    ax1.axvline(x=len(h1.history["accuracy"]), color="gray", linestyle=":", label="Phase 2 Fine-Tuning Start")
    ax1.set_title("EfficientNetB0 Accuracy Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#C62828", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#EF6C00", linewidth=2, linestyle="--")
    ax2.axvline(x=len(h1.history["loss"]), color="gray", linestyle=":", label="Phase 2 Fine-Tuning Start")
    ax2.set_title("EfficientNetB0 Categorical Crossentropy Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_plot_path = MODELS_DIR / "training_history.png"
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    print(f"[History Plot] Saved training metrics plot to {history_plot_path}")


if __name__ == "__main__":
    train_model()
