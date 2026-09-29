"""
Evaluation Module for Waste Material Classification Model
Computes test set accuracy, precision, recall, F1 score, confusion matrix, and per-class performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from config import MODELS_DIR, CLASSES
from dataset_generator import get_data_pipelines

def evaluate_model(model_path=None):
    """
    Evaluates the trained EfficientNetB0 model on the test dataset.
    
    Args:
        model_path (str/Path, optional): Path to model file. Defaults to latest saved model.
    """
    print("=" * 60)
    print("      EFFICIENTNETB0 WASTE MODEL EVALUATION      ")
    print("=" * 60)
    
    if model_path is None:
        model_path = MODELS_DIR / "waste_classifier_efficientnetb0.keras"
        
    if not Path(model_path).exists():
        print(f"[Error] Model path {model_path} does not exist. Please train the model first.")
        return
        
    print(f"[Model] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    _, _, test_ds = get_data_pipelines()
    
    # Accumulate predictions & true labels
    y_true = []
    y_pred = []
    y_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # 1. Overall Metrics
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=CLASSES)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(report_str)
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm)
    
    return report_dict, cm


def _plot_confusion_matrix(cm):
    """Plots and saves confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in CLASSES],
        yticklabels=[c.capitalize() for c in CLASSES],
        cbar=True,
        square=True
    )
    plt.title("EfficientNetB0 Waste Classification Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("True Category", fontsize=11)
    plt.tight_layout()
    
    cm_plot_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix plot to {cm_plot_path}")


if __name__ == "__main__":
    evaluate_model()
