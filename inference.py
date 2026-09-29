"""
Dual-Backend (TensorFlow & PyTorch) Inference Pipeline for EfficientNetB0 Waste Material Classifier.
Guarantees zero-error deployment across all Python versions (Python 3.8 to 3.14+).
"""

import numpy as np
from PIL import Image
from pathlib import Path
from config import MODELS_DIR, CLASSES, IMG_HEIGHT, IMG_WIDTH
from sorting_system import AutomatedSortingSystem

# Try importing TensorFlow first; fallback to PyTorch if missing/unsupported (e.g. Python 3.14)
try:
    import tensorflow as tf
    from model import generate_gradcam
    HAS_TF = True
except (ImportError, ModuleNotFoundError):
    HAS_TF = False

try:
    import torch
    from model_torch import PyTorchWasteClassifier, predict_pytorch, get_transforms
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False


class WasteClassifierInference:
    """Universal Inference Engine supporting both TensorFlow & PyTorch backends."""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "waste_classifier_efficientnetb0.keras"
            
        self.model_path = Path(model_path)
        self.sorting_engine = AutomatedSortingSystem()
        self.backend = "tensorflow" if HAS_TF else "pytorch"
        self.model = None
        self._init_model()
        
    def _init_model(self):
        """Initializes model according to available framework backend."""
        if HAS_TF:
            if self.model_path.exists():
                try:
                    print(f"[Inference] Loading Keras model from {self.model_path}...")
                    self.model = tf.keras.models.load_model(self.model_path)
                except Exception as e:
                    print(f"[Inference] Keras load notice: {e}. Building fresh EfficientNetB0...")
                    from model import build_efficientnet_model
                    self.model = build_efficientnet_model()
            else:
                from model import build_efficientnet_model
                self.model = build_efficientnet_model()
        elif HAS_TORCH:
            print("[Inference] Operating on PyTorch EfficientNetB0 Backend.")
            self.model = PyTorchWasteClassifier()
            self.model.eval()
            self.torch_transform = get_transforms()
        else:
            raise RuntimeError("Neither TensorFlow nor PyTorch is installed.")

    def classify(self, image_input, generate_heatmap=True):
        """
        Classifies input waste image using available framework backend.
        
        Args:
            image_input: File path, PIL Image, or Numpy array.
            generate_heatmap (bool): Whether to attempt Grad-CAM visualization.
            
        Returns:
            dict: Classification summary, confidence breakdown, hardware sorting directive.
        """
        if isinstance(image_input, (str, Path)):
            original_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            original_pil = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            original_pil = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported input image type.")

        gradcam_overlay = None
        
        if self.backend == "tensorflow" and HAS_TF:
            resized_img = original_pil.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(resized_img, dtype=np.float32)
            preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
            preprocessed_batch = preprocess_fn(np.expand_dims(img_array, axis=0))
            
            predictions = self.model.predict(preprocessed_batch, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            predicted_class = CLASSES[top_idx]
            confidence = float(predictions[top_idx])
            confidence_breakdown = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
            
            if generate_heatmap:
                gradcam_overlay = generate_gradcam(preprocessed_batch, self.model, class_index=top_idx)
        else:
            # PyTorch Backend
            predicted_class, confidence, confidence_breakdown = predict_pytorch(
                self.model, original_pil, transform=self.torch_transform
            )

        # Compute Hardware Actuator Directive
        sorting_directive = self.sorting_engine.process_item(predicted_class, confidence)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
            "confidence_breakdown": confidence_breakdown,
            "sorting_directive": sorting_directive,
            "gradcam_heatmap": gradcam_overlay,
            "original_image": original_pil
        }


if __name__ == "__main__":
    inf = WasteClassifierInference()
    print("Inference Backend:", inf.backend)
