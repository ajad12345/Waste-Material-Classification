"""
Inference Pipeline for EfficientNetB0 Waste Material Classifier
Supports single-image analysis, batch folder processing, and Grad-CAM visual explainability overlay.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from config import MODELS_DIR, CLASSES, IMG_HEIGHT, IMG_WIDTH
from model import generate_gradcam
from sorting_system import AutomatedSortingSystem

class WasteClassifierInference:
    """Inference engine for waste material classification."""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "waste_classifier_efficientnetb0.keras"
            
        self.model_path = Path(model_path)
        self.model = None
        self.sorting_engine = AutomatedSortingSystem()
        self._load_model()
        
    def _load_model(self):
        """Loads model weights if available, otherwise builds pre-trained default."""
        if self.model_path.exists():
            print(f"[Inference] Loading model weights from {self.model_path}...")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"[Inference Warning] Failed to load saved keras model: {e}. Building fresh pre-trained model.")
                from model import build_efficientnet_model
                self.model = build_efficientnet_model()
        else:
            print("[Inference] Saved model checkpoint not found. Instantiating pre-trained EfficientNetB0 architecture.")
            from model import build_efficientnet_model
            self.model = build_efficientnet_model()

    def preprocess_image(self, image_input):
        """
        Preprocesses PIL Image, raw numpy array, or file path to model input format.
        
        Args:
            image_input (Image.Image or str or Path or np.ndarray): Raw input image.
            
        Returns:
            tuple: (preprocessed_batch_array, original_pil_image)
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input format.")
            
        resized_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(resized_img, dtype=np.float32)
        
        # Apply EfficientNet preprocessing
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        preprocessed_batch = preprocess_fn(np.expand_dims(img_array, axis=0))
        
        return preprocessed_batch, pil_img

    def classify(self, image_input, generate_heatmap=True):
        """
        Classifies input waste image and computes automated sorting parameters.
        
        Args:
            image_input: Input waste image.
            generate_heatmap (bool): Whether to generate Grad-CAM explainability heatmap.
            
        Returns:
            dict: Comprehensive prediction, confidence breakdown, Grad-CAM heatmap, and hardware sorting signal.
        """
        preprocessed_batch, original_pil = self.preprocess_image(image_input)
        
        predictions = self.model.predict(preprocessed_batch, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        predicted_class = CLASSES[top_idx]
        confidence = float(predictions[top_idx])
        
        # Confidence breakdown across all 6 waste classes
        confidence_breakdown = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
        
        # Hardware sorting directive
        sorting_directive = self.sorting_engine.process_item(predicted_class, confidence)
        
        # Grad-CAM heatmap
        gradcam_overlay = None
        if generate_heatmap:
            gradcam_overlay = generate_gradcam(preprocessed_batch, self.model, class_index=top_idx)
            
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
            "confidence_breakdown": confidence_breakdown,
            "sorting_directive": sorting_directive,
            "gradcam_heatmap": gradcam_overlay,
            "original_image": original_pil
        }
        return result


if __name__ == "__main__":
    from dataset_generator import generate_synthetic_dataset, TEST_DIR
    import random
    
    pipeline = WasteClassifierInference()
    sample_file = list(TEST_DIR.glob("**/*.png"))
    if sample_file:
        test_img = random.choice(sample_file)
        res = pipeline.classify(test_img)
        print("Prediction Result:", res["predicted_class"], f"({res['confidence_percent']})")
        print("Sorting Directive:", res["sorting_directive"]["action"], "->", res["sorting_directive"]["target_bin"])
