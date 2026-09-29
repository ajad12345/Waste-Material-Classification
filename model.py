"""
EfficientNetB0 Model Definition and Grad-CAM Explainability Module for Waste Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from PIL import Image
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE, L2_REGULARIZATION

def get_data_augmentation_layer():
    """Builds spatial data augmentation block for training."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")
    return data_augmentation

def build_efficientnet_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, trainable_backbone=False):
    """
    Builds the EfficientNetB0 classification model for waste sorting.
    
    Args:
        num_classes (int): Number of waste categories.
        input_shape (tuple): Shape of input image (224, 224, 3).
        trainable_backbone (bool): Whether backbone weights are trainable initially.
        
    Returns:
        tf.keras.Model: Compiled EfficientNetB0 model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Optional Data Augmentation
    augmented = get_data_augmentation_layer()(inputs)
    
    # EfficientNetB0 pre-trained backbone
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=augmented
    )
    
    # Set backbone trainability
    base_model.trainable = trainable_backbone
    
    # Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_head")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
        name="dense_feature_head"
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_head")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="waste_predictions"
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Waste_Classifier_EfficientNetB0")
    return model


def unfreeze_efficientnet_layers(model, num_unfreeze_layers=30):
    """
    Unfreezes the top N convolutional layers of EfficientNetB0 for fine-tuning.
    
    Args:
        model (tf.keras.Model): EfficientNetB0 model.
        num_unfreeze_layers (int): Number of top layers to unfreeze.
    """
    # Find base model layer inside current model
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break
            
    if base_model is None:
        # Check if model itself is base_model
        base_model = model
        
    base_model.trainable = True
    
    # Freeze all layers except the last num_unfreeze_layers
    for layer in base_model.layers[:-num_unfreeze_layers]:
        layer.trainable = False
        
    print(f"[Model] Unfroze last {num_unfreeze_layers} layers of EfficientNetB0 for fine-tuning.")


def generate_gradcam(img_array, model, last_conv_layer_name="top_activation", class_index=None):
    """
    Generates Grad-CAM heatmap visualization highlighting key visual features.
    
    Args:
        img_array (np.ndarray): Input preprocessed image array (1, 224, 224, 3).
        model (tf.keras.Model): Trained EfficientNetB0 model.
        last_conv_layer_name (str): Name of final feature output layer in EfficientNetB0.
        class_index (int, optional): Target class index for Grad-CAM.
        
    Returns:
        np.ndarray: Overlay heatmap image (RGB).
    """
    try:
        # Locate the conv layer in sub-model or main model
        base_model = None
        for l in model.layers:
            if "efficientnet" in l.name.lower():
                base_model = l
                break
                
        if base_model is not None:
            # Build grad model from base_model
            try:
                conv_layer = base_model.get_layer(last_conv_layer_name)
            except ValueError:
                # Fallback to last conv layer available
                conv_layer = [l for l in base_model.layers if isinstance(l, layers.Conv2D)][-1]
                
            grad_model = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=[conv_layer.output, base_model.output]
            )
        else:
            grad_model = model

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Compute gradients of class score w.r.t feature map
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation to heatmaps
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        # Resize heatmap to 224x224
        if HAS_CV2:
            heatmap_resized = cv2.resize(heatmap_np, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
            return colormap_rgb
        else:
            pil_hm = Image.fromarray(np.uint8(255 * heatmap_np)).resize((INPUT_SHAPE[1], INPUT_SHAPE[0]), Image.Resampling.BILINEAR)
            hm_np = np.array(pil_hm)
            colormap_rgb = np.stack([hm_np, 255 - hm_np, np.zeros_like(hm_np)], axis=-1)
            return colormap_rgb
    except Exception as e:
        print(f"[Grad-CAM Warning] Heatmap generation fallback: {e}")
        # Return neutral gradient grid as fallback
        dummy_heatmap = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1], 3), dtype=np.uint8)
        dummy_heatmap[:, :, 0] = 120
        return dummy_heatmap


if __name__ == "__main__":
    m = build_efficientnet_model()
    m.summary()
