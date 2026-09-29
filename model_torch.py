"""
PyTorch EfficientNetB0 Model Definition for Waste Material Classification
Compatible with Python 3.10, 3.11, 3.12, 3.13, and 3.14+
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from config import NUM_CLASSES, INPUT_SHAPE, CLASSES

class PyTorchWasteClassifier(nn.Module):
    """EfficientNetB0 model using PyTorch."""
    def __init__(self, num_classes=NUM_CLASSES):
        super(PyTorchWasteClassifier, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """Standard ImageNet transform for EfficientNetB0."""
    return transforms.Compose([
        transforms.Resize((INPUT_SHAPE[0], INPUT_SHAPE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_pytorch(model, image, transform=None):
    """Performs inference on a PIL Image."""
    if transform is None:
        transform = get_transforms()
        
    model.eval()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
        
    tensor_img = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        
    top_idx = int(np.argmax(probs))
    predicted_class = CLASSES[top_idx]
    confidence = float(probs[top_idx])
    breakdown = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    
    return predicted_class, confidence, breakdown
