"""
Configuration Settings for Automated Waste Material Classification & Sorting System
Model Architecture: EfficientNetB0
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for folder in [DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Waste Categories (Standard 6-class Materials Recovery Facility sorting categories)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

# Class Details & Sorting Bins
CLASS_INFO = {
    "cardboard": {
        "bin": "Bin 1 - Corrugated Paperboard",
        "color": "#D2B48C",
        "stream": "Paper & Packaging Stream",
        "co2_saved_per_kg": 1.5,  # kg CO2 saved per kg recycled
        "energy_saved_kwh": 3.2,
        "recyclable": True,
        "pneumatic_jet_id": 1,
    },
    "glass": {
        "bin": "Bin 2 - Glass Container Stream",
        "color": "#20B2AA",
        "stream": "Glass Recovery Stream",
        "co2_saved_per_kg": 0.3,
        "energy_saved_kwh": 1.1,
        "recyclable": True,
        "pneumatic_jet_id": 2,
    },
    "metal": {
        "bin": "Bin 3 - Ferrous & Non-Ferrous Metals",
        "color": "#C0C0C0",
        "stream": "Metal Recovery Stream",
        "co2_saved_per_kg": 4.2,
        "energy_saved_kwh": 14.0,
        "recyclable": True,
        "pneumatic_jet_id": 3,
    },
    "paper": {
        "bin": "Bin 4 - Mixed Fiber & Paper",
        "color": "#F5F5DC",
        "stream": "Paper Stream",
        "co2_saved_per_kg": 1.2,
        "energy_saved_kwh": 4.0,
        "recyclable": True,
        "pneumatic_jet_id": 4,
    },
    "plastic": {
        "bin": "Bin 5 - Polymers (PET / HDPE)",
        "color": "#4682B4",
        "stream": "Plastic Stream",
        "co2_saved_per_kg": 1.8,
        "energy_saved_kwh": 5.8,
        "recyclable": True,
        "pneumatic_jet_id": 5,
    },
    "trash": {
        "bin": "Bin 6 - Residual / Non-Recyclable Waste",
        "color": "#708090",
        "stream": "Landfill / Energy Recovery Stream",
        "co2_saved_per_kg": 0.0,
        "energy_saved_kwh": 0.0,
        "recyclable": False,
        "pneumatic_jet_id": 6,
    }
}

# Image Input Dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Model Training Parameters
BATCH_SIZE = 16
PHASE1_EPOCHS = 8      # Top head training
PHASE2_EPOCHS = 8      # Fine-tuning unfrozen layers
INITIAL_LR = 1e-3
FINETUNE_LR = 1e-5
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 1e-4

# Conveyor Belt Sorting System Simulation Specs
CONVEYOR_BELT_SPEED_MPS = 1.2  # meters per second
CAMERA_DISTANCE_TO_JETS_M = 1.5 # distance from camera sensor to pneumatic ejection jets (meters)
AIR_JET_PULSE_MS = 60           # duration of air jet pulse in ms
AIR_JET_PRESSURE_BAR = 6.0      # compressed air jet pressure in bar
CONFIDENCE_THRESHOLD = 0.65     # min confidence to activate automated ejection
