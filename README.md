# ♻️ Waste Material Classification & Automated Sorting System

An end-to-end Deep Learning computer vision system designed for **Materials Recovery Facilities (MRFs)** and automated sorting conveyor belts, using **EfficientNetB0**.

---

## 🌟 Key Features

- **EfficientNetB0 Deep Architecture**: Pre-trained ImageNet transfer learning with unfrozen layer fine-tuning.
- **6 Standard Waste Categories**:
  1. `Cardboard` -> Sorting Bin 1 (Paperboard Stream)
  2. `Glass` -> Sorting Bin 2 (Glass Stream)
  3. `Metal` -> Sorting Bin 3 (Aluminum & Steel Stream)
  4. `Paper` -> Sorting Bin 4 (Mixed Fiber Stream)
  5. `Plastic` -> Sorting Bin 5 (Polymers PET/HDPE Stream)
  6. `Trash` -> Sorting Bin 6 (Residual / Landfill Stream)
- **Automated Conveyor Ejection Signals**: Computes pneumatic air jet IDs, compressed air pressure (Bar), pulse duration (ms), and travel delay timing based on conveyor speed.
- **Grad-CAM Explainability Heatmaps**: Visualizes feature region attention maps for inspecting classification rationale.
- **Interactive Streamlit Web Dashboard**: Real-time image upload, conveyor simulation batch runner, and sustainability impact metrics.

---

## 📁 Repository Structure

```
wasteMaterial/
├── config.py             # System parameters, classes, hardware specs, CO2 metrics
├── dataset_generator.py # Synthetic waste image generator & dataset loader
├── model.py             # EfficientNetB0 network architecture & Grad-CAM engine
├── train.py             # Two-stage training pipeline (top-head + fine-tuning)
├── evaluate.py          # Model evaluation, confusion matrix, per-class metrics
├── sorting_system.py    # Conveyor belt pneumatic ejection simulation engine
├── inference.py         # Real-time image classification engine
├── app.py               # Streamlit interactive web dashboard
├── requirements.txt     # Python package dependencies
└── models/              # Saved weights, loss curves, confusion matrices
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset (Optional / Auto-run)
```bash
python dataset_generator.py
```

### 3. Train EfficientNetB0 Model
```bash
python train.py
```
*Saves trained model weights to `models/waste_classifier_efficientnetb0.keras` and training history curves to `models/training_history.png`.*

### 4. Evaluate Model Performance
```bash
python evaluate.py
```
*Generates test set metrics and confusion matrix plot saved to `models/confusion_matrix.png`.*

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## ⚡ Automated Conveyor Belt Sorting Mechanics

| Material | Target Bin | Pneumatic Jet ID | Trigger Pressure | CO2 Savings / kg |
| :--- | :--- | :---: | :---: | :---: |
| **Cardboard** | Bin 1 - Corrugated Paperboard | Jet 1 | 6.0 Bar | 1.5 kg |
| **Glass** | Bin 2 - Glass Containers | Jet 2 | 6.0 Bar | 0.3 kg |
| **Metal** | Bin 3 - Ferrous & Non-Ferrous Metals | Jet 3 | 6.0 Bar | 4.2 kg |
| **Paper** | Bin 4 - Mixed Fiber & Paper | Jet 4 | 6.0 Bar | 1.2 kg |
| **Plastic** | Bin 5 - Polymers (PET / HDPE) | Jet 5 | 6.0 Bar | 1.8 kg |
| **Trash** | Bin 6 - Residual / Non-Recyclable | Jet 6 | 0.0 Bar | 0.0 kg |

---

## 🧠 Model Architecture

- **Backbone**: `EfficientNetB0`
- **Input Size**: `224 x 224 x 3`
- **Regularization**: L2 Weight Decay (`1e-4`) + Spatial Data Augmentation + Dropout (`0.3`)
- **Loss Function**: `Categorical Crossentropy`
- **Optimizer**: `Adam` (Initial `1e-3`, Fine-Tuning `1e-5`)
