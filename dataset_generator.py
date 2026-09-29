"""
Dataset Generator & Loader for Waste Material Classification
Generates synthetic waste material datasets and creates TensorFlow data pipelines.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import tensorflow as tf
from pathlib import Path
from config import (
    CLASSES, DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
)

def _draw_cardboard(draw, width, height):
    """Draw synthetic corrugated cardboard item."""
    base_color = (random.randint(180, 210), random.randint(140, 170), random.randint(90, 120))
    bg_color = (random.randint(220, 240), random.randint(220, 240), random.randint(220, 240))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Cardboard piece polygon
    margin = random.randint(20, 40)
    box = [margin, margin, width - margin, height - margin]
    draw.rectangle(box, fill=base_color, outline=(140, 100, 60), width=3)
    
    # Corrugated lines
    for y in range(margin + 5, height - margin - 5, 8):
        draw.line([margin + 5, y, width - margin - 5, y], fill=(150, 110, 70), width=2)
    
    # Flap crease
    draw.line([width // 2, margin, width // 2, height - margin], fill=(130, 90, 50), width=3)
    return img

def _draw_glass(draw, width, height):
    """Draw synthetic glass bottle / jar item."""
    bg_color = (random.randint(210, 230), random.randint(210, 230), random.randint(210, 230))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    glass_hues = [
        (30, 140, 100, 180),  # Green bottle
        (100, 140, 180, 180), # Blue bottle
        (180, 150, 100, 180), # Amber bottle
        (200, 220, 230, 180)  # Clear glass
    ]
    glass_color = random.choice(glass_hues)
    
    # Bottle silhouette
    cx, cy = width // 2, height // 2
    r_body = random.randint(45, 65)
    r_neck = random.randint(15, 25)
    
    bottle_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    b_draw = ImageDraw.Draw(bottle_img)
    
    # Body & neck
    b_draw.ellipse([cx - r_body, cy - r_body + 20, cx + r_body, cy + r_body + 40], fill=glass_color, outline=(255, 255, 255, 220), width=3)
    b_draw.rectangle([cx - r_neck, cy - r_body - 30, cx + r_neck, cy - r_body + 20], fill=glass_color, outline=(255, 255, 255, 220), width=2)
    
    # Specular light highlight
    b_draw.line([cx - r_body + 15, cy - 20, cx - r_body + 15, cy + 40], fill=(255, 255, 255, 230), width=5)
    
    img = Image.alpha_composite(img.convert("RGBA"), bottle_img).convert("RGB")
    return img

def _draw_metal(draw, width, height):
    """Draw synthetic aluminum/steel metal can item."""
    bg_color = (random.randint(220, 240), random.randint(220, 240), random.randint(220, 240))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    cx, cy = width // 2, height // 2
    w_can, h_can = random.randint(70, 100), random.randint(100, 130)
    
    # Metallic base
    draw.rectangle([cx - w_can//2, cy - h_can//2, cx + w_can//2, cy + h_can//2], fill=(190, 195, 205), outline=(100, 105, 115), width=3)
    
    # Metallic gradient reflection lines
    for offset in range(-w_can//2 + 10, w_can//2 - 10, 15):
        alpha_val = random.randint(210, 255)
        draw.line([cx + offset, cy - h_can//2, cx + offset, cy + h_can//2], fill=(alpha_val, alpha_val, alpha_val), width=4)
        
    # Top rim & pull tab
    draw.ellipse([cx - w_can//2, cy - h_can//2 - 10, cx + w_can//2, cy - h_can//2 + 10], fill=(220, 225, 235), outline=(120, 125, 135), width=2)
    draw.ellipse([cx - 8, cy - h_can//2 - 3, cx + 8, cy - h_can//2 + 5], fill=(150, 155, 165), outline=(80, 85, 95), width=2)
    return img

def _draw_paper(draw, width, height):
    """Draw synthetic paper / sheet / document item."""
    bg_color = (random.randint(200, 220), random.randint(200, 220), random.randint(200, 220))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    margin = random.randint(25, 45)
    paper_color = (random.randint(245, 255), random.randint(245, 255), random.randint(240, 250))
    
    # Sheet contour
    draw.rectangle([margin, margin, width - margin, height - margin], fill=paper_color, outline=(180, 180, 180), width=2)
    
    # Simulated printed text lines
    for y in range(margin + 20, height - margin - 20, 12):
        line_w = random.randint(width // 3, width - margin - 30)
        draw.line([margin + 15, y, margin + 15 + line_w, y], fill=(60, 60, 70), width=3)
        
    # Crease / fold line
    draw.line([margin, height // 3, width - margin, height // 3 + 15], fill=(200, 200, 200), width=2)
    return img

def _draw_plastic(draw, width, height):
    """Draw synthetic plastic bottle / container item."""
    bg_color = (random.randint(220, 240), random.randint(220, 240), random.randint(220, 240))
    img = Image.new("RGB", (width, height), bg_color)
    
    poly_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    p_draw = ImageDraw.Draw(poly_img)
    
    cx, cy = width // 2, height // 2
    w_p, h_p = random.randint(60, 90), random.randint(110, 140)
    
    plastic_color = (random.randint(70, 140), random.randint(160, 210), random.randint(230, 255), 170)
    
    # Plastic container body
    p_draw.rounded_rectangle([cx - w_p//2, cy - h_p//2, cx + w_p//2, cy + h_p//2], radius=15, fill=plastic_color, outline=(40, 100, 180, 230), width=3)
    
    # Bottle Cap
    cap_color = (230, 50, 50, 255) if random.random() > 0.5 else (40, 180, 50, 255)
    p_draw.rectangle([cx - 15, cy - h_p//2 - 15, cx + 15, cy - h_p//2], fill=cap_color)
    
    # Plastic ribbed texture wrinkles
    for y in range(cy - h_p//4, cy + h_p//4, 15):
        p_draw.arc([cx - w_p//2 + 5, y - 5, cx + w_p//2 - 5, y + 5], start=0, end=180, fill=(255, 255, 255, 200), width=2)
        
    img = Image.alpha_composite(img.convert("RGBA"), poly_img).convert("RGB")
    return img

def _draw_trash(draw, width, height):
    """Draw synthetic organic / general waste item."""
    bg_color = (random.randint(190, 210), random.randint(190, 210), random.randint(190, 210))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Irregular organic blob shape
    for _ in range(random.randint(4, 7)):
        rx = random.randint(30, width - 30)
        ry = random.randint(30, height - 30)
        rw = random.randint(25, 55)
        rh = random.randint(25, 55)
        trash_color = (random.randint(80, 130), random.randint(60, 100), random.randint(40, 80))
        draw.ellipse([rx - rw, ry - rh, rx + rw, ry + rh], fill=trash_color, outline=(40, 30, 20), width=2)
        
    # Wrapper noise fragments
    for _ in range(15):
        px = random.randint(20, width - 20)
        py = random.randint(20, height - 20)
        size = random.randint(4, 12)
        draw.rectangle([px, py, px + size, py + size], fill=(random.randint(150, 250), random.randint(50, 250), random.randint(50, 150)))
        
    return img


DRAW_FUNCTIONS = {
    "cardboard": _draw_cardboard,
    "glass": _draw_glass,
    "metal": _draw_metal,
    "paper": _draw_paper,
    "plastic": _draw_plastic,
    "trash": _draw_trash,
}


def generate_synthetic_dataset(samples_per_class=60):
    """
    Generates a structured synthetic dataset of waste images for immediate training & validation.
    
    Args:
        samples_per_class (int): Number of total images to generate per class.
    """
    print(f"[Dataset] Generating synthetic waste material dataset ({samples_per_class * len(CLASSES)} total images)...")
    
    train_ratio = 0.70
    val_ratio = 0.15
    # test_ratio = 0.15
    
    n_train = int(samples_per_class * train_ratio)
    n_val = int(samples_per_class * val_ratio)
    
    for cls in CLASSES:
        # Create subfolders
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / cls).mkdir(parents=True, exist_ok=True)
        
        draw_fn = DRAW_FUNCTIONS[cls]
        
        for i in range(samples_per_class):
            dummy_draw = None
            img = draw_fn(dummy_draw, IMG_WIDTH, IMG_HEIGHT)
            
            # Apply slight random rotation or blur for variation
            angle = random.choice([0, 90, 180, 270, random.randint(-15, 15)])
            if angle != 0:
                img = img.rotate(angle)
                
            if random.random() > 0.7:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
                
            # Determine split folder
            if i < n_train:
                dest_path = TRAIN_DIR / cls / f"{cls}_{i:04d}.png"
            elif i < n_train + n_val:
                dest_path = VAL_DIR / cls / f"{cls}_{i:04d}.png"
            else:
                dest_path = TEST_DIR / cls / f"{cls}_{i:04d}.png"
                
            img.save(dest_path)
            
    print("[Dataset] Synthetic dataset generation complete!")


def get_data_pipelines():
    """
    Loads dataset folders into tf.data.Dataset with data augmentation and EfficientNet preprocessing.
    
    Returns:
        train_ds, val_ds, test_ds
    """
    if not any(TRAIN_DIR.iterdir()):
        generate_synthetic_dataset()
        
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASSES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASSES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASSES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False
    )
    
    # EfficientNet preprocessing layer (normalizes inputs according to EfficientNet requirements)
    preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    
    train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    generate_synthetic_dataset(samples_per_class=60)
