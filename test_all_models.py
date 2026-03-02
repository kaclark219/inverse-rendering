import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

MODELS_DIR = "./models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
np.random.seed(42)
random.seed(42)

def load_all_models():
    models = {}
    
    model_files = {
        "light_count_detector": "light_count_detector.keras",
        "light_type_classifier": "light_type_classifier.keras",
        "angular_predictor": "angular_predictor.keras",
        "color_power_predictor": "color_power_predictor.keras",
        "tri_angular_predictor": "tri_angular_predictor.keras",
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            try:
                models[model_name] = keras.models.load_model(model_path)
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        else:
            print(f"✗ Model file not found: {model_path}")
    
    return models

def load_label_mapping():
    mapping_path = os.path.join(MODELS_DIR, "light_type_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            return json.load(f)
    return None

def load_and_preprocess_image(path: str, size: tuple = IMG_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

def find_random_test_image():
    test_imgs_dir = "./data/render-lighting/Torus/PlasticGlossy"
    if os.path.exists(test_imgs_dir):
        images = glob.glob(os.path.join(test_imgs_dir, "**/*.png"), recursive=True)
        images += glob.glob(os.path.join(test_imgs_dir, "**/*.jpg"), recursive=True)
        lighting_types = ["Area Light", "Point Light", "Spot Light", "Tri Lighting"]
        filtered_images = [img for img in images if any(lt in img for lt in lighting_types)]
        if filtered_images:
            return random.choice(filtered_images)
        elif images:
            return random.choice(images)
    return None

def test_models(img_path, models, label_mapping):
    print(f"\n{'='*60}")
    print(f"Testing image: {img_path}")
    print(f"{'='*60}\n")
    
    img = load_and_preprocess_image(img_path, IMG_SIZE)
    img_batch = np.expand_dims(img, 0)
    
    img_128 = load_and_preprocess_image(img_path, (128, 128))
    img_128_batch = np.expand_dims(img_128, 0)
    
    count_class = None
    type_class = None
    count_pred = None
    type_pred = None

    if "light_count_detector" in models:
        try:
            count_pred = models["light_count_detector"].predict(img_batch, verbose=0)
            count_class = np.argmax(count_pred)  # 0, 1, 2 for 1, 2, 3 lights
            count_conf = np.max(count_pred) * 100
            print(f"Light Count Detector:")
            print(f"  Predicted: {count_class + 1} light(s)")
            print(f"  Confidence: {count_conf:.2f}%")
            print(f"  Raw predictions: {count_pred[0]}")
        except Exception as e:
            print(f"Light Count Detector - Error: {e}")

    if "light_type_classifier" in models and label_mapping:
        try:
            type_pred = models["light_type_classifier"].predict(img_batch, verbose=0)
            type_class = np.argmax(type_pred)
            type_name = label_mapping["idx_to_light_type"][str(type_class)]
            type_conf = np.max(type_pred) * 100
            print(f"\nLight Type Classifier:")
            print(f"  Predicted: {type_name}")
            print(f"  Confidence: {type_conf:.2f}%")
            print(f"  Raw predictions: {type_pred[0]}")
        except Exception as e:
            print(f"Light Type Classifier - Error: {e}")

    if "angular_predictor" in models:
        print(f"\nAngular Predictor - Skipped")
        print(f"  (Requires tabular metadata matching training preprocessing)")

    if "color_power_predictor" in models:
        try:
            color_power_pred = models["color_power_predictor"].predict(img_128_batch, verbose=0)
            if isinstance(color_power_pred, list):
                color_power_pred = color_power_pred[0]
            print(f"\nColor Power Predictor:")
            print(f"  Output shape: {color_power_pred.shape}")
            print(f"  Raw predictions: {color_power_pred[0]}")
        except Exception as e:
            print(f"Color Power Predictor - Error: {e}")

    if "tri_angular_predictor" in models:
        print(f"\nTri Angular Predictor - Skipped")
        print(f"  (Requires tabular metadata matching training preprocessing)")

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Test Image: {os.path.basename(img_path)}")
    plt.tight_layout()
    plt.savefig("test_result.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved visualization to test_result.png")

def main():
    print("\n" + "="*60)
    print("Testing All Trained Models")
    print("="*60 + "\n")
    
    print("Loading models...")
    models = load_all_models()
    
    if not models:
        print("\nNo models found! Make sure you've trained and saved the models.")
        return
    
    print(f"\nLoaded {len(models)} model(s)\n")

    label_mapping = load_label_mapping()

    print("Searching for test image...")
    img_path = find_random_test_image()
    
    if not img_path:
        print("No test images found in testing-imgs or render-lighting directories!")
        return
    
    print(f"Found test image: {img_path}\n")

    test_models(img_path, models, label_mapping)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()