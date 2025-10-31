import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

def detect_task_type(data_path):
    data_path = Path(data_path)
    
    if (data_path / 'images').exists() and (data_path / 'labels').exists():
        return 'detection'
    
    if any(data_path.iterdir()):
        first_item = next(data_path.iterdir())
        if first_item.is_dir():
            return 'classification'
    
    return 'unknown'

def validate_dataset(data_path, task_type):
    data_path = Path(data_path)
    
    if task_type == 'detection':
        return _validate_detection_dataset(data_path)
    elif task_type == 'classification':
        return _validate_classification_dataset(data_path)
    
    return False

def _validate_detection_dataset(data_path):
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Detection dataset must have 'images' and 'labels' directories")
        return False
    
    image_files = list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.png'))
    label_files = list(labels_dir.rglob('*.txt'))
    
    if len(image_files) == 0:
        print("❌ No images found in dataset")
        return False
    
    print(f"✅ Found {len(image_files)} images and {len(label_files)} labels")
    return True

def _validate_classification_dataset(data_path):
    class_folders = [f for f in data_path.iterdir() if f.is_dir()]
    
    if len(class_folders) == 0:
        print("❌ Classification dataset must have class folders")
        return False
    
    total_images = 0
    for class_folder in class_folders:
        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        total_images += len(images)
        print(f"   {class_folder.name}: {len(images)} images")
    
    if total_images == 0:
        print("❌ No images found in class folders")
        return False
    
    print(f"✅ Found {len(class_folders)} classes with {total_images} total images")
    return True

def run_prediction(model_path, image_path):
    if 'yolo' in model_path.lower():
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image_path)
        results[0].show()
    else:
        model = torch.load(model_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            print(f"📊 Prediction: Class {predicted.item()}")