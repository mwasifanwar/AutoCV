import argparse
import os
import sys
from pathlib import Path
import torch

from trainers.yolo_trainer import YOLOTrainer
from trainers.classifier_trainer import ClassifierTrainer
from utils.data_loader import detect_task_type, validate_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='AutoCV: One-click Computer Vision Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size (auto-detected if not specified)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--device', type=str, default=None, help='Device: cuda:0, cpu, or auto-detect')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer: Adam, SGD, AdamW')
    parser.add_argument('--predict', action='store_true', help='Run inference on trained model')
    parser.add_argument('--model_path', type=str, help='Model path for inference')
    parser.add_argument('--image', type=str, help='Image path for inference')
    return parser.parse_args()

def setup_device(device_arg):
    if device_arg:
        return device_arg
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    args = parse_args()
    
    if args.predict:
        run_inference(args)
        return
    
    print("🚀 AutoCV: Starting automated computer vision training...")
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Error: Dataset path {data_path} does not exist")
        sys.exit(1)
    
    device = setup_device(args.device)
    print(f"📱 Using device: {device}")
    
    print("🔍 Analyzing dataset structure...")
    task_type = detect_task_type(data_path)
    print(f"🎯 Detected task type: {task_type}")
    
    print("✅ Validating dataset...")
    if not validate_dataset(data_path, task_type):
        print("❌ Dataset validation failed")
        sys.exit(1)
    
    if task_type == "detection":
        print("🎯 Training YOLOv8 object detection model...")
        trainer = YOLOTrainer(data_path, args.epochs, args.imgsz or 640, 
                            args.batch, args.name, device, args.optimizer)
    else:
        print("🎯 Training ResNet image classification model...")
        trainer = ClassifierTrainer(data_path, args.epochs, args.imgsz or 224,
                                  args.batch, args.name, device, args.optimizer)
    
    print("🏋️ Starting training...")
    results = trainer.train()
    
    print("📊 Training completed! Results:")
    for metric, value in results.items():
        print(f"   {metric}: {value:.4f}")

def run_inference(args):
    if not args.model_path or not args.image:
        print("❌ For inference, provide both --model_path and --image")
        return
    
    from utils.data_loader import run_prediction
    run_prediction(args.model_path, args.image)

if __name__ == "__main__":
    main()