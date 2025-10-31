from ultralytics import YOLO
import torch
import os
from pathlib import Path

class YOLOTrainer:
    def __init__(self, data_path, epochs, imgsz, batch_size, name, device, optimizer):
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.name = name or f"yolo_detection_{os.path.basename(data_path)}"
        self.device = device
        self.optimizer = optimizer
        
        self.model = YOLO('yolov8n.pt')
        
    def train(self):
        results = self.model.train(
            data=self._prepare_data_config(),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            name=self.name,
            device=self.device,
            optimizer=self.optimizer,
            lr0=0.01,
            patience=10,
            save=True,
            exist_ok=True
        )
        
        return self._extract_metrics(results)
    
    def _prepare_data_config(self):
        images_dir = self.data_path / 'images'
        labels_dir = self.data_path / 'labels'
        
        if (images_dir / 'train').exists() and (images_dir / 'val').exists():
            return {
                'train': str(images_dir / 'train'),
                'val': str(images_dir / 'val'),
                'test': str(images_dir / 'test') if (images_dir / 'test').exists() else None
            }
        else:
            return {
                'train': str(images_dir),
                'val': str(images_dir)
            }
    
    def _extract_metrics(self, results):
        if hasattr(results, 'box'):
            return {
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.p,
                'recall': results.box.r
            }
        return {'accuracy': 0.0}