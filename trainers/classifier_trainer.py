import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from pathlib import Path
from tqdm import tqdm

class ClassifierTrainer:
    def __init__(self, data_path, epochs, imgsz, batch_size, name, device, optimizer):
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.name = name or f"classification_{os.path.basename(data_path)}"
        self.device = device
        self.optimizer_type = optimizer
        
        self.model = models.resnet18(pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        return {
            'train': transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    
    def _prepare_datasets(self):
        train_dir = self.data_path / 'train'
        val_dir = self.data_path / 'val'
        
        if train_dir.exists() and val_dir.exists():
            train_dataset = datasets.ImageFolder(train_dir, self.transform['train'])
            val_dataset = datasets.ImageFolder(val_dir, self.transform['val'])
        else:
            full_dataset = datasets.ImageFolder(self.data_path, self.transform['train'])
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        
        return train_dataset, val_dataset
    
    def train(self):
        train_dataset, val_dataset = self._prepare_datasets()
        
        num_classes = len(train_dataset.dataset.classes) if hasattr(train_dataset, 'dataset') else len(set([label for _, label in train_dataset]))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        if self.optimizer_type == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        elif self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            accuracy = self._validate(val_loader)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), f'runs/classify/{self.name}/best_model.pth')
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')
        
        return {'accuracy': best_accuracy}
    
    def _validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total