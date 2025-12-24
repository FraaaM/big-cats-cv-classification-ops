import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml
import json
import os

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    DATA_PATH = params['data']['images_dir']
    MODEL_PATH = params['train']['model_path']
    IMG_SIZE = params['base']['img_size']
    BATCH_SIZE = params['train']['batch_size']
    LR = params['train']['lr']
    EPOCHS = params['train']['epochs']
    SEED = params['base']['random_seed']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HISTORY_PATH = "results/training_history.json"

    print(f"--- Стадия: Дообучение (Device: {DEVICE}) ---")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        df = ImageFolder(root=DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f"Папка {DATA_PATH} не найдена.")
        return
    
    # Сплит данных (70/15/15)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    test_size = len(df) - train_size - val_size
    
    train_subset, val_subset, _ = random_split(
        df, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Загрузка предобученной модели EfficientNetV2...")
    model = models.efficientnet_v2_s(weights='DEFAULT')
    
    num_classes = len(df.classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    history = []
    best_acc = 0.0

    # Цикл обучения
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        
        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss_sum / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Логирование
        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": avg_val_loss
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"-- Новая наилучшая модель сохранена в {MODEL_PATH}")
    
    print(f"Сохраняем графики в {HISTORY_PATH}...")
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    print("Обучение завершено успешно.")

if __name__ == "__main__":
    train()