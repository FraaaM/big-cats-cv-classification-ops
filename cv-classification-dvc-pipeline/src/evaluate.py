import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import json
import os
from tqdm import tqdm

def evaluate():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    print("--- Стадия: Оценка ---")
    
    DATA_PATH = params['data']['images_dir']
    IMG_SIZE = params['base']['img_size']
    BATCH_SIZE = params['train']['batch_size']
    SEED = params['base']['random_seed']
    MODEL_PATH = params['train']['model_path']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = ImageFolder(root=DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f"Папка {DATA_PATH} не найдена.")
        return

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    _, _, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    class_names = full_dataset.classes
    print(f"Тестовая выборка: {len(test_subset)} изображений")

    # Загрузка модели
    print("Загрузка модели")
    model = models.efficientnet_v2_s(weights=None) 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена. Сначала запустите train.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Инференс (Сбор предсказаний)
    all_preds = []
    all_labels = []

    print("Запуск предсказаний")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Расчет метрик
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    print(f"Final Test Accuracy: {acc:.4f}")

    metrics = {
        "test_accuracy": acc,
        "weighted_f1": report['weighted avg']['f1-score'],
        "macro_precision": report['macro avg']['precision']
    }
    
    with open("results/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    with open("results/classification_report.txt", "w") as f:
        f.write(report_text)

    #  Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    print("Результаты сохранены: evaluation_metrics.json, confusion_matrix.png")
    print("Найти их можно в папке results")

if __name__ == "__main__":
    evaluate()