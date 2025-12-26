import torch
import torch.nn as nn
from torchvision import models, transforms
import yaml
import os

def load_params():
    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError("params.yaml не найден!")

def load_embedding_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Архитектура
    model = models.efficientnet_v2_s(weights=None)
    
    # 2. Веса
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл {weights_path} не найден!")
        
    state_dict = torch.load(weights_path, map_location=device)
    
    # очистка от классификатора
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    model.load_state_dict(filtered_dict, strict=False)
    
    # 3. Feature Extractor
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model, device

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])