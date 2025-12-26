import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import faiss
import numpy as np
import json
import os
from tqdm import tqdm
from utils import load_params, load_embedding_model, get_transform

def build():
    params = load_params()
    data_dir = params['data']['images_dir']
    weights_path = params['model']['weights_path']
    index_path = params['output']['index_file']
    paths_path = params['output']['paths_file']
    
    print("--- Стадия: Создание базы FAISS ---")
    
    model, device = load_embedding_model(weights_path)
    transform = get_transform(params['base']['img_size'])
    
    try:
        dataset = ImageFolder(data_dir, transform=transform)
    except:
        print("Датасет пуст или структура папок неверна.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Инференс
    embeddings = []
    paths = []
    
    print(f"Векторизация {len(dataset)} файлов...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader)):
            images = images.to(device)
            output = model(images)
            embeddings.append(output.cpu().numpy())
            
            start = i * 32
            end = start + len(images)
            batch_paths = [os.path.relpath(x[0], start=data_dir) for x in dataset.samples[start:end]]
            paths.extend(batch_paths)
            
    # FAISS
    emb_matrix = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(emb_matrix)
    
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(paths_path, 'w') as f:
        json.dump(paths, f)
        
    print(f"Индекс готов: {index_path}")

if __name__ == "__main__":
    build()