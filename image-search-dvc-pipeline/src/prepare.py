import pandas as pd
import yaml
import os
from utils import load_params

def generalize_name(name):
    if pd.isna(name): return name
    return " ".join(name.split()[:2])

def prepare():
    params = load_params()
    raw_path = params['data']['raw_file']
    clean_path = params['data']['clean_csv']

    print("--- Стадия: Подготовка данных ---")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Исходный файл {raw_path} не найден. Положите observations.csv в папку raw_data/")

    df = pd.read_csv(raw_path)
    
    keep_columns = ['id', 'image_url', 'scientific_name', 'taxon_id']
    existing_cols = [col for col in keep_columns if col in df.columns]
    df = df[existing_cols].copy()
    
    df['scientific_name'] = df['scientific_name'].apply(generalize_name)
    allowed = params['data']['allowed_species']
    df = df[df['scientific_name'].isin(allowed)]
    df = df.dropna(subset=['image_url', 'scientific_name'])
    
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    
    print(f"Готово. Сохранено записей: {len(df)}")
    print(f"Файл: {clean_path}")

if __name__ == "__main__":
    prepare()