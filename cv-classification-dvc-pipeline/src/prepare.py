import pandas as pd
import yaml
import os

def generalize_name(name):
    if pd.isna(name): return name
    return " ".join(name.split()[:2])

def prepare():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    print("--- Стадия: Подготовка данных ---")
    df = pd.read_csv(params['data']['raw_file'])
    
    keep_columns = ['id', 'image_url', 'scientific_name', 'taxon_id']
    df = df[keep_columns].copy()
    df['scientific_name'] = df['scientific_name'].apply(generalize_name)
    
    allowed = params['data']['allowed_species']
    df = df[df['scientific_name'].isin(allowed)]
    
    df = df.dropna(subset=['image_url', 'scientific_name'])
    
    os.makedirs(os.path.dirname(params['data']['clean_csv']), exist_ok=True)
    df.to_csv(params['data']['clean_csv'], index=False)
    
    print(f"Готово. Классов: {len(df['scientific_name'].unique())}")
    print(f"Сохранено в: {params['data']['clean_csv']}")

if __name__ == "__main__":
    prepare()