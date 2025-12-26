import pandas as pd
import requests
import os
from tqdm import tqdm
from utils import load_params

def download():
    params = load_params()
    csv_path = params['data']['clean_csv']
    output_dir = params['data']['images_dir']
    
    print("--- Стадия: Загрузка изображений ---")
    df = pd.read_csv(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    
    count = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Загрузка"):
        img_url = row['image_url']
        species = row['scientific_name'].replace(" ", "_")
        img_id = row['id']
        
        species_dir = os.path.join(output_dir, species)
        os.makedirs(species_dir, exist_ok=True)
        
        file_path = os.path.join(species_dir, f"{img_id}.jpg")
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            count += 1
            continue
            
        try:
            response = session.get(img_url, timeout=5)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                count += 1
        except Exception:
            pass
            
    print(f"Всего изображений: {count}")

if __name__ == "__main__":
    download()