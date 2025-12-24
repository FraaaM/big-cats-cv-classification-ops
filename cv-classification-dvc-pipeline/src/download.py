import pandas as pd
import requests
import os
import yaml
from tqdm import tqdm

def download():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    print("--- Стадия: Загрузка изображений ---")
    df = pd.read_csv(params['data']['clean_csv'])
    output_dir = params['data']['images_dir']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session = requests.Session()
    success_count = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_url = row['image_url']
        class_name = row['scientific_name'].replace(" ", "_")
        img_id = row['id']

        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        file_path = os.path.join(class_dir, f"{img_id}.jpg")

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            success_count += 1
            continue

        try:
            response = session.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
        except Exception:
            pass

    print(f"Скачано изображений: {success_count}/{len(df)}")

if __name__ == "__main__":
    download()