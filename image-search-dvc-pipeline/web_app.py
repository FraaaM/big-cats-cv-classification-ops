import streamlit as st
import torch
import faiss
import numpy as np
import json
import os
from PIL import Image

from src.utils import load_params, load_embedding_model, get_transform

st.set_page_config(page_title="Visual Search", page_icon="🔍", layout="wide")

@st.cache_resource
def init_app():
    try:
        params = load_params()
    except:
        st.error("params.yaml не найден")
        return None

    weights_path = params['model']['weights_path']
    index_file = params['output']['index_file']
    paths_file = params['output']['paths_file']
    data_root = params['data']['images_dir']
    
    if not os.path.exists(index_file):
        return None, "Индекс не найден. Запустите `dvc repro`"
        
    model, device = load_embedding_model(weights_path)
    index = faiss.read_index(index_file)
    with open(paths_file, 'r') as f:
        paths = json.load(f)
        
    return (model, device, index, paths, data_root, params['base']['img_size']), None

# --- UI ---
init_res = init_app()

if not init_res or init_res[1]:
    st.error(f"Ошибка запуска: {init_res[1] if init_res else 'Unknown'}")
    st.stop()
    
model, device, index, db_paths, data_root, IMG_SIZE = init_res[0]

st.title("🔍 Поиск похожих изображений больших кошек")
uploaded = st.file_uploader("Загрузить фото", type=['jpg', 'png'])

if uploaded:
    c1, c2 = st.columns([1, 2])
    query_img = Image.open(uploaded).convert('RGB')
    
    with c1:
        st.image(query_img, caption="Запрос", use_container_width=True)
        
    # Поиск
    transform = get_transform(IMG_SIZE)
    t = transform(query_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(t).cpu().numpy()
    
    faiss.normalize_L2(emb)
    D, I = index.search(emb, 10)
    
    with c2:
        st.subheader("Результаты")
        cols = st.columns(5)
        for i, idx in enumerate(I[0]):
            if idx == -1: continue
            
            rel_path = db_paths[idx]
            full_path = os.path.join(data_root, rel_path)
            name = os.path.dirname(rel_path).replace("_", " ")
            
            with cols[i % 5]:
                try:
                    img = Image.open(full_path)
                    st.image(img, caption=name, use_container_width=True)
                except:
                    st.error("Файл не найден")