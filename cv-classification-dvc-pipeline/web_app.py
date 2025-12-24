import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# !!! Для запуска программы 
# выполнить в терминале streamlit run app.py

NAME_MAPPING = {
    'Panthera leo': 'Лев',
    'Panthera tigris': 'Тигр',
    'Panthera onca': 'Ягуар',
    'Panthera pardus': 'Леопард',
    'Panthera uncia': 'Ирбис (Снежный барс)'
}

def get_common_name(scientific_name):
    clean_name = scientific_name.replace("_", " ")
    return NAME_MAPPING.get(clean_name, clean_name)


st.set_page_config(
    page_title="Big Cats Vision",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_params():
    if os.path.exists("params.yaml"):
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    return None

PARAMS = load_params()

if PARAMS:
    MODEL_PATH = PARAMS['train']['model_path']
    IMG_SIZE = PARAMS['base']['img_size']
    DATA_DIR = PARAMS['data']['images_dir']
else:
    st.error("Файл params.yaml не найден! Используются настройки по умолчанию.")
    MODEL_PATH = "models/best_model.pth"
    IMG_SIZE = 224
    DATA_DIR = "data/dataset_images"

FALLBACK_CLASSES = [
    'Panthera leo', 
    'Panthera onca', 
    'Panthera pardus', 
    'Panthera tigris', 
    'Panthera uncia'
]


@st.cache_data
def get_classes(data_dir):
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if classes:
            return classes
    return sorted(FALLBACK_CLASSES)

@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        if not os.path.exists(model_path):
            return None, f"Файл модели не найден: {model_path}"
            
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.to(device)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- Основной экран ---
st.sidebar.title("Настройки")
st.sidebar.info(
    """
    Это демо-приложение для классификации видов больших кошек (Panthera).
    
    **Модель:** EfficientNet V2 Small
    **Источник данных:** iNaturalist
    """
)

st.sidebar.write("---")
confidence_threshold = st.sidebar.slider("Порог уверенности (%)", 0, 100, 50)

st.title("🦁 Panthera Vision AI")
st.markdown("Загрузите фото **Льва, Тигра, Ягуара, Леопарда или Ирбиса**, и нейросеть определит вид.")

classes = get_classes(DATA_DIR)
# st.write(f"Загружено классов: {len(classes)}") # Дебаг

model, error_msg = load_model(MODEL_PATH, len(classes))

if model is None:
    st.error(f"Ошибка загрузки модели: {error_msg}")
    st.warning("Убедитесь, что вы запустили `dvc repro` и файл модели существует.")
    st.stop()

uploaded_file = st.file_uploader("Перетащите изображение сюда...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='Загруженное фото', use_container_width=True)

    with col2:
        st.subheader("Результат анализа")
        
        with st.spinner('В процессе...'):
            device = next(model.parameters()).device
            input_tensor = preprocess_image(image, IMG_SIZE).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
            
            top_prob, top_idx = torch.topk(probs, 1)
            confidence = top_prob.item() * 100
            pred_class = get_common_name(classes[top_idx.item()])

        if confidence < confidence_threshold:
            st.warning(f"⚠️ Низкая уверенность ({confidence:.2f}%)")
            st.write(f"Модель склоняется к **{pred_class}**, но не уверена.")
        else:
            if confidence > 90:
                st.success(f"🎯 Это точно **{pred_class}**!")
            elif confidence > 75:
                st.info(f"✅ Скорее всего это **{pred_class}**.")
            else:
                st.info(f"🤔 Возможно это **{pred_class}**.")
            
            st.metric("Уверенность модели", f"{confidence:.2f}%")

        st.write("---")
        st.write("**Распределение вероятностей:**")
        
        top5_prob, top5_idx = torch.topk(probs, min(5, len(classes)))
        probs_np = top5_prob.cpu().numpy() * 100
        
        classes_np = [get_common_name(classes[idx]) for idx in top5_idx.cpu().numpy()]
        
        df_probs = pd.DataFrame({
            'Вид': classes_np,
            'Вероятность (%)': probs_np
        })
        
        st.bar_chart(df_probs.set_index('Вид'), color="#4CAF50")

        with st.expander("Показать сырые данные"):
            st.dataframe(df_probs.style.format({"Вероятность (%)": "{:.2f}"}))