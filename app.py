import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path
import gdown
import urllib.request
# Page settings
st.set_page_config(page_title="Vehicle Image Classifier", page_icon="🚗")
st.title("🚗 Vehicle Image Classification")
st.write("Upload a vehicle image to predict its class and confidence score.")

# Paths
MODEL_PATH = Path("models/vehicle_classifier_best.h5")
CLASS_NAMES_PATH = Path("models/class_names.json")


# Google Drive direct download link for your model (.h5)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1l0Y5efRxBRS6usPhLovl2lNHgCwQAo1h"

def ensure_model_file():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        st.info("Model file not found in repo. Downloading from Google Drive... (first run only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
# Load model and class names
@st.cache_resource
def load_model():
    ensure_model_file()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    confidence = float(preds[pred_idx])

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence Score:** {confidence*100:.2f}%")

    # Show all class probabilities
    st.subheader("All Class Probabilities")
    prob_data = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    prob_data = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))

    st.bar_chart(prob_data)
