import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

# Download model if not present
MODEL_PATH = "cats_dogs_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=19D8puZdaTFnXSJG83lLVsjllsXVEfCCd"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱"
)

st.title("🐱 Cat vs Dog Image Classifier 🐶")

class_names = ["Cat", "Dog"]

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((128, 128)).convert("RGB")
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")