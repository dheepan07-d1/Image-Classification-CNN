import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

# Title
st.title("🐱 Cat vs Dog Image Classifier 🐶")
st.write("Upload an image and let the CNN model predict whether it is a **Cat** or a **Dog**!")

# Load trained model
model = tf.keras.models.load_model("cats_dogs_model.keras")

# Class labels
class_names = ["Cat", "Dog"]

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128, 128))
    img = img.convert("RGB")

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = np.max(prediction) * 100
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")

    st.info(f"Confidence: **{confidence:.2f}%**")