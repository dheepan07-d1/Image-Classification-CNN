import tensorflow as tf
import numpy as np

# Load saved model
model = tf.keras.models.load_model("cats_dogs_model.keras")

# Load test image
img = tf.keras.utils.load_img(
    "test.jpg",
    target_size=(128, 128)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Predict
prediction = model.predict(img_array)

classes = ["cats", "dogs"]

print("Prediction:", classes[np.argmax(prediction)])