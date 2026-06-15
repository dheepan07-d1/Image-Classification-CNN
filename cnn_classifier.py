import tensorflow as tf
import matplotlib.pyplot as plt

# Load Dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=32
)

# Build CNN Model
model = tf.keras.Sequential([

    tf.keras.Input(shape=(128,128,3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(2,activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    dataset,
    epochs=10
)

# Save Model
model.save("cats_dogs_model.h5")

# Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot Loss Graph
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
history = model.fit(
    dataset,
    epochs=10
)

print("Training Completed!")

model.save("cats_dogs_model.keras")

print("Model Saved Successfully!")
