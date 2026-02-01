import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Configuration
# ---------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"   # data/Cloudy , data/Clear

# ---------------------------
# Data loading
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ---------------------------
# Model definition
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# Training
# ---------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------------------------
# Save model
# ---------------------------
model.save("cloud_detector_model")

print("Training complete. Model saved.")
