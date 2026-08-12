import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Starting Car Colour Model Training...")

# Paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# Settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Classes:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

# CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training started...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save model
model_path = os.path.join(MODEL_DIR, "car_colour_model.keras")
model.save(model_path)

# Save class names
class_names = {
    str(value): key
    for key, value in train_data.class_indices.items()
}

with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(class_names, f, indent=4)

print("=" * 50)
print("Training Completed Successfully!")
print("Model saved:", model_path)
print("Classes saved: model/classes.json")
print("=" * 50)