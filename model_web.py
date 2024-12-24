import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load a pretrained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's layers
base_model.trainable = False

# Add custom layers for age, gender, and skin tone detection
x = Flatten()(base_model.output)
x = Dropout(0.5)(x)  # Add dropout for regularization

# Age prediction: Use 10 classes for finer-grained age predictions (e.g., 0-9, 10-19, ...)
age_output = Dense(10, activation="softmax", name="age_output")(x)

# Gender prediction: Binary classification (0: Male, 1: Female)
gender_output = Dense(1, activation="sigmoid", name="gender_output")(x)

# Skin tone prediction: Assume 3 classes (e.g., Light, Medium, Dark)
skin_output = Dense(3, activation="softmax", name="skin_output")(x)

# Define the updated model
model = Model(inputs=base_model.input, outputs=[age_output, gender_output, skin_output])

# Compile the model
model.compile(
    optimizer="adam",
    loss={
        "age_output": "categorical_crossentropy",
        "gender_output": "binary_crossentropy",
        "skin_output": "categorical_crossentropy",
    },
    metrics={
        "age_output": "accuracy",
        "gender_output": "accuracy",
        "skin_output": "accuracy",
    },
)

print("Updated model built successfully!")

# Simulate training data for demonstration (replace with your real dataset)
num_samples = 200
train_images = np.random.rand(num_samples, 224, 224, 3)  # Random images
train_ages = to_categorical(np.random.randint(0, 10, size=(num_samples,)), num_classes=10)  # 10 age groups
train_genders = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)  # Binary gender
train_skins = to_categorical(np.random.randint(0, 3, size=(num_samples,)), num_classes=3)  # 3 skin tones

# Train the model
print("Training the model...")
history = model.fit(
    train_images,
    {"age_output": train_ages, "gender_output": train_genders, "skin_output": train_skins},
    epochs=5,
    batch_size=16,
)

# Save the trained model to disk
model_path = "trained_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}!")
