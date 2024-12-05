import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Build the new model
input_layer = Input(shape=(224, 224, 3))

# Shared layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Output layers
age_output = Dense(4, activation='softmax', name='age_output')(x)  # 4 age groups
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # Binary gender
skin_output = Dense(2, activation='softmax', name='skin_output')(x)  # 2 skin tones

# Define the model
model = Model(inputs=input_layer, outputs=[age_output, gender_output, skin_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss={
        'age_output': 'categorical_crossentropy',
        'gender_output': 'binary_crossentropy',
        'skin_output': 'categorical_crossentropy'
    },
    metrics={
        'age_output': 'accuracy',
        'gender_output': 'accuracy',
        'skin_output': 'accuracy'
    }
)

print("Model built and compiled successfully!")

# Simulate training data for demonstration purposes
num_samples = 100
train_images = np.random.rand(num_samples, 224, 224, 3)
train_ages = to_categorical(np.random.randint(0, 4, size=(num_samples,)), num_classes=4)
train_genders = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)
train_skins = to_categorical(np.random.randint(0, 2, size=(num_samples,)), num_classes=2)

# Train the model (replace with your real dataset)
print("Training the model...")
history = model.fit(
    train_images,
    {'age_output': train_ages, 'gender_output': train_genders, 'skin_output': train_skins},
    epochs=10,
    batch_size=8
)

# Use the model in real-time prediction from webcam
age_labels = ['Baby', 'Child', 'Adult', 'Senior']
gender_labels = ['Male', 'Female']
skin_labels = ['Light Skin', 'Dark Skin']

# Function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to match model input
    normalized_frame = resized_frame / 255.0       # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame for the model
    input_frame = preprocess_frame(frame)

    # Predict using the model
    predictions = model.predict(input_frame)
    age_pred = np.argmax(predictions[0])  # Age group prediction
    gender_pred = int(predictions[1][0] > 0.5)  # Gender prediction
    skin_pred = np.argmax(predictions[2])  # Skin tone prediction

    # Overlay predictions on the frame
    label = f"Age: {age_labels[age_pred]}, Gender: {gender_labels[gender_pred]}, Skin: {skin_labels[skin_pred]}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Prediction', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
