from app2 import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the Flask app
app = Flask(__name__)

# Build and compile the model (same as in your previous code)
input_layer = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
age_output = Dense(4, activation='softmax', name='age_output')(x)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
skin_output = Dense(2, activation='softmax', name='skin_output')(x)
model = Model(inputs=input_layer, outputs=[age_output, gender_output, skin_output])
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

# Simulate training for demo purposes (use real training in practice)
num_samples = 10
train_images = np.random.rand(num_samples, 224, 224, 3)
train_ages = tf.keras.utils.to_categorical(np.random.randint(0, 4, size=(num_samples,)), num_classes=4)
train_genders = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)
train_skins = tf.keras.utils.to_categorical(np.random.randint(0, 2, size=(num_samples,)), num_classes=2)
model.fit(train_images, {'age_output': train_ages, 'gender_output': train_genders, 'skin_output': train_skins}, epochs=1)

# Labels
age_labels = ['Baby', 'Child', 'Adult', 'Senior']
gender_labels = ['Male', 'Female']
skin_labels = ['Light Skin', 'Dark Skin']

# Function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

# Generator function to capture webcam feed and predict in real-time
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            break

        input_frame = preprocess_frame(frame)
        predictions = model.predict(input_frame)

        age_pred = np.argmax(predictions[0])
        gender_pred = int(predictions[1][0] > 0.5)
        skin_pred = np.argmax(predictions[2])

        label = f"Age: {age_labels[age_pred]}, Gender: {gender_labels[gender_pred]}, Skin: {skin_labels[skin_pred]}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Define Flask routes
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
