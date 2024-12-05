import os
from app2 import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('models/age_gender_skin_model.h5')

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is part of the request
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Preprocess image and predict
            img_array = preprocess_image(filename)

            # Predict age, gender, and skin
            predictions = model.predict(img_array)
            print("Predictions:", predictions)

            # Debugging: check output shapes and values
            print("Age prediction:", predictions[0])  # Raw probabilities
            print("Gender prediction:", predictions[1])
            print("Skin prediction:", predictions[2])

            # Get predicted categories
            age_pred = np.argmax(predictions[0])  # For age
            gender_pred = 1 if predictions[1][0][0] > 0.5 else 0  # Threshold for binary gender
            print("Raw gender probabilities:", predictions[1])
            print("Predicted gender index:", gender_pred)

            skin_pred = np.argmax(predictions[2])  # For skin tone

            # Map predictions to labels
            age_label = ["baby", "child", "teen", "adult"]
            gender_label = ["Male", "Female"]  # Fixed mapping for gender
            skin_label = ["Light", "Dark"]

            # Handle out-of-bound errors
            try:
                age = age_label[age_pred]
            except IndexError:
                age = "Unknown"  # Default if prediction is out of bounds
                print("Warning: Age index out of range.")

            try:
                gender = gender_label[gender_pred]  # Corrected label mapping
            except IndexError:
                gender = "Unknown"  # Default if prediction is out of bounds
                print("Warning: Gender index out of range.")

            try:
                skin = skin_label[skin_pred]
            except IndexError:
                skin = "Unknown"  # Default if prediction is out of bounds
                print("Warning: Skin index out of range.")

            return render_template('index.html', filename=file.filename, age=age, gender=gender, skin=skin)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
