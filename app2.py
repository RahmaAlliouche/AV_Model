from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for testing purposes

# Load your pretrained model
model = tf.keras.models.load_model("trained_model.h5") 


# Labels for predictions
age_labels = [f"{i*10}-{(i+1)*10-1}" for i in range(10)]  # E.g., "0-9", "10-19", ...
gender_labels = ["Male", "Female"]
skin_labels = ["Light Skin", "Medium Skin", "Dark Skin"]

# Helper function to preprocess a Base64 image
def preprocess_base64_image(base64_image):
    try:
        # Remove the Base64 header if present
        base64_image = base64_image.split(",")[-1]

        # Decode the Base64 string
        image_data = base64.b64decode(base64_image)

        # Convert bytes to a NumPy array and read it as an image
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Resize and normalize the image
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)
    except Exception as e:
        print("Error during image preprocessing:", str(e))
        raise e

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Preprocess the image
        img = preprocess_base64_image(data["image"])

        # Perform predictions
        predictions = model.predict(img)
        age_pred = np.argmax(predictions[0])
        gender_pred = int(predictions[1][0] > 0.5)
        skin_pred = np.argmax(predictions[2])

        # Map predictions to labels
        result = {
            "age": age_labels[age_pred],
            "gender": gender_labels[gender_pred],
            "skin": skin_labels[skin_pred],
        }
        return jsonify(result), 200
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
