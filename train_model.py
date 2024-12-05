import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the dataset
df = pd.read_csv('data/dataset.csv')  # Adjust path as needed

# Debugging: Check initial dataset
print("Initial Dataset shape:", df.shape)
print("Sample rows before preprocessing:\n", df.head())

# Categorize age into intervals
def categorize_age(age):
    if 0 <= age <= 10: return 0  # Baby
    elif 11 <= age <= 17: return 1  # Child
    elif 18 <= age <= 64: return 2  # Adult
    elif age >= 65: return 3  # Senior
    else: return None  # Invalid value

# Apply categorization
df['age'] = df['age'].apply(categorize_age)

# Remove rows with invalid age or skin values
df = df[df['age'].notna()]
df = df[df['skin'].isin([0, 1])]

# Debugging: Check unique values in age and skin after processing
print("Unique age values after categorization:", df['age'].unique())
print("Unique skin values:", df['skin'].unique())

# One-hot encode age and skin
df['age'] = list(to_categorical(df['age'], num_classes=4))
df['skin'] = list(to_categorical(df['skin'], num_classes=2))

# Preprocess images
def preprocess_image(image_path):
    full_path = f"data/{image_path}"
    if not os.path.exists(full_path):
        print(f"Image not found: {full_path}")
        return None
    try:
        image = load_img(full_path, target_size=(224, 224))
        return img_to_array(image) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Apply image preprocessing and remove rows with invalid images
df['image'] = df['image'].apply(preprocess_image)
df = df[df['image'].notnull()]

# Debugging: Check the dataset after image processing
print("Dataset shape after image preprocessing:", df.shape)
print("Sample rows after preprocessing:\n", df.head())

# Convert data to numpy arrays
images = np.array(list(df['image']))
ages = np.array(list(df['age']))
genders = np.array(df['gender']).astype(np.float32)
skins = np.array(list(df['skin']))

# Check the shapes of the data
print("Images shape:", images.shape)
print("Ages shape:", ages.shape)
print("Genders shape:", genders.shape)
print("Skins shape:", skins.shape)

# Split data into training and validation sets
if len(images) < 5:  # If dataset is too small for validation split
    print("Dataset too small for validation split. Using all data for training.")
    train_images, train_ages, train_genders, train_skins = images, ages, genders, skins
    val_images, val_ages, val_genders, val_skins = None, None, None, None
else:
    train_images, val_images, train_ages, val_ages, train_genders, val_genders, train_skins, val_skins = train_test_split(
        images, ages, genders, skins, test_size=0.2, random_state=42
    )

# Build the model
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

# Train the model
history = model.fit(
    train_images,
    {'age_output': train_ages, 'gender_output': train_genders, 'skin_output': train_skins},
    validation_data=(val_images, {'age_output': val_ages, 'gender_output': val_genders, 'skin_output': val_skins}) if val_images is not None else None,
    epochs=20,
    batch_size=32
)

# Save the trained model
model.save('models/age_gender_skin_model.h5')

# Save training history plot (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot age accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['age_output_accuracy'], label='Train Age Accuracy')
plt.plot(history.history['val_age_output_accuracy'], label='Val Age Accuracy' if val_images is not None else 'Validation Not Used')
plt.legend()
plt.title('Age Accuracy')

# Plot gender accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['gender_output_accuracy'], label='Train Gender Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Val Gender Accuracy' if val_images is not None else 'Validation Not Used')
plt.legend()
plt.title('Gender Accuracy')

# Plot skin accuracy
plt.subplot(1, 3, 3)
plt.plot(history.history['skin_output_accuracy'], label='Train Skin Accuracy')
plt.plot(history.history['val_skin_output_accuracy'], label='Val Skin Accuracy' if val_images is not None else 'Validation Not Used')
plt.legend()
plt.title('Skin Accuracy')

plt.tight_layout()
plt.savefig('results/training_history.png')
plt.show()
