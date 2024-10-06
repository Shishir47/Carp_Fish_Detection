import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO messages

# Step 1: Preprocess Images
def preprocess_images(image_dir, img_size=(128, 128)):
    images = []
    labels = []

    # Loop through Carp fish images
    for img_file in os.listdir(image_dir + '/carp'):
        img_path = os.path.join(image_dir + '/carp', img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize the image
            img = img / 255.0  # Normalize the pixel values
            images.append(img)
            labels.append(1)  # Carp fish label

    # Loop through non-Carp fish images
    for img_file in os.listdir(image_dir + '/non_carp'):
        img_path = os.path.join(image_dir + '/non_carp', img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize the image
            img = img / 255.0  # Normalize the pixel values
            images.append(img)
            labels.append(0)  # Non-Carp fish label

    return np.array(images), np.array(labels)

# Set path to your dataset of Carp fish images
image_dir = "C:/Users/hp/Downloads/Data set of Carp fish"  # Replace with your actual path
X, y = preprocess_images(image_dir)

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=100)

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100}%")

# Step 6: Save the Model
model.save('carp_fish_detection_model.h5')

# Step 7: Load the Model and Make Predictions
def predict_image(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize the input image
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match the input shape
    prediction = model.predict(img)
    return "Carp Fish" if prediction > 0.5 else "Not Carp Fish"

# Load the saved model
model = load_model('carp_fish_detection_model.h5')

# Predict on a new image
result = predict_image("C:/Users/hp/Downloads/jkfgkl.jpeg", model)  # Replace with actual image path
print(result)
