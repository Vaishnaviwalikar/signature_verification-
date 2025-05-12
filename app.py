"""
This is a Flask web application that allows users to upload an image of a signature and verify
whether it is genuine or forged using a trained Support Vector Machine (SVM) model.

The application preprocesses the uploaded image by resizing it to 200x100 pixels, thresholding it
using the Otsu thresholding algorithm, and flattening it into a feature vector. The feature vector is
then normalized using a StandardScaler and passed to the SVM model for prediction.

The SVM model is trained using a set of genuine and forged signature images. The model is saved to a
file named "model.pkl" and the scaler is saved to a file named "scaler.pkl".

The application consists of two routes: the home page and the verification page. The home page
renders an HTML template with a form that allows users to upload an image of a signature. The
verification page takes the uploaded image, preprocesses it, and passes it to the SVM model for
prediction. The prediction result is then rendered in an HTML template.

"""

from flask import Flask, render_template,request
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import base64


app = Flask(__name__)
genuine_images_path = "Gen"
forged_images_path = "For"

genuine_image_filenames = os.listdir(genuine_images_path)
forged_image_filenames = os.listdir(forged_images_path)

preprocessed_images = {}

def preprocess_image(image):
    """
    Preprocess the input image by resizing it to 200x100 pixels, thresholding it using the Otsu
    thresholding algorithm, and flattening it into a feature vector.

    Parameters:
        image (numpy array): The input image

    Returns:
        A flattened feature vector
    """
    if isinstance(image, str):  # If it's a path, read it
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    resized_image = cv2.resize(threshold_image, (200, 100))
    return resized_image.flatten()

genuine_image_features = [preprocess_image(os.path.join(genuine_images_path, name)) for name in genuine_image_filenames]
forged_image_features = [preprocess_image(os.path.join(forged_images_path, name)) for name in forged_image_filenames]

# Create labels for the images (1 for genuine, 0 for forged)
genuine_labels = np.ones(len(genuine_image_features))
forged_labels = np.zeros(len(forged_image_features))

# Combine genuine and forged features and labels
all_features = np.concatenate((genuine_image_features, forged_image_features))
all_labels = np.concatenate((genuine_labels, forged_labels))

# Normalize the feature vectors
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, all_labels, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the trained model and scaler
with open("model.pkl", "wb") as file:
    pickle.dump(svm_model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


@app.route('/')
def home():
    """
    Render the home page with a form that allows users to upload an image of a signature.
    """
    return render_template('index.html')

with open("model.pkl", "rb") as file:
    svm_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route('/verify', methods=['POST'])
def verify():
    """
    Take the uploaded image, preprocess it, and pass it to the SVM model for prediction.
    """
    # Get the selected image file from the form
    image_file = request.files['image']

    # Read the image file as a numpy array
    in_memory_file = np.fromstring(image_file.read(), np.uint8)
    image = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    feature_vector = preprocess_image(image)
    scaled_feature = scaler.transform(feature_vector.reshape(1, -1))

    # Make prediction
    prediction = svm_model.predict(scaled_feature)

    # Return the prediction result
    if prediction == 1:
        result = 'The signature is genuine.'
    else:
        result = 'The signature is forged.'

    # Base64 encode the preprocessed image
    _, encoded_image = cv2.imencode('.png', feature_vector.reshape(100, 200))
    preprocessed_image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    # Pass the result and preprocessed image URL to the template
    return render_template('result.html', result=result, preprocessed_image_url=preprocessed_image_base64)


if __name__ == '__main__':
    app.run()

