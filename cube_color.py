import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# --- Load your trained Keras model ---
model = load_model(
    "/home/developer/Documents/HL-Engine-AI/Cubes/color_cubes/keras_model.h5"
)  # replace with your actual .h5 file path

# --- Define your class labels (adjust to your dataset) ---
class_labels = ["red", "green", "blue", "black", "white"]


# --- Function to preprocess image ---
def preprocess_image(img_path, target_size=(128, 128)):
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)

    # Normalize (0–1 range) if model trained with normalization
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Prediction function ---
def predict_cube_color(img_path):
    processed_img = preprocess_image(
        img_path, target_size=(128, 128)
    )  # adjust size to your training input
    prediction = model.predict(processed_img)

    # If categorical (softmax), take highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    print(f"Prediction: {predicted_label} (Confidence: {np.max(prediction)*100:.2f}%)")
    return predicted_label


# --- Example usage ---
if __name__ == "__main__":
    img_path = "/home/developer/Documents/HL-Engine-AI/Cubes/Dataset/green.jpg"  # replace with the path to your cube image
    predict_cube_color(img_path)
