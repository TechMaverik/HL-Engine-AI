import cv2
import numpy as np
import tensorflow.keras
from PIL import Image

# Load the trained Keras model
model = tensorflow.keras.models.load_model(
    "ai_models/ailab/keras_model.h5", compile=False
)

# Define your class names in the same order as the model's output layer
class_names = [
    "Shashank",
    "Akhil",
    "Divya",
    "Unknown",
]  # Replace with actual class names

# Set up the video capture from the default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))  # Resize according to model input size

    img_array = np.asarray(img)
    normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1  # normalize

    data = np.expand_dims(normalized_img_array, axis=0)  # Batch dimension

    # Predict the class of the image
    prediction = model.predict(data)

    # Get the index of the highest confidence class
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    # Get class name from the list
    class_name = class_names[class_index]

    # Prepare label text
    label = f"{class_name}: {confidence:.2f}"

    # Show label on video frame
    cv2.putText(
        frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
    )

    # Display the frame
    cv2.imshow("Live Classification", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
