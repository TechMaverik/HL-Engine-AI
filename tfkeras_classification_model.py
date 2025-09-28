import os
import cv2
import numpy as np
import tensorflow.keras
from PIL import Image

"""
USAGE

class_names=["class1",........]
modelpath = "ai_models/keras_model.h5"
payload = "test_dataset_folder"
video_source = "sample.mp4"
HLEngineTFClassModel(class_names=class_names, model_path=modelpath).launch_classification(payload, True) # For Image
HLEngineTFClassModel(class_names=class_names, model_path=modelpath).launch_classification(video_source, False, True)

"""


class HLEngineTFClassModel:

    def __init__(self, class_names: list, model_path: str):
        self.model = tensorflow.keras.models.load_model(model_path, compile=False)
        self.class_names = class_names

    def run_classification_model_on_image(
        self,
        image_folder_path: str,
        display_image: bool,
    ):
        img = Image.open(image_folder_path).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.asarray(img)
        normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
        data = np.expand_dims(normalized_img_array, axis=0)

        prediction = self.model.predict(data)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        class_name = self.class_names[class_index]
        if display_image is True:
            img_cv = cv2.imread(image_folder_path)
            cv2.putText(
                img_cv,
                f"{class_name}: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Classification", img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return {
                "image_path": image_folder_path,
                "class_name": class_name,
                "confidence": confidence,
            }

    def run_classification_model_on_video(self, video_source: str):
        if video_source == "default":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((224, 224))

            img_array = np.asarray(img)
            normalized_img_array = (
                img_array.astype(np.float32) / 127.0
            ) - 1  # normalize

            data = np.expand_dims(normalized_img_array, axis=0)  # Batch dimension
            prediction = self.model.predict(data)
            class_index = np.argmax(prediction)
            confidence = prediction[0][class_index]
            class_name = self.class_names[class_index]
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Live Classification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def launch_classification(
        self, source: str, display_image_status: bool, isVideSource: bool
    ):
        if isVideSource is False:
            if os.path.isdir(source):
                for filename in os.listdir(source):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        # classify_image(os.path.join(image_path, filename))
                        self.run_classification_model_on_image(
                            os.path.join(source, filename),
                            display_image_status,
                        )
            else:
                self.run_classification_model_on_image(
                    source,
                    display_image_status,
                )
        else:
            self.run_classification_model_on_video(source)
