import cv2
from ultralytics import YOLO


# Download the model from https://www.kaggle.com/models/ultralytics/yolov8


class HLEngineYOLO8:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        pass

    def detect_objects_from_video_source(self, source):
        if source == "default":
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model.predict(
                    source=frame, imgsz=640, conf=0.5, show=False
                )
                annotated_frame = results[0].plot()
                cv2.imshow("HLEngine-AI-YOLOv8 Real-Time Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
