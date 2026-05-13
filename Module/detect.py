from ultralytics import YOLO

class VehicleDetector:

    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(
            frame,
            conf=0.5, 
            iou=0.45,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        detections = []
        height, width = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                # Lấy tọa độ chuẩn xác
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                
            
                
                detections.append([x1, y1, x2, y2, cls])

        return detections
