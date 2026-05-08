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
                
                # --- LOGIC LỌC THÔNG MINH (FIX TRIỆT ĐỂ NHẬN DIỆN SAI) ---
                w_box = x2 - x1
                h_box = y2 - y1
                area_ratio = (w_box * h_box) / (width * height) # Tỉ lệ diện tích xe/khung hình
                aspect_ratio = w_box / (h_box + 0.001)         # Tỉ lệ ngang/cao

                # 1. FIX Ô TÔ NHẦM THÀNH XE MÁY (Trường hợp ID 24 của bạn)
                # Nếu AI bảo xe máy (3) nhưng diện tích > 3% khung hình HOẶC dáng xe quá rộng (ngang/cao > 0.8)
                if cls == 3:
                    if area_ratio > 0.03 or aspect_ratio > 0.8:
                        cls = 2 # Ép về Car (Ô tô)

                # 2. FIX XE MÁY NHẦM THÀNH Ô TÔ (Xe máy ở xa)
                # Nếu AI bảo ô tô (2) nhưng dáng xe cao gầy (ngang/cao < 0.6)
                if cls == 2 and aspect_ratio < 0.6:
                    cls = 3 # Ép về Motorcycle (Xe máy)
                
                detections.append([x1, y1, x2, y2, cls])

        return detections