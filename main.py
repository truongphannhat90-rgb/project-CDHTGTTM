import cv2
import numpy as np
from ultralytics import YOLO
def main():
    # Khởi tạo hệ thống 
    model = YOLO('yolov8n.pt') 
    video_path = "traffic_video.mp4" 
    cap = cv2.VideoCapture(video_path)
    car_lane_polygon = np.array([[100, 700], [500, 700], [450, 400], [200, 400]], np.int32)
    bike_lane_polygon = np.array([[550, 700], [900, 700], [800, 400], [600, 400]], np.int32)

   # Vòng lặp Frame 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

      # Tiền xử lý 
        frame_resized = cv2.resize(frame, (640, 640))

      # Phát hiện phương tiện
        results = model.predict(frame_resized, conf=0.5)
        
        # Duy trì Tracking ID 
        tracks = model.track(frame_resized, persist=True)

        for result in tracks[0].boxes:
            # Lấy tọa độ 
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cls = int(result.cls[0])
            conf = float(result.conf[0])
            
           
            bottom_center = (int((x1 + x2) / 2), y2)
            
            # xác định vi phạm 
            is_violation = False
            label = "Hợp lệ"
            color = (0, 255, 0) # Màu xanh cho xe đúng làn

            # Kiểm tra Point-in-Polygon
            in_car_lane = cv2.pointPolygonTest(car_lane_polygon, bottom_center, False) >= 0
            in_bike_lane = cv2.pointPolygonTest(bike_lane_polygon, bottom_center, False) >= 0

            
            if cls == 2: 
                if not in_car_lane:
                    is_violation = True
            elif cls == 3:
                if not in_bike_lane:
                    is_violation = True

            if is_violation:
                label = "VI PHAM SAI LAN"
                color = (0, 0, 255) # Màu đỏ cho vi phạm
            
            # 3. Hiển thị kết quả
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

       
        cv2.polylines(frame_resized, [car_lane_polygon], True, (255, 255, 0), 2)
        cv2.polylines(frame_resized, [bike_lane_polygon], True, (0, 255, 255), 2)

        cv2.imshow("Wrong Lane Detection System", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
