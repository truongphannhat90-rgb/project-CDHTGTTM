import cv2
import numpy as np

def detect_lanes(frame):
    height, width = frame.shape[:2]

    # THIẾT LẬP 4 ĐIỂM TẠO THÀNH VÙNG LÀN ĐƯỜNG 
    top_left     = [int(width * 0.35), int(height * 0.45)]
    top_right    = [int(width * 0.61), int(height * 0.45)] # Điểm x1_fixed
    bottom_right = [int(width * 0.85), height]             # Điểm x2_fixed 
    bottom_left  = [int(width * 0.10), height]

    lane_polygon = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)

    # VẼ VÙNG LÀN ĐƯỜNG TRONG SUỐT
    overlay = frame.copy()
    cv2.fillPoly(overlay, [lane_polygon], (255, 100, 0)) # Màu xanh dương nhạt
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Vẽ đường viền
    cv2.polylines(frame, [lane_polygon], isClosed=True, color=(255, 0, 255), thickness=3)
    
    cv2.putText(frame, "LAN DUONG THEO DOI", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return lane_polygon, frame
