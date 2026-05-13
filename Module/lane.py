import cv2
import numpy as np

def detect_lanes(frame):
    height, width = frame.shape[:2]

    # THIẾT LẬP RANH GIỚI CỐ ĐỊNH (FIXED BOUNDARY)
    
    # Điểm phía xa (Top)
    x1_fixed = int(width * 0.61) 
    y1_fixed = int(height * 0.45)
    
    # Điểm phía gần camera (Bottom)
    x2_fixed = int(width * 0.65)
    y2_fixed = height

    # Lưu vào danh sách theo định dạng cũ để không làm lỗi các hàm khác
    lane_lines = [np.array([[x1_fixed, y1_fixed, x2_fixed, y2_fixed]])]

    #VẼ RANH GIỚI ĐỂ QUAN SÁT 
    # cv2.line(frame, (x1_fixed, y1_fixed), (x2_fixed, y2_fixed), (255, 0, 255), 3)
    # Hiển thị nhãn ranh giới
    cv2.putText(frame, "RANH GIOI CHUAN", (x1_fixed - 50, y1_fixed - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return lane_lines, frame
