import cv2
import numpy as np
import datetime

# Các biến toàn cục
violated_ids = set()
vehicle_counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
counted_ids = set()
history_positions = {} 

STATIONARY_THRESHOLD = 12 
FRAME_WINDOW = 15        
OFFSET = 30 

def get_vehicle_name(cls):
    mapping = {3: "motorcycle", 2: "car", 5: "bus", 7: "truck"}
    return mapping.get(int(cls), "unknown")

def check_and_draw_violations(frame, tracked_objects, detections, lane_lines):
    global violated_ids, vehicle_counts, counted_ids, history_positions
    violations = []
    height, width = frame.shape[:2]

    # --- HÀM TÍNH TOÁN RANH GIỚI ĐỘNG 
    def get_sep_A(y):
        x_top = width * 0.2; x_bottom = width * 0.001; y_top = height * 0.35   
        y = min(max(y, y_top), height)
        return x_top + (x_bottom - x_top) * (y - y_top) / (height - y_top + 0.001)

    def get_sep_B(y):
        x_top = width * 0.55; x_bottom = width * 0.58; y_top = height * 0.35   
        y = min(max(y, y_top), height)
        return x_top + (x_bottom - x_top) * (y - y_top) / (height - y_top + 0.001)

    # --- VẼ KHUNG XÁC ĐỊNH LÀN ĐƯỜNG (ZONES) ---
    overlay = frame.copy()
    y_start, y_end = int(height * 0.35), height
    
    # Tạo các điểm đa giác cho 3 làn
    pts_lane1 = np.array([[0, y_start], [get_sep_A(y_start), y_start], [get_sep_A(y_end), y_end], [0, y_end]], np.int32)
    pts_lane2 = np.array([[get_sep_A(y_start), y_start], [get_sep_B(y_start), y_start], [get_sep_B(y_end), y_end], [get_sep_A(y_end), y_end]], np.int32)
    pts_lane3 = np.array([[get_sep_B(y_start), y_start], [width, y_start], [width, y_end], [get_sep_B(y_end), y_end]], np.int32)

    # Vẽ màu mờ cho từng làn để dễ phân biệt
    cv2.fillPoly(overlay, [pts_lane1], (255, 0, 0))   # Làn 1 (Xanh dương)
    cv2.fillPoly(overlay, [pts_lane2], (0, 255, 0))   # Làn 2 (Xanh lá)
    cv2.fillPoly(overlay, [pts_lane3], (0, 0, 255))   # Làn 3 (Đỏ)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    # Vẽ 2 đường ranh giới 
    cv2.line(frame, (int(get_sep_A(y_start)), y_start), (int(get_sep_A(y_end)), y_end), (255, 0, 255), 3) # Vạch Tím
    cv2.line(frame, (int(get_sep_B(y_start)), y_start), (int(get_sep_B(y_end)), y_end), (0, 255, 255), 3)   # Vạch Vàng

    # --- XỬ LÝ NHẬN DIỆN VÀ VI PHẠM ---
    for obj_id, data in tracked_objects.items():
        if len(data) >= 5: x1, y1, x2, y2, cls = map(int, data)
        else: x1, y1, x2, y2 = map(int, data[:4]); cls = 3
            
        cx, cy = int((x1 + x2) / 2), int(y2) 

        # Vùng miễn trừ & Khử nhiễu đứng yên
        if cy > (height * 0.75) or cx > (width * 0.90): continue

        if obj_id not in history_positions: history_positions[obj_id] = []
        history_positions[obj_id].append((cx, cy))
        
        if len(history_positions[obj_id]) > FRAME_WINDOW:
            old_x, old_y = history_positions[obj_id][0]
            if np.sqrt((cx - old_x)**2 + (cy - old_y)**2) < STATIONARY_THRESHOLD:
                history_positions[obj_id].pop(0); continue
            history_positions[obj_id].pop(0)

        # Kiểm tra loại xe và đếm
        vehicle_name = get_vehicle_name(cls)
        if obj_id not in counted_ids:
            counted_ids.add(obj_id)
            if vehicle_name in vehicle_counts: vehicle_counts[vehicle_name] += 1

        # Logic vi phạm
        is_this_vehicle_violating = False
        sep_A_val = get_sep_A(cy)
        sep_B_val = get_sep_B(cy)

        if cls == 3: # XE MÁY: Sai khi vào Làn 3 (Bên phải vạch B)
            if cx > (sep_B_val + OFFSET): is_this_vehicle_violating = True
        elif cls in [2, 5, 7]: # Ô TÔ: Sai khi vào Làn 1 (Bên trái vạch A)
            if cx < (sep_A_val - OFFSET): is_this_vehicle_violating = True

        # Hiển thị
        color = (0, 0, 255) if is_this_vehicle_violating else (0, 255, 0)
        if is_this_vehicle_violating:
            cv2.putText(frame, "SAI LAN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if obj_id not in violated_ids:
                violations.append({"id": obj_id, "type": vehicle_name, "time": str(datetime.datetime.now())})
                violated_ids.add(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{vehicle_name} ID:{obj_id}", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    return violations, frame
