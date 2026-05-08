import cv2
import numpy as np

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

    # --- ĐỊNH NGHĨA 2 RANH GIỚI ĐỂ CHIA 3 LÀN ---
    
    # Ranh giới A: Giữa Làn 1 (Trái) và Làn 2 (Giữa)
    def get_sep_A(y):
        x_top = width * 0.2    # Điểm xa bên trái
        x_bottom = width * 0.01 # Điểm gần bên trái
        y_top = height * 0.35   
        y = min(y, height * 0.9)
        return x_top + (x_bottom - x_top) * (y - y_top) / (height - y_top + 0.001)

    # Ranh giới B: Giữa Làn 2 (Giữa) và Làn 3 (Sát vỉa hè)
    def get_sep_B(y):
        x_top = width * 0.55    # Điểm xa sát bồn hoa
        x_bottom = width * 0.58 # Điểm gần sát bồn hoa
        y_top = height * 0.35   
        y = min(y, height * 0.9)
        return x_top + (x_bottom - x_top) * (y - y_top) / (height - y_top + 0.001)

    for obj_id, data in tracked_objects.items():
        is_this_vehicle_violating = False
        
        if len(data) >= 5:
            x1, y1, x2, y2, cls = map(int, data)
        else:
            x1, y1, x2, y2 = map(int, data[:4])
            cls = 3 
            
        cx = int((x1 + x2) / 2)
        cy = int(y2) 


        # Vùng miễn trừ sát mép dưới (Fix lỗi xe máy thoát khung hình)
        if cy > (height * 0.75):
            is_this_vehicle_violating = False 
        
        # Loại bỏ xe bên kia bồn hoa
        elif cx > (width * 0.90):
            continue

        # Kiểm tra xe đứng yên
        if obj_id not in history_positions:
            history_positions[obj_id] = []
        history_positions[obj_id].append((cx, cy))
        
        is_moving = True
        if len(history_positions[obj_id]) > FRAME_WINDOW:
            old_x, old_y = history_positions[obj_id][0]
            distance = np.sqrt((cx - old_x)**2 + (cy - old_y)**2)
            if distance < STATIONARY_THRESHOLD:
                is_moving = False
            history_positions[obj_id].pop(0)

        if not is_moving:
            continue 

        vehicle_name = get_vehicle_name(cls)
        sep_A = get_sep_A(cy)
        sep_B = get_sep_B(cy)

        # 1. ĐẾM XE
        if obj_id not in counted_ids:
            counted_ids.add(obj_id)
            if vehicle_name in vehicle_counts:
                vehicle_counts[vehicle_name] += 1

        # --- LOGIC PHÂN LÀN THEO CÁCH TÍNH CỦA BẠN (Làn 1 bên Phải) ---
        # sep_A: Ranh giới bên TRÁI (Chia làn 2 và 3)
        # sep_B: Ranh giới bên PHẢI (Chia làn 1 và 2)
        
        sep_trai = get_sep_A(cy) # Vạch sát dải phân cách
        sep_phai = get_sep_B(cy) # Vạch sát vỉa hè

                # sep_A là vạch TÍM (Trái), sep_B là vạch VÀNG (Phải)
        current_sep_A = get_sep_A(cy)
        current_sep_B = get_sep_B(cy)

        if cls == 3: # XE MÁY
            # Xe máy ĐÚNG ở Làn 1 & 2 (Bên TRÁI vạch vàng)
            # Xe máy SAI khi vào Làn 3 (Bên PHẢI vạch vàng sep_B)
            if cx > (current_sep_B + OFFSET):
                is_this_vehicle_violating = True
                
        elif cls in [2, 5, 7]: # Ô TÔ, XE BUÝT, XE TẢI
            # Ô tô ĐÚNG ở Làn 2 & 3 (Bên PHẢI vạch tím)
            # Ô tô SAI khi vào Làn 1 (Bên TRÁI vạch tím sep_A)
            if cx < (current_sep_A - OFFSET):
                is_this_vehicle_violating = True


    #    # 2. LOGIC PHÂN LÀN HỖN HỢP
    #    if not is_this_vehicle_violating:
    #        if cls in [2, 5, 7]: # Ô TÔ
    #            # Ô tô được đi làn 1, 2. Sai khi lấn hẳn vào sát lề (Làn 3)
    #            if cx > (sep_B + OFFSET):
    #                is_this_vehicle_violating = True
    #        elif cls == 3: # XE MÁY
    #            # Xe máy được đi làn 2, 3. Sai khi lấn hẳn sang dải phân cách trái (Làn 1)
    #            if cx < (sep_A - 5):
    #                is_this_vehicle_violating = True

        # 3. HIỂN THỊ
        color = (0, 255, 0)
        if is_this_vehicle_violating:
            color = (0, 0, 255)
            cv2.putText(frame, "SAI LAN", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if obj_id not in violated_ids:
                violations.append({"id": obj_id, "type": vehicle_name})
                violated_ids.add(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{vehicle_name} ID:{obj_id}", (x1, y1 - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # ===== VẼ RANH GIỚI ẢO ĐỂ CĂN CHỈNH =====
    # Vẽ Ranh giới A (Màu Tím - Giữa Làn 1 và 2)
    # Lấy tọa độ X tại đỉnh (y_top) và đáy (height) để vẽ đường thẳng dài
  #  xa_top = int(get_sep_A(height * 0.35))
   # xa_bottom = int(get_sep_A(height))
   # cv2.line(frame, (xa_top, int(height * 0.35)), (xa_bottom, height), (255, 0, 255), 3)
   # cv2.putText(frame, "RANH GIOI A", (xa_top, int(height * 0.35) - 10), 
   #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Vẽ Ranh giới B (Màu Vàng - Giữa Làn 2 và 3)
 #   xb_top = int(get_sep_B(height * 0.35))
  #  xb_bottom = int(get_sep_B(height))
  #  cv2.line(frame, (xb_top, int(height * 0.35)), (xb_bottom, height), (0, 255, 255), 3)
  #  cv2.putText(frame, "RANH GIOI B", (xb_top, int(height * 0.35) - 10), 
   #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return violations, frame