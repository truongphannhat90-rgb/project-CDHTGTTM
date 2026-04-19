import cv2
import numpy as np

def is_in_wrong_lane(cx, lane_lines):
    """Logic sai làn (demo: bên trái đường phân làn = sai làn)."""
    if not lane_lines:
        return False

    divider_xs = [(line[0] + line[2]) / 2 for line in lane_lines]
    divider = np.mean(divider_xs)

    return cx < divider   # ← chỉnh dòng này nếu cần

def check_and_draw_violations(frame, tracked_objects, lane_lines):
    violations = []
    for obj_id, box in tracked_objects.items():
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)

        if is_in_wrong_lane(cx, lane_lines):
            violations.append(obj_id)
            cv2.putText(frame, "SAI LÀN!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    return violations, frame