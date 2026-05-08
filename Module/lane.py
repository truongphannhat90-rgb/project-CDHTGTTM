import numpy as np

class CentroidTracker:

    def __init__(self, max_distance=70):
        self.nextObjectID = 0
        self.boxes = {} # Lưu {id: [x1, y1, x2, y2, cls]}
        self.max_distance = max_distance

    def update(self, detections):
        new_boxes = {}

        for det in detections:
            # Lấy đủ 5 tham số bao gồm cả Class ID (cls)
            x1, y1, x2, y2, cls = det

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            matched_id = None
            min_distance = float("inf")

            for obj_id, old_data in self.boxes.items():
                # Lấy dữ liệu cũ (old_data bây giờ có 5 phần tử)
                ox1, oy1, ox2, oy2, old_cls = old_data[:5]

                ocx = int((ox1 + ox2) / 2)
                ocy = int((oy1 + oy2) / 2)

                distance = np.sqrt((cx - ocx)**2 + (cy - ocy)**2)

                # CẢI TIẾN: Chỉ match nếu cùng loại xe (cls == old_cls)
                if cls == old_cls and distance < self.max_distance and distance < min_distance:
                    matched_id = obj_id
                    min_distance = distance

            if matched_id is not None:
                # Cập nhật xe cũ, giữ nguyên ID và lưu cả cls mới
                new_boxes[matched_id] = [x1, y1, x2, y2, cls]
            else:
                # Tạo xe mới và lưu kèm cls
                new_boxes[self.nextObjectID] = [x1, y1, x2, y2, cls]
                self.nextObjectID += 1

        # Cập nhật danh sách boxes chính
        self.boxes = new_boxes