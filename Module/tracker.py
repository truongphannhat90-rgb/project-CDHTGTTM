import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_object_id = 0
        self.objects = {}       # id -> centroid (x, y)
        self.boxes = {}         # id -> [x1, y1, x2, y2]
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return []

        input_centroids = []
        input_boxes = []
        for det in detections:
            x1, y1, x2, y2, _, _ = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            input_boxes.append([x1, y1, x2, y2])

        # Nếu chưa có object nào
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_boxes[i])
            return list(self.objects.keys())

        # Tính ma trận khoảng cách
        D = np.zeros((len(self.objects), len(input_centroids)))
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # Gán nearest
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.boxes[oid] = input_boxes[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Đăng ký object mới
        for col in range(len(input_centroids)):
            if col not in used_cols:
                self._register(input_centroids[col], input_boxes[col])

        # Xóa object mất tích
        for row in range(len(object_centroids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        return list(self.objects.keys())

    def _register(self, centroid, box):
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]