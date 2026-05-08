import cv2
import numpy as np
from ultralytics import YOLO

from lane import LaneDetectorHough
from tracker import CentroidTracker
from violation import check_and_draw_violations

class VehicleDetector:

    def __init__(self):

        # Load model YOLOv8
        self.model = YOLO("yolov8n.pt")

        # Khởi tạo bộ phát hiện làn đường
        self.lane_detector = LaneDetectorHough()

        # Khởi tạo tracker
        self.tracker = CentroidTracker(max_disappeared=30)

        # Các class phương tiện trong COCO
        self.vehicle_classes = [2, 3, 5, 7]

        """
        2 = car
        3 = motorbike
        5 = bus
        7 = truck
        """

    def process_frame(self, frame):

        # Danh sách detection
        detections = []

        # ==================================
        # PHÁT HIỆN PHƯƠNG TIỆN
        # ==================================
        results = self.model.predict(
            frame,
            conf=0.5,
            verbose=False
        )

        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Lấy class
                cls = int(box.cls[0])

                # Độ tin cậy
                conf = float(box.conf[0])

                # Chỉ lấy phương tiện giao thông
                if cls not in self.vehicle_classes:
                    continue

                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append([
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls
                ])

        # ==================================
        # TRACKING PHƯƠNG TIỆN
        # ==================================
        object_ids = self.tracker.update(detections)

        tracked_objects = {}

        for oid in object_ids:

            if oid in self.tracker.boxes:
                tracked_objects[oid] = self.tracker.boxes[oid]

        # ==================================
        # PHÁT HIỆN LÀN ĐƯỜNG
        # ==================================
        left_line, right_line = self.lane_detector.detect_lanes(frame)

        lane_lines = []

        # Vẽ làn trái
        if left_line is not None:

            lane_lines.append(left_line)

            cv2.line(
                frame,
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                (0, 255, 0),
                5
            )

        # Vẽ làn phải
        if right_line is not None:

            lane_lines.append(right_line)

            cv2.line(
                frame,
                (right_line[0], right_line[1]),
                (right_line[2], right_line[3]),
                (255, 0, 0),
                5
            )

        # ==================================
        # HIỂN THỊ TRACKING
        # ==================================
        for obj_id, box in tracked_objects.items():

            x1, y1, x2, y2 = map(int, box)

            # Vẽ bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            # Hiển thị ID
            cv2.putText(
                frame,
                f"ID {obj_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # ==================================
        # KIỂM TRA SAI LÀN
        # ==================================
        violations, frame = check_and_draw_violations(
            frame,
            tracked_objects,
            lane_lines
        )

        # ==================================
        # HIỂN THỊ SỐ LƯỢNG VI PHẠM
        # ==================================
        cv2.putText(
            frame,
            f"So vi pham: {len(violations)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        return frame
