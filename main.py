import cv2
import os
import datetime

from module.capture import VideoCapture
from module.detect import VehicleDetector
from module.lane import detect_lanes
from module.tracker import CentroidTracker
from module.violation import (
    check_and_draw_violations,
    vehicle_counts  # Dictionary chứa số lượng xe
)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Dọn dẹp file cũ
    for file in ["results/output.avi", "results/violations.txt"]:
        if os.path.exists(file):
            os.remove(file)

    capture = VideoCapture("data/traffic.mp4")
    detector = VehicleDetector()
    tracker = CentroidTracker()

    width = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(
        "results/output.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        frame = capture.get_frame()
        if frame is None:
            break

        frame_count += 1

        # 1. NHẬN DIỆN VÀ TRACKING
        detections = detector.detect(frame)
        tracker.update(detections)
        current_tracked = tracker.boxes

        # 2. KIỂM TRA VI PHẠM & VẼ LÀN ĐƯỜNG

        violations, frame = check_and_draw_violations(
            frame,
            current_tracked,
            detections,
            None 
        )

        # 4. GHI LOG VI PHẠM
        if violations:
            with open("results/violations.txt", "a", encoding="utf-8") as f:
                for v in violations:
                    time_now = datetime.datetime.now().strftime("%H:%M:%S")
                    f.write(f"[{time_now}] Xe ID {v['id']} ({v['type']}) vi pham tai frame {frame_count}\n")

        # 5. XUẤT KẾT QUẢ
        out.write(frame)
        cv2.imshow("He Thong Giam Sat Giao Thong", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ HOAN THANH! Ket qua luu tai thu muc 'results/'")
