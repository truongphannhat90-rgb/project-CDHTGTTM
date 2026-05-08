import cv2
import os

from module.capture import VideoCapture
from module.detect import VehicleDetector
from module.lane import detect_lanes
from module.tracker import CentroidTracker
from module.violation import (
    check_and_draw_violations,
    vehicle_counts
)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    if os.path.exists("results/output.avi"):
        os.remove("results/output.avi")

    if os.path.exists("results/violations.txt"):
        os.remove("results/violations.txt")

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

    # ===== THAY ĐỔI QUAN TRỌNG: CỐ ĐỊNH LÀN ĐƯỜNG =====
    fixed_lanes = None
    frame_count = 0

    while True:
        frame = capture.get_frame()
        if frame is None:
            break

        frame_count += 1

        # Lấy lane chuẩn ở 10 frame đầu (lúc đường có thể vắng) 
        # Sau đó dùng cố định để không bị nhảy lane theo thân xe
        if frame_count <= 10:
            lanes, frame_with_lanes = detect_lanes(frame)
            if lanes:
                fixed_lanes = lanes

        # DETECT VÀ TRACK XE
        detections = detector.detect(frame)
        tracker.update(detections)
        current_tracked = tracker.boxes

        # KIỂM TRA VI PHẠM (Sử dụng fixed_lanes để triệt để lỗi nhảy lane)
        violations, frame = check_and_draw_violations(
            frame,
            current_tracked,
            detections,
            fixed_lanes if fixed_lanes else []
        )

        # HIỂN THỊ THÔNG TIN
      #  cv2.putText(frame, f"Violations: {len(violations)}", (30, 40),
       #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # GHI LOG
        for v in violations:
            with open("results/violations.txt", "a", encoding="utf-8") as f:
                f.write(f"Xe ID {v['id']} ({v['type']}) vi pham tai frame {frame_count}\n")

        out.write(frame)
        cv2.imshow("Traffic Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ HOAN THANH!")