import cv2
from detect import VehicleDetector

def main():
    # Video input
    video_path = "traffic_video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Nếu muốn dùng webcam:
    # cap = cv2.VideoCapture(0)

    detector = VehicleDetector()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (1280, 720))

        # Detect + Tracking + Violation
        output_frame = detector.process_frame(frame)

        # Hiển thị
        cv2.imshow("ITS - Wrong Lane Detection", output_frame)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
