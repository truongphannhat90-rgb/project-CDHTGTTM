import cv2
from detect import VehicleDetector

def main():

    # Đường dẫn video
    video_path = "traffic_video.mp4"

    # Đọc video
    cap = cv2.VideoCapture(video_path)

    # Nếu muốn dùng webcam thì mở dòng dưới
    # cap = cv2.VideoCapture(0)

    # Khởi tạo detector
    detector = VehicleDetector()

    while cap.isOpened():

        # Đọc từng frame
        ret, frame = cap.read()

        # Nếu hết video
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (1280, 720))

        # Xử lý detect + tracking + sai làn
        output_frame = detector.process_frame(frame)

        # Hiển thị kết quả
        cv2.imshow("ITS - He Thong Nhan Dien Sai Lan", output_frame)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng bộ nhớ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
