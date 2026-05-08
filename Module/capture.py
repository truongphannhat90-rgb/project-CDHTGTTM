import cv2

class VideoCapture:

    def __init__(self, source='data/traffic.mp4'):

        print(f"Đang mở: {source}")

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise Exception("Không mở được video!")

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()