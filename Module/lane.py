
import cv2
import numpy as np

class LaneDetectorHough:
    def __init__(self):
        self.left_line = None
        self.right_line = None
    
    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        h, w = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, h), (w, h), (w, h//2), (0, h//2)]])
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, 
                                threshold=50, minLineLength=100, maxLineGap=150)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                if slope < -0.3:  # Vạch trái
                    left_lines.append(line)
                elif slope > 0.3:  # Vạch phải
                    right_lines.append(line)
        
        # Lấy đường trung bình
        self.left_line = self._average_lines(left_lines) if left_lines else self.left_line
        self.right_line = self._average_lines(right_lines) if right_lines else self.right_line
        
        return self.left_line, self.right_line
    
    def _average_lines(self, lines):
        if not lines:
            return None
        x1_avg = int(np.mean([l[0][0] for l in lines]))
        y1_avg = int(np.mean([l[0][1] for l in lines]))
        x2_avg = int(np.mean([l[0][2] for l in lines]))
        y2_avg = int(np.mean([l[0][3] for l in lines]))
        return (x1_avg, y1_avg, x2_avg, y2_avg)
    
    def get_lane_center(self):
        if self.left_line and self.right_line:
            lx1, ly1, lx2, ly2 = self.left_line
            rx1, ry1, rx2, ry2 = self.right_line
            center_x = (lx1 + rx1) // 2
            return center_x
        return None
    
    def is_in_lane(self, point):
        if self.left_line and self.right_line:
            x, y = point
            lx1, ly1, lx2, ly2 = self.left_line
            rx1, ry1, rx2, ry2 = self.right_line
            
            # Nội suy x tại y
            left_x = int(lx1 + (lx2 - lx1) * (y - ly1) / (ly2 - ly1 + 0.001))
            right_x = int(rx1 + (rx2 - rx1) * (y - ry1) / (ry2 - ry1 + 0.001))
            
            return left_x <= x <= right_x
        return True
