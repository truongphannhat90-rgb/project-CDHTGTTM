from shapely.geometry import Point, Polygon

class LaneChecker:

    def __init__(self, moto_polygon, car_polygon):

        # Tạo polygon cho từng làn
        self.moto_lane = Polygon(moto_polygon)
        self.car_lane = Polygon(car_polygon)

        # Buffer giúp tránh lỗi point nằm đúng mép lane
        self.buffer_size = 5

    def check_violation(self, bbox, class_id, moto_classes, car_classes):
        x1, y1, x2, y2 = map(float, bbox)
        cx = (x1 + x2) / 2
        cy = y2
        point = Point(cx, cy)

        is_moto = class_id in moto_classes
        is_car = class_id in car_classes
        
        in_moto_lane = (self.moto_lane.buffer(self.buffer_size).contains(point))
        in_car_lane = (self.car_lane.buffer(self.buffer_size).contains(point))

        if is_moto and in_car_lane:
            return (True, "moto_in_car_lane")
        elif is_car and in_moto_lane:
            return (True, "car_in_moto_lane")
        else:
            return (False,"ok")
