# -*- coding:utf-8 -*-


class Line:
    def __init__(self, line):
        [x1, y1, x2, y2] = line
        self.li = line
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)

        if x2 - x1 != 0:
            self.k = float((y2 - y1)) / float((x2 - x1))
            self.b = int(y1 - self.k * x1)
        else:
            self.k = None
            self.b = x1

        # p1 --> p2 从左到右，从上到下
        if self.k is None:
            if self.p1[1] > self.p2[1]:
                self.li = (self.p2[0], self.p2[1], self.p1[0], self.p1[1])
                self.p1, self.p2 = self.p2, self.p1
        else:
            if self.p1[0] > self.p2[0]:
                self.li = (self.p2[0], self.p2[1], self.p1[0], self.p1[1])
                self.p1, self.p2 = self.p2, self.p1

        self.length = self.get_points_dist()

    def get_points_dist_square(self):
        return (self.p1[0] - self.p2[0]) ** 2 + (self.p1[1] - self.p2[1]) ** 2

    def get_points_dist(self):
        dist_sq = self.get_points_dist_square()
        return dist_sq ** 0.5

    def __str__(self):
        return str(self.li)


def is_parallel(line1, line2):
    if line1.k is None and line2.k is None:
        return True
    elif line1.k is not None and line2.k is not None:
        return abs(line1.k - line2.k) < 0.000001
    else:
        return False


def get_point_of_intersection(line1, line2):
    if is_parallel(line1, line2):
        return None
    else:
        if line1.k is None:
            p_x = line1.b
            p_y = int(line2.k * p_x + line2.b)
            point = (p_x, p_y)
            return point
        if line2.k is None:
            p_x = line2.b
            p_y = int(line1.k * p_x + line1.b)
            point = (p_x, p_y)
            return point

        p_x = (line1.b - line2.b) / (line2.k - line1.k)
        p_y = line1.k * p_x + line1.b
        point = (int(p_x), int(p_y))
        return point


def points_to_line(p1, p2):
    return Line([p1[0], p1[1], p2[0], p2[1]])


class Vector:
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point[0] - start_point[0]
        self.y = end_point[1] - start_point[1]

    def get_cos_x(self):
        return self.x / (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self):
        return "({}, {})".format(self.x, self.y)
