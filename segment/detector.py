import cv2 as cv
import numpy as np
import os
from collections import defaultdict
from functools import cmp_to_key

import helper.detector_helper as helper
from classifier import Classifier
import cfg
from img_preprocess import pre_process
import pools_detector
import model_exporter
import translator

input_img = []

pools = []
all_elements = []


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

        if self.k is None:
            if self.p1[1] > self.p2[1]:
                self.li = (self.p2[0], self.p2[1], self.p1[0], self.p1[1])
                self.p1, self.p2 = self.p2, self.p1
        else:
            if self.p1[0] > self.p2[0]:
                self.li = (self.p2[0], self.p2[1], self.p1[0], self.p1[1])
                self.p1, self.p2 = self.p2, self.p1

        self.length = get_points_dist(self.p1, self.p2)


class Vector:
    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point[0] - start_point[0]
        self.y = end_point[1] - start_point[1]

    def get_cos_x(self):
        return self.x / (self.x ** 2 + self.y ** 2) ** 0.5

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


def cross_product(v1, v2):
    return int(v1.x) * int(v2.y) - int(v2.x) * int(v1.y)


def is_collinear(v1, v2):
    if v1.x == 0 and v1.y == 0:
        return False
    if v2.x == 0 and v2.y == 0:
        return False

    return cross_product(v1, v2) == 0


def is_opposite(v1, v2):
    if is_collinear(v1, v2):
        if v1.x == 0:
            return v1.y * v2.y < 0
        else:
            return v1.x * v2.x < 0
    else:
        return False


def get_points_dist(p1, p2):
    [x1, y1] = p1
    [x2, y2] = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def draw_pools(pools_list):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for pool in pools_list:
        pool_rect = pool["rect"]
        header = (pool_rect[0], pool_rect[1], cfg.DEFAULT_POOL_HEADER_WIDTH, pool_rect[3])
        pool_lanes = pool["lanes"]
        drawing = helper.draw_one_rect(drawing, pool_rect, cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
        # cv.namedWindow("pool", 0)
        # cv.imshow("pool", drawing)
        # cv.waitKey(0)
        drawing = helper.draw_one_rect(drawing, header, cfg.COLOR_GREEN, cfg.CONTOUR_THICKNESS)
        # cv.imshow("pool", drawing)
        # cv.waitKey(0)

        sub_procs = pool.get("sub_procs", {})
        for i, lane in enumerate(pool_lanes):
            drawing = helper.draw_one_rect(drawing, lane, cfg.COLOR_BLUE, cfg.CONTOUR_THICKNESS)
            # print(lane)
            procs = sub_procs.get(i, None)
            if procs is not None:
                for proc in procs:
                    drawing = helper.draw_one_rect(drawing, proc, cfg.COLOR_GREEN, cfg.CONTOUR_THICKNESS)
            # cv.imshow("pool", drawing)
            # cv.waitKey(0)

        elements = pool.get("elements")
        if elements is not None:
            keys = list(elements.keys())
            keys.sort()
            for key in keys:
                elements_in_lane = elements[key]
                for element in elements_in_lane:
                    drawing = helper.draw_one_rect(drawing, element, cfg.COLOR_BLUE, cfg.CONTOUR_THICKNESS)

    return drawing


def draw_line_info(base, info, k, b, color, thickness):
    line = info_to_line(info, k, b)
    cv.line(base, line.p1, line.p2, color, thickness, cv.LINE_AA)
    return base


def draw_lines(base, arrow_lines, color, thickness):
    for k in arrow_lines.keys():
        v = arrow_lines[k]
        for key in v.keys():
            base = draw_line_info(base, v[key], key[0], key[1], color, thickness)
    return base


def remove_elements(element_border):
    # remove elements
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    reverse = np.multiply(np.ones_like(input_img, dtype=np.uint8), 255) - input_img
    mask = np.zeros_like(drawing, dtype=np.uint8)

    for pool in pools:
        pool_rect = pool["rect"]
        drawing = truncate(drawing, reverse, pool_rect)
        drawing = helper.draw_one_rect(drawing, pool_rect, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        sub_procs = pool.get("sub_procs", {})

        for i, lane in enumerate(lanes):
            drawing = helper.draw_one_rect(drawing, lane, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
            elements_i = elements[i]
            for element in elements_i:
                e_rect = helper.dilate(element, element_border)
                drawing = truncate(drawing, mask, e_rect)
            lane_sub_procs = sub_procs.get(i, None)
            if lane_sub_procs is not None:
                for sub_proc in lane_sub_procs:
                    drawing = helper.draw_one_rect(drawing, sub_proc, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
    return drawing


def truncate(base, target, roi_rect):
    base[max(0, roi_rect[1]):roi_rect[1] + roi_rect[3], max(0, roi_rect[0]):roi_rect[0] + roi_rect[2], :] \
        = target[max(0, roi_rect[1]):roi_rect[1] + roi_rect[3], max(0, roi_rect[0]):roi_rect[0] + roi_rect[2], :]
    return base


def get_arrows(flows_img):
    arrows = []
    erode_element = helper.get_structure_ele(cv.MORPH_RECT, 2)
    erode = cv.erode(flows_img, erode_element)
    _, erode = cv.threshold(erode, 50, 255, cv.THRESH_BINARY)

    gray = cv.cvtColor(erode, cv.COLOR_BGR2GRAY)
    _, arrow_contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i, arrow_contour in enumerate(arrow_contours):
        bound = helper.get_one_contour_rec(i, arrow_contours)
        ratio = bound[0][3] / bound[0][2]
        # print("area:{}, ratio:{}".format(bound[2], "%.3f" % ratio))
        # TODO How to detect an arrow
        if 5 < bound[2] < 200 and 0.39 < ratio < 4.9:
            bound_rec = helper.dilate(bound[0], cfg.BOUNDARY_OFFSET)
            arrows.append(bound_rec)
    # cv.imshow("remove_elements", flows_img)
    # cv.imshow("arrow_erode", erode)
    return arrows


def remove_text(flows_only):
    # remove text
    mask = np.zeros_like(input_img, dtype=np.uint8)
    element1 = helper.get_structure_ele(cv.MORPH_RECT, 2)
    morph = cv.morphologyEx(flows_only, cv.MORPH_BLACKHAT, element1)
    element2 = helper.get_structure_ele(cv.MORPH_RECT, 3)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, element2)

    morph.dtype = np.uint8
    gray = cv.cvtColor(morph, cv.COLOR_BGR2GRAY)
    _, del_contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(del_contours)):
        bound = helper.get_one_contour_rec(i, del_contours)
        bound_rec = helper.dilate(bound[0], 2)
        if bound[1] > 2:
            flows_only = truncate(flows_only, mask, bound_rec)
    _, flows_only = cv.threshold(flows_only, 200, 255, cv.THRESH_BINARY)
    return flows_only


def detect_lines(flows_copy):
    flows_only = cv.cvtColor(flows_copy, cv.COLOR_BGR2GRAY)
    detected_lines = cv.HoughLinesP(flows_only, 1, np.pi / 180, cfg.SEQ_FLOW_THRESHOLD, None, cfg.SEQ_MIN_LENGTH,
                                    cfg.SEQ_MAX_GAP)
    line_list = []
    for detect_line in detected_lines:
        one_line = detect_line[0]
        line_list.append(create_line(one_line))
    return line_list


def create_line(one_line):
    if abs(one_line[0] - one_line[2]) < cfg.RECT_DILATION_VALUE:
        # 竖直线
        mean_x = (one_line[0] + one_line[2]) // 2
        one_line = (int(mean_x), int(one_line[1]), int(mean_x), int(one_line[3]))
    if abs(one_line[1] - one_line[3]) < cfg.RECT_DILATION_VALUE:
        # 水平线
        mean_y = (one_line[1] + one_line[3]) // 2
        one_line = (int(one_line[0]), int(mean_y), int(one_line[2]), int(mean_y))
    return Line(one_line)


def similar_lines_merge(similar):
    k = similar[0].k
    b = similar[0].b
    for line in similar:
        if line.k is None:
            k = None
            break
        elif line.k == 0:
            k = 0
            break
    merged_similar = []
    while len(similar) > 0:
        one_line = similar[0]
        overlap_list = [0]
        for i in range(1, len(similar)):
            line = similar[i]
            if parallel_lines_are_overlap(one_line, line, k):
                overlap_list.append(i)
        if len(overlap_list) == 1:
            merged_similar.append(one_line)
            similar.pop(0)
        else:
            temp = []
            if k is None:
                for index in overlap_list:
                    overlap_line = similar[index]
                    temp.append(overlap_line.p1[1])
                    temp.append(overlap_line.p2[1])
                    similar[index] = None
                begin = min(temp)
                end = max(temp)
                one_line = Line([b, begin, b, end])
            else:
                for index in overlap_list:
                    overlap_line = similar[index]
                    temp.append(overlap_line.p1[0])
                    temp.append(overlap_line.p2[0])
                    similar[index] = None
                begin = min(temp)
                end = max(temp)
                li = [begin, int(k * begin + b), end, int(k * end + b)]
                one_line = Line(li)
            merged_similar.append(one_line)
            similar = list(filter(lambda l: l is not None, similar))
    return merged_similar


def parallel_lines_are_overlap(line1, line2, k):
    length_sum = line1.length + line2.length
    points = [line1.p1, line1.p2, line2.p1, line2.p2]
    if k is None:
        points.sort(key=lambda p: p[1])
        direct_length = get_points_dist(points[0], points[-1])
    else:
        points.sort(key=lambda p: p[0])
        direct_length = get_points_dist(points[0], points[-1])
    if direct_length < length_sum + 30:
        return True
    else:
        return False


def normalize_all_lines(lines):
    normalized_lines = []
    while len(lines) > 0:
        one_line = lines.pop(0)
        k = one_line.k
        b = one_line.b
        borders = get_parallel_lines_with_d(k, b, 3)
        similar = [one_line]
        for i, line in enumerate(lines):
            if is_between(line, borders[0], borders[1]):
                similar.append(line)
                lines[i] = None
        lines = list(filter(lambda l: l is not None, lines))
        if len(similar) > 1:
            merged_similar = similar_lines_merge(similar)
            normalized_lines.extend(merged_similar)
        else:
            normalized_lines.extend(similar)
    return normalized_lines


def is_between(line, line_border1, line_border2):
    if point_is_between(line.p1, line_border1, line_border2) and point_is_between(line.p2, line_border1, line_border2):
        return True
    return False


def point_is_between(point, line_border1, line_border2):
    [k, b1] = line_border1
    b2 = line_border2[1]
    if k is None:
        a = min(b1, b2)
        b = max(b1, b2)
        if a < point[0] < b:
            return True
    else:
        d1_1 = k * point[0] - point[1] + b1
        d1_2 = k * point[0] - point[1] + b2
        if d1_1 * d1_2 <= 0:
            return True
    return False


def get_parallel_lines_with_d(k, b, d):
    borders = []
    if k is None:
        borders.append((k, b - d))
        borders.append((k, b + d))
    else:
        borders.append((k, b - d * ((1 + k ** 2) ** 0.5)))
        borders.append((k, b + d * ((1 + k ** 2) ** 0.5)))
    return borders


def get_line_info(line, p1_in, p2_in):
    if p1_in or p2_in:  # 判断线段的方向
        info = None
        if line.k is None:  # 线垂直
            if p1_in:  # 下到上
                info = [line.p1[1], line.p2[1], 3]
            elif p2_in:  # 上到下
                info = [line.p1[1], line.p2[1], 1]
        else:  # 线不垂直
            if p1_in:  # 右到左
                info = [line.p1[0], line.p2[0], 2]
            elif p2_in:  # 左到右
                info = [line.p1[0], line.p2[0], 4]
        return info
    return None


def add_info_to_arrow_lines(line, info, arrow_lines, i):
    line_map = arrow_lines.get(i)
    if line_map is None:
        # 一个箭头找到了第一条线段
        arrow_lines[i] = {(line.k, line.b): info}
    else:
        # 已经有线段的箭头找到了另一条线段
        one_arrow_line = line_map.get((line.k, line.b))
        if one_arrow_line is None:
            # 这条线段和已有的线段不在同一直线上
            line_map[(line.k, line.b)] = info
        else:
            # 这条线段和已有的线段在同一直线上
            merge_info = line_with_same_k_b_merge(one_arrow_line, info)
            line_map[(line.k, line.b)] = merge_info
    return arrow_lines


# line2 merge to line1
def line_with_same_k_b_merge(line1_info, line2_info):
    if line1_info[2] == line2_info[2]:
        all_points = [line1_info[0], line1_info[1], line2_info[0], line2_info[1]]
        begin = min(all_points)
        end = max(all_points)
        return [begin, end, line1_info[2]]
    else:
        return line1_info


def get_rect_vertices(rec):
    # the four vertices of a rectangle
    points = list()
    points.append((rec[0], rec[1]))
    points.append((rec[0] + rec[2], rec[1]))
    points.append((rec[0] + rec[2], rec[1] + rec[3]))
    points.append((rec[0], rec[1] + rec[3]))
    return points


def line_is_intersect_rec(line, rec):
    if helper.is_in(rec, (line[0], line[1], 0, 0)) or helper.is_in(rec, (line[2], line[3], 0, 0)):
        return True

    points = get_rect_vertices(rec)
    vectors = list()
    p1 = (line[0], line[1])
    p2 = (line[2], line[3])
    line_vector = Vector(p1, p2)
    # print(line_vector)
    for point in points:
        vectors.append(Vector(p1, point))

    # [print(x) for x in vectors]
    vectors.sort(key=cmp_to_key(cross_product))
    # [print(x) for x in vectors]

    return cross_product(line_vector, vectors[0]) * cross_product(line_vector, vectors[-1]) <= 0


def get_rec_center(rec):
    rec_center = (rec[0] + rec[2] // 2, rec[1] + rec[3] // 2)
    return rec_center


def get_line_seg_rec_dist(line, rec):
    # 获取线段到箭头的距离，以及哪一端更近
    rec_center = get_rec_center(rec)

    dist_1 = get_points_dist(line.p1, rec_center)
    dist_2 = get_points_dist(line.p2, rec_center)

    if dist_1 < dist_2:
        return dist_1, 0
    else:
        return dist_2, 1


def get_line_rec_dist(line, rec):
    # 获取箭头到直线的距离
    rec_center = get_rec_center(rec)

    if line.k is None:
        return abs(rec_center[0] - line.b)
    else:
        dist = abs(line.k * rec_center[0] - rec_center[1] + line.b) / (1 + line.k ** 2) ** 0.5
        return dist


def info_to_line(info, k, b):
    if k is None:
        return Line((b, info[0], b, info[1]))
    else:
        line = [info[0], int(k * info[0] + b), info[1], int(k * info[1] + b)]
        return Line(line)


def extend_line(line, rect):
    center = get_rec_center(rect)
    if line.k is None:
        temp_list = [line.li[1], line.li[3], center[1]]
        begin = min(temp_list)
        end = max(temp_list)
        return Line([line.b, begin, line.b, end])
    else:
        temp_list = [line.li[0], line.li[2], center[0]]
        begin = min(temp_list)
        end = max(temp_list)
        return Line([begin, int(line.k * begin + line.b), end, int(line.k * end + line.b)])


def get_initial_lines(arrows, line_list):
    # 区分与arrow 直连和分离的线
    # arrow_lines: {(k, b): info} info: [start_point, end_point, direction]
    arrow_lines = dict()
    discrete_lines = []
    for line in line_list:
        to_next = False
        intersected_arrows = list()
        for i, arrow in enumerate(arrows):
            p1_in = helper.is_in(arrow, (line.li[0], line.li[1], 0, 0))
            p2_in = helper.is_in(arrow, (line.li[2], line.li[3], 0, 0))
            info = get_line_info(line, p1_in, p2_in)
            if info is not None:  # 线段有一端在箭头中
                # cv.line(flows_cp, line.p1, line.p2, COLOR_GREEN, CONTOUR_THICKNESS)
                arrow_lines = add_info_to_arrow_lines(line, info, arrow_lines, i)
                to_next = True
                break
            else:  # 线段不与箭头直接连接

                if line_is_intersect_rec(line.li, arrow):
                    intersected_arrows.append(i)
        if not to_next:

            if len(intersected_arrows) > 0:  # 线段不在箭头中，但所在直线经过箭头
                # cv.line(flows_cp, line.p1, line.p2, COLOR_BLUE, CONTOUR_THICKNESS)
                min_dist = float("inf")
                min_index = -1
                closest_point = -1
                # 找到距离最近的 arrow
                for index in intersected_arrows:
                    temp = get_line_seg_rec_dist(line, arrows[index])
                    if temp[0] < min_dist:
                        min_dist = temp[0]
                        min_index = index
                        closest_point = temp[1]
                info = get_line_info(line, closest_point == 0, closest_point == 1)
                if min_dist < 45:  # 距离小于45 说明直连
                    arrow_lines = add_info_to_arrow_lines(line, info, arrow_lines, min_index)
                else:
                    # cv.line(flows_cp, line.p1, line.p2, COLOR_RED, CONTOUR_THICKNESS)
                    line_map = arrow_lines.get(min_index)
                    if line_map is not None:
                        one_arrow_line = line_map.get((line.k, line.b))
                        if one_arrow_line is not None and one_arrow_line[2] == info[2] and min_dist < 100:
                            merge_info = line_with_same_k_b_merge(one_arrow_line, info)
                            line_map[(line.k, line.b)] = merge_info
                        else:
                            discrete_lines.append(line)
                    else:
                        discrete_lines.append(line)
            else:
                # cv.line(flows_cp, line.p1, line.p2, COLOR_BLUE, CONTOUR_THICKNESS)
                discrete_lines.append(line)

    # 若是独立箭头，连接箭头与部分分离线段，箭头位于线段一端的垂线上，否则合并相似连线
    for i in range(len(arrows)):
        arrow_line = arrow_lines.get(i)
        if arrow_line is None:
            for discrete_line in discrete_lines:
                seg_dist = get_line_seg_rec_dist(discrete_line, arrows[i])
                line_dist = get_line_rec_dist(discrete_line, arrows[i])
                if seg_dist[0] < 60 and 0 <= seg_dist[0] - line_dist < 5:
                    begin = (discrete_line.li[seg_dist[1] * 2], discrete_line.li[seg_dist[1] * 2 + 1])
                    end = get_rec_center(arrows[i])
                    new_line = create_line((begin[0], begin[1], end[0], end[1]))
                    p1_in = helper.is_in(arrows[i], (new_line.li[0], new_line.li[1], 0, 0))
                    p2_in = helper.is_in(arrows[i], (new_line.li[2], new_line.li[3], 0, 0))
                    info = get_line_info(new_line, p1_in, p2_in)
                    arrow_lines[i] = {(new_line.k, new_line.b): info}
        else:
            one_arrow_line = []
            for k, v in arrow_line.items():
                one_arrow_line.append(info_to_line(v, k[0], k[1]))

            normalized_arrow_line = normalize_all_lines(one_arrow_line)
            arrow_line = {}
            arrow = arrows[i]
            for normalized_line in normalized_arrow_line:
                p1_in = helper.is_in(arrow, (normalized_line.li[0], normalized_line.li[1], 0, 0))
                p2_in = helper.is_in(arrow, (normalized_line.li[2], normalized_line.li[3], 0, 0))
                info = get_line_info(normalized_line, p1_in, p2_in)
                if info is None:
                    normalized_line = extend_line(normalized_line, arrow)
                    temp = get_line_seg_rec_dist(normalized_line, arrow)
                    info = get_line_info(normalized_line, temp[1] == 0, temp[1] == 1)
                arrow_line[(normalized_line.k, normalized_line.b)] = info
            arrow_lines[i] = arrow_line
    return arrow_lines, discrete_lines


def get_arrow_ele_direct(arrow, ele):
    arrow_center = get_rec_center(arrow)
    ele_shrink = helper.shrink(ele, 10)
    # ele_shrink = helper.shrink(ele, cfg.)
    p1 = (ele_shrink[0], ele_shrink[1])
    p2 = (ele_shrink[0] + ele_shrink[2], ele_shrink[1])

    v1 = Vector(arrow_center, p1)
    cos1 = v1.get_cos_x()
    v2 = Vector(arrow_center, p2)
    cos2 = v2.get_cos_x()

    if cos1 * cos2 > 0:  # 水平
        if arrow[0] < ele_shrink[0]:  # 在左边
            return 4
        else:  # 在右边
            return 2
    else:  # 竖直
        if arrow[1] < ele_shrink[1]:  # 在上边
            return 1
        else:  # 在下边
            return 3


def get_opposite_direct_list(direct):
    if direct == 1:
        return [3]
    elif direct == 3:
        return [1]
    elif direct == 2:
        return [1, 3, 4]
    elif direct == 4:
        return [1, 2, 3]


def filter_arrow_lines(arrow_id, direct, arrow_lines):
    arrow_line = arrow_lines.get(arrow_id)
    if arrow_line is not None:
        del_direct = get_opposite_direct_list(direct)
        remove_list = []
        for k in arrow_line.keys():
            line_info = arrow_line[k]
            if line_info[2] in del_direct:
                remove_list.append(k)
        for rm_k in remove_list:
            arrow_line.pop(rm_k)
        if len(arrow_line) == 0:
            arrow_lines.pop(arrow_id)
        else:
            arrow_lines[arrow_id] = arrow_line
    return arrow_lines


def create_virtual_element(arrow, direct):
    default_width = 70
    default_height = 70

    if direct == 1:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] + arrow[3]
    elif direct == 3:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] - default_height
    elif direct == 2:
        p_x = arrow[0] - default_width
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2
    elif direct == 4:
        p_x = arrow[0] + arrow[2]
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2
    else:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2

    default_element = [int(p_x), int(p_y), default_width, default_height]
    return default_element


def get_possible_element(arrow_id_list, arrows):
    x_list = []
    y_list = []
    for arrow_id in arrow_id_list:
        arrow = arrows[arrow_id]
        arrow_center = get_rec_center(arrow)
        x_list.append(arrow_center[0])
        y_list.append(arrow_center[1])

    p_x = int(np.mean(x_list))
    p_y = int(np.mean(y_list))
    height = max(y_list) - min(y_list)
    width = max(x_list) - min(x_list)
    length = max(height, width, 40)

    ele_rect = (p_x - length // 2, p_y - length // 2, length, length)
    return ele_rect


def add_one_element_to_pool(ele_rect):
    global all_elements
    for pool_id, pool in enumerate(pools):
        pool_lanes_rect = pool["lanes_rect"]
        if helper.is_overlap(pool_lanes_rect, ele_rect):
            elements = pool.get("elements", defaultdict(list))
            lanes = pool["lanes"]
            for lane_id, lane in enumerate(lanes):
                if helper.is_overlap(lane, ele_rect):
                    elements_in_lane = elements.get(lane_id, [])
                    for ele_id, element in enumerate(elements_in_lane):
                        if helper.is_overlap(element, ele_rect):
                            ele_path = (pool_id, lane_id, ele_id, 0)
                            return ele_path
                    ele_id = len(elements_in_lane)
                    elements_in_lane.append(ele_rect)
                    elements[lane_id] = elements_in_lane
                    ele_path = (pool_id, lane_id, ele_id, 0)
                    all_elements.append(ele_path)
                    return ele_path
            break
    return None


def get_element_rec_by_id(ele_id):
    path = all_elements[ele_id]
    return get_element_rec_by_path(path)


def get_element_rec_by_path(path):
    pool = pools[path[0]]
    # print(path)
    if path[3] == 0:
        elements = pool.get("elements", defaultdict(list))
        elements_i = elements.get(path[1], [])
        rect = elements_i[path[2]]
    else:
        sub_procs = pool["sub_procs"]
        sub_procs_in_lane = sub_procs.get(path[1])
        rect = sub_procs_in_lane[path[2]]
    return rect


def modify_element_rec_by_id(ele_id, ele_rec):
    path = all_elements[ele_id]
    modify_element_rec_by_path(path, ele_rec)


def modify_element_rec_by_path(path, ele_rec):
    global pools
    pool = pools[path[0]]
    elements = pool["elements"]
    element_i = elements[path[1]]
    element_i[path[2]] = ele_rec


def get_element_id(ele_path):
    for i, path in enumerate(all_elements):
        if path[3] == ele_path[3] and ele_path[0] == path[0] and ele_path[1] == path[1] and ele_path[2] == path[2]:
            return i
    print("element not found!")
    return -1


def get_all_elements():
    global all_elements
    # 统计所有的元素
    # [[pool_id, lane_id, element_id, is_sub_process]]
    all_elements = []
    for i, pool in enumerate(pools):
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        sub_procs = pool.get("sub_procs", {})
        for j in range(len(lanes)):
            elements_in_lane = elements.get(j)
            if elements_in_lane is not None:
                for k in range(len(elements_in_lane)):
                    all_elements.append((i, j, k, 0))
            sub_procs_in_lane = sub_procs.get(j, None)
            if sub_procs_in_lane is not None:
                for k in range(len(sub_procs_in_lane)):
                    all_elements.append((i, j, k, 1))


def match_arrows_and_elements(arrow_lines, arrows):
    # 匹配 arrow 与 element
    # 依据两者位置关系，删除一些线
    # 依据 arrow 确定一些未检出的元素
    # arrwo_ele_map :{ arrow_id : [ele_path, arrow_direction]}
    arrow_ele_map = dict()
    possible_ele = dict()
    for arrow_id, arrow in enumerate(arrows):
        found = False
        for pool_id, pool in enumerate(pools):
            pool_lanes_rect = pool["lanes_rect"]
            if helper.is_in(pool_lanes_rect, arrow):
                elements = pool.get("elements", defaultdict(list))
                lanes = pool["lanes"]
                sub_procs = pool.get("sub_procs", {})
                for lane_id, lane in enumerate(lanes):
                    if helper.is_in(lane, arrow):
                        sub_procs_in_lane = sub_procs.get(lane_id, None)
                        elements_in_lane = elements.get(lane_id)
                        dilate_arrow = helper.dilate(arrow, cfg.RECT_DILATION_VALUE)

                        if elements_in_lane is not None:
                            for ele_id, ele in enumerate(elements_in_lane):
                                if helper.is_overlap(dilate_arrow, ele):
                                    found = True
                                    arrow_direct = get_arrow_ele_direct(arrow, ele)
                                    arrow_line = arrow_lines.get(arrow_id)
                                    if arrow_line is not None and len(arrow_line) > 1:
                                        arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                                    arrow_ele_map[arrow_id] = [(pool_id, lane_id, ele_id, 0), arrow_direct]
                                    break

                        if not found and sub_procs_in_lane is not None:
                            for sub_proc_id, sub_proc in enumerate(sub_procs_in_lane):
                                if not helper.is_in(sub_proc, dilate_arrow) and helper.is_overlap(dilate_arrow,
                                                                                                  sub_proc):
                                    found = True
                                    arrow_direct = get_arrow_ele_direct(arrow, sub_proc)
                                    arrow_line = arrow_lines.get(arrow_id)
                                    if arrow_line is not None and len(arrow_line) > 1:
                                        arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                                    arrow_ele_map[arrow_id] = [(pool_id, lane_id, sub_proc_id, 1), arrow_direct]
                                    break

                        if not found:
                            arrow_line = arrow_lines.get(arrow_id)
                            if arrow_line is not None:
                                values = list(arrow_line.values())
                                direct_count = [[1, 0], [2, 0], [3, 0], [4, 0]]
                                for value in values:
                                    direct = value[2]
                                    direct_count[direct - 1][1] += 1
                                direct_count.sort(key=lambda x: x[1], reverse=True)
                                ele_rect = create_virtual_element(arrow, direct_count[0][0])
                            else:
                                ele_rect = create_virtual_element(arrow, 0)
                            possible_ele[arrow_id] = ele_rect
                        break
                break

    if len(possible_ele) > 0:
        keys = list(possible_ele.keys())
        while len(keys) > 0:
            one_arrow_id = keys.pop(0)
            ele_rec = possible_ele[one_arrow_id]
            adjacent_arrows = [one_arrow_id]
            for i, key in enumerate(keys):
                if helper.is_overlap(ele_rec, possible_ele[key]):
                    adjacent_arrows.append(key)
                    keys[i] = None
            keys = list(filter(lambda x: x is not None, keys))
            if len(adjacent_arrows) > 1:
                ele_rec = get_possible_element(adjacent_arrows, arrows)
            ele_path = add_one_element_to_pool(ele_rec)
            if ele_path is not None:
                ele = get_element_rec_by_path(ele_path)
                for arrow_id in adjacent_arrows:
                    arrow = arrows[arrow_id]
                    arrow_direct = get_arrow_ele_direct(arrow, ele)
                    arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                    arrow_ele_map[arrow_id] = [ele_path, arrow_direct]

    return arrow_lines, arrow_ele_map


def is_parallel(line1, line2):
    if line1.k is None and line2.k is None:
        return True
    elif line1.k is not None and line2.k is not None:
        return abs(line1.k - line2.k) < 0.000001
    else:
        return False


def line_is_in_rec(line, rec):
    p1_in = helper.is_in(rec, (line.li[0], line.li[1], 0, 0))
    p2_in = helper.is_in(rec, (line.li[2], line.li[3], 0, 0))
    if p1_in and p2_in:
        return True, -1
    elif p1_in:
        return True, 0
    elif p2_in:
        return True, 1
    else:
        return False, -1


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


def is_begin_point(point, line, end_ele_id):
    point_rec = helper.dilate((point[0], point[1], 0, 0), 10)
    overlapped_ele = {}
    for i in range(len(all_elements)):
        ele_rec = get_element_rec_by_id(i)
        if all_elements[i][3] == 0:
            if helper.is_overlap(ele_rec, point_rec) and i != end_ele_id:
                overlapped_ele[i] = ele_rec
        else:
            if not helper.is_in(ele_rec, point_rec) and helper.is_overlap(ele_rec, point_rec) and i != end_ele_id:
                return True, i

    if len(overlapped_ele) == 0:
        return False, -1
    else:
        for ele_id, ele_rec in overlapped_ele.items():
            if line_is_intersect_rec(line.li, ele_rec):
                return True, ele_id
        return False, -1


def point_is_near_line(point, line, line_range):
    borders = get_parallel_lines_with_d(line.k, line.b, 10)
    if point_is_between(point, borders[0], borders[1]):
        if line_range[0][0] <= point[line_range[1]] <= line_range[0][1]:
            return True
        else:
            return False
    else:
        return False


def add_one_flow(flows, flow, arrow_id):
    arrow_flows = flows[arrow_id]
    existed = False
    for flow_id, one_flow in enumerate(arrow_flows):
        if one_flow[0] == flow[0] and one_flow[2] == flow[2]:
            existed = True
            if len(flow[1]) < len(one_flow[1]):
                arrow_flows[flow_id] = flow
            break
    if not existed:
        arrow_flows.append(flow)


def detect_one_flow(flow_points, discrete_lines, end_ele_id, flows, arrow_id, merged_lines):
    curr_begin = flow_points[-1]
    last_begin = flow_points[-2]
    curr_line = Line([last_begin[0], last_begin[1], curr_begin[0], curr_begin[1]])

    line_range = None
    if curr_begin[0] < last_begin[0]:
        line_range = [[curr_begin[0] - cfg.LINE_AREA_THRESHOLD, last_begin[0]], 0]
    elif curr_begin[0] > last_begin[0]:
        line_range = [[last_begin[0], curr_begin[0] + cfg.LINE_AREA_THRESHOLD], 0]
    else:
        if curr_begin[1] < last_begin[1]:
            line_range = [[curr_begin[1] - cfg.LINE_AREA_THRESHOLD, last_begin[1]], 1]
        elif curr_begin[1] > last_begin[1]:
            line_range = [[last_begin[1], curr_begin[1] + cfg.LINE_AREA_THRESHOLD], 1]

    is_begin = is_begin_point(curr_begin, curr_line, end_ele_id)
    if is_begin[0]:
        temp = flow_points.copy()
        flow = [end_ele_id, temp, is_begin[1]]
        add_one_flow(flows, flow, arrow_id)

    to_extend_lines = []
    intersected_lines = []
    next_discrete_lines = []
    for i, discrete_line in enumerate(discrete_lines):
        p1_near = point_is_near_line(discrete_line.p1, curr_line, line_range)
        p2_near = point_is_near_line(discrete_line.p2, curr_line, line_range)
        if (p1_near and not p2_near) or (p2_near and not p1_near):
            if is_parallel(discrete_line, curr_line):
                to_extend_lines.append(discrete_line)
            else:
                intersected_lines.append(discrete_line)
        else:
            next_discrete_lines.append(discrete_line)
    discrete_lines = next_discrete_lines
    if len(to_extend_lines) == 0 and len(intersected_lines) == 0:
        if not is_begin[0]:
            flow = [end_ele_id, flow_points, None]
            flows[arrow_id].append(flow)
    else:
        merged_lines.extend(to_extend_lines)
        merged_lines.extend(intersected_lines)
        if len(to_extend_lines) > 0:
            temp = []
            if curr_line.k is None:
                for to_extend_line in to_extend_lines:
                    temp.append(to_extend_line.p1[1])
                    temp.append(to_extend_line.p2[1])
                if curr_begin[1] < last_begin[1]:
                    new_begin = (curr_line.b, min(temp))
                else:
                    new_begin = (curr_line.b, max(temp))
                flow_points[-1] = new_begin
            else:
                for to_extend_line in to_extend_lines:
                    temp.append(to_extend_line.p1[0])
                    temp.append(to_extend_line.p2[0])
                if curr_begin[0] < last_begin[0]:
                    new_x = min(temp)
                    new_begin = (new_x, int(curr_line.k * new_x + curr_line.b))
                else:
                    new_x = max(temp)
                    new_begin = (new_x, int(curr_line.k * new_x + curr_line.b))
                flow_points[-1] = new_begin
            detect_one_flow(flow_points, discrete_lines, end_ele_id, flows, arrow_id, merged_lines)

        if len(intersected_lines) > 0:
            flow_points_list = [flow_points.copy() for l in range(len(intersected_lines))]
            for i, intersected_line in enumerate(intersected_lines):
                intersection = get_point_of_intersection(curr_line, intersected_line)
                d1 = get_points_dist(intersected_line.p1, intersection)
                d2 = get_points_dist(intersected_line.p2, intersection)
                flow_points_list[i][-1] = intersection
                if d1 > d2:
                    flow_points_list[i].append(intersected_line.p1)
                else:
                    flow_points_list[i].append(intersected_line.p2)
                detect_one_flow(flow_points_list[i], discrete_lines, end_ele_id, flows, arrow_id, merged_lines)


def connect_elements(arrows, arrow_lines, arrow_ele_map, discrete_lines):
    # n = len(all_elements)
    # graph = [[-1 for j in range(n)] for i in range(n)]
    # flows: {arrow_id: [[end_ele_id, flow_points, start_ele_id]]}
    flows = defaultdict(list)
    merged_lines = []
    for arrow_id in range(len(arrows)):
        ele_map = arrow_ele_map.get(arrow_id)
        arrow_line = arrow_lines.get(arrow_id)
        end_ele_id = get_element_id(ele_map[0])
        if arrow_line is not None:
            for key, info in arrow_line.items():
                line = info_to_line(info, key[0], key[1])
                flow_points = list()

                if info[2] == 1 or info[2] == 4:
                    flow_points.extend([line.p2, line.p1])
                else:
                    flow_points.extend([line.p1, line.p2])
                detect_one_flow(flow_points, discrete_lines, end_ele_id, flows, arrow_id, merged_lines)
    for merged_line in merged_lines:
        try:
            discrete_lines.remove(merged_line)
        except ValueError:
            # print("remove error")
            continue

    return flows, discrete_lines


def points_to_line(p1, p2):
    return Line([p1[0], p1[1], p2[0], p2[1]])


def get_p2_2_p1_direct(p1, p2):
    if p1[0] < p2[0]:  # 右到左
        direct = 2
    elif p1[0] > p2[0]:  # 左到右
        direct = 4
    else:
        if p1[1] < p2[1]:  # 下到上
            direct = 3
        elif p1[1] > p2[1]:  # 上到下
            direct = 1
        else:
            direct = 0
    return direct


def get_end_point_rec_dis(p1, p2, rec, ele_type):
    direct = get_p2_2_p1_direct(p1, p2)
    if ele_type == 0:
        rec_center = get_rec_center(rec)
        valid = False

        if direct == 1:
            if rec_center[1] >= (p1[1] - 30):
                valid = True
        elif direct == 3:
            if rec_center[1] <= (p1[1] + 30):
                valid = True
        elif direct == 2:
            if rec_center[0] <= (p1[0] + 30):
                valid = True
        elif direct == 4:
            if rec_center[0] >= (p1[0] - 30):
                valid = True
        if valid:
            return get_points_dist(rec_center, p1)
        else:
            return -1
    else:
        return -1


def get_begin_ele_id(p1, p2):
    begin_ele_id = -1
    min_dis = float("inf")
    for i in range(len(all_elements)):
        ele_type = all_elements[i][3]
        ele_rec = get_element_rec_by_id(i)
        dist = get_end_point_rec_dis(p1, p2, ele_rec, ele_type)

        if 0 < dist < min_dis:
            min_dis = dist
            begin_ele_id = i
    return begin_ele_id


def complete_flow(begin_ele_id, flow_points):
    p1 = flow_points[-1]
    p2 = flow_points[-2]

    end_line = points_to_line(p1, p2)
    ele_rec = get_element_rec_by_id(begin_ele_id)
    ele_center = get_rec_center(ele_rec)
    if line_is_intersect_rec(end_line.li, ele_rec):
        if end_line.k is None:
            end_point = (end_line.b, ele_center[1])
        else:
            end_point = (ele_center[0], int(end_line.k * ele_center[0] + end_line.b))
        flow_points[-1] = end_point
    else:
        point_rec = helper.dilate((p1[0], p1[1], 0, 0), 5)
        direct = get_arrow_ele_direct(point_rec, ele_rec)
        if direct == 1 or direct == 3:
            end_point = (p1[0], ele_center[1])
            flow_points.append(end_point)
        else:
            if end_line.k is not None:
                artificial_line = Line([ele_center[0], ele_center[1], ele_center[0], ele_center[1] - 5])
            else:
                artificial_line = Line([ele_center[0], ele_center[1], ele_center[0] - 5, ele_center[1]])

            intersection = get_point_of_intersection(artificial_line, end_line)
            connect_line = points_to_line(intersection, p1)
            if is_parallel(connect_line, end_line):
                flow_points[-1] = intersection
            else:
                flow_points.append(intersection)
            flow_points.append(ele_center)
    return flow_points


def get_ele_edge_point(begin_ele_id, flow_points):
    p1 = flow_points[-1]
    p2 = flow_points[-2]
    ele_path = all_elements[begin_ele_id]
    rec = get_element_rec_by_id(begin_ele_id)
    rec_type = ele_path[3]
    line = points_to_line(p1, p2)
    vertices = get_rect_vertices(rec)

    if rec_type == 1 and not line_is_intersect_rec(line.li, rec):
        flow_points = complete_flow(begin_ele_id, flow_points)
        p1 = flow_points[-1]
        p2 = flow_points[-2]
        line = points_to_line(p1, p2)

    intersections = []
    for i, vertice in enumerate(vertices):
        one_border = points_to_line(vertice, vertices[(i + 1) % 4])
        intersection = get_point_of_intersection(line, one_border)
        if intersection is not None:
            v1 = Vector(intersection, p1)
            v2 = Vector(intersection, p2)
            if is_opposite(v1, v2):
                flow_points[-1] = intersection
                return flow_points
            else:
                intersections.append(intersection)
    if len(intersections) > 0:
        intersections.sort(key=lambda x: get_points_dist(p1, x))
        flow_points[-1] = intersections[0]
    return flow_points


def is_same(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]


def parse_img(file_path):
    global pools
    global input_img

    input_img, layers, contours, contours_rec, partial_elements = pre_process(file_path)
    pools, type_tag = pools_detector.get_pools(layers, contours_rec)
    show_im(input_img, "input")
    # pools_img = draw_pools(pools)
    # show_im(pools_img, "pools_img_no_elements")
    # pools = pools_detector.get_elements(input_img, layers, contours_rec, partial_elements, pools, type_tag)
    pools = pools_detector.get_elements(input_img, layers, contours_rec, partial_elements, pools, type_tag)

    pools_img = draw_pools(pools)
    # show_im(pools_img, "raw_elements")

    flows_img = remove_elements(2)
    arrows = get_arrows(flows_img)

    get_all_elements()
    flows = defaultdict(list)
    if len(arrows) > 0:
        flows_img = remove_elements(4)
        flows_only = remove_text(flows_img)

        # flows_cp = np.copy(flows_only)
        line_list = detect_lines(flows_only)

        line_list = normalize_all_lines(line_list)
        # 获取与箭头相连的直线
        arrow_lines, discrete_lines = get_initial_lines(arrows, line_list)

        # 将箭头与元素匹配，并获取一些之前未检测出的元素
        arrow_lines, arrow_ele_map = match_arrows_and_elements(arrow_lines, arrows)
        discrete_lines = normalize_all_lines(discrete_lines)

        # 去除一端在元素里一端在元素外的离散线段
        for i, line in enumerate(discrete_lines):
            for ele_path in all_elements:
                if ele_path[3] == 0:
                    ele_rec = get_element_rec_by_path(ele_path)
                    if helper.point_is_in(ele_rec, line.p1) and helper.point_is_in(ele_rec, line.p2):
                        discrete_lines[i] = None
        discrete_lines = list(filter(lambda x: x is not None, discrete_lines))

        for line in discrete_lines:
            cv.line(flows_only, line.p1, line.p2, cfg.COLOR_RED, cfg.CONTOUR_THICKNESS, cv.LINE_AA)
        draw_lines(flows_only, arrow_lines, cfg.COLOR_GREEN, cfg.CONTOUR_THICKNESS)
        for arrow in arrows:
            flows_only = helper.draw_one_rect(flows_only, arrow, cfg.COLOR_GREEN, cfg.CONTOUR_THICKNESS)

        # 依据顺序流的末端，递归回溯，找到起点
        flows, discrete_lines = connect_elements(arrows, arrow_lines, arrow_ele_map, discrete_lines)

        for i, line in enumerate(discrete_lines):
            p1_is_begin = is_begin_point(line.p1, line, -1)
            p2_is_begin = is_begin_point(line.p2, line, -1)

            if p1_is_begin[0] and p2_is_begin[0]:
                if p1_is_begin[1] == p2_is_begin[1]:
                    discrete_lines[i] = None
                discrete_lines[i] = None
                continue
            elif p1_is_begin[0]:
                begin_points = [line.p2, line.p1]
                begin_ele_id = p1_is_begin[1]
            elif p2_is_begin[0]:
                begin_points = [line.p1, line.p2]
                begin_ele_id = p2_is_begin[1]
            else:
                discrete_lines[i] = None
                continue

            for arrow_id in range(len(arrows)):
                arrow_flows = flows[arrow_id]
                for flow in arrow_flows:
                    if flow[2] is None:
                        flow_points = flow[1]
                        p1 = flow_points[-1]
                        p2 = flow_points[-2]
                        end_line = Line([p1[0], p1[1], p2[0], p2[1]])

                        intersection = get_point_of_intersection(line, end_line)
                        if intersection is not None:
                            d1 = get_points_dist(intersection, begin_points[0])
                            d2 = get_points_dist(intersection, begin_points[1])
                            if d2 <= d1:
                                continue
                            else:
                                d3 = get_points_dist(intersection, p1)
                                if (d3 <= 5 and d1 < 100) or (d1 <= 5 and d3 < 100):
                                    flow_points[-1] = intersection
                                    flow_points.append(begin_points[-1])
                                    flow[2] = begin_ele_id
                                    discrete_lines[i] = None
        # discrete_lines = list(filter(lambda x: x is not None, discrete_lines))

        for arrow_id_i in range(len(arrows)):
            arrow_flows_i = flows[arrow_id_i]
            for flow_id_i, flow_i in enumerate(arrow_flows_i):
                if flow_i[2] is None:
                    to_next = False
                    flow_i_points = flow_i[1]
                    p1 = flow_i_points[-1]
                    p2 = flow_i_points[-2]
                    end_line = points_to_line(p1, p2)
                    for arrow_id_j in range(len(arrows)):
                        arrow_flows_j = flows[arrow_id_j]
                        for flow_id_j, flow_j in enumerate(arrow_flows_j):
                            if arrow_id_i != arrow_id_j or flow_id_i != flow_id_j:
                                flow_j = arrow_flows_j[flow_id_j]
                                flow_j_points = flow_j[1]
                                for i in range(1, len(flow_j_points)):
                                    p3 = flow_j_points[i - 1]
                                    p4 = flow_j_points[i]
                                    flow_seg = points_to_line(p3, p4)
                                    intersection = get_point_of_intersection(end_line, flow_seg)
                                    if intersection is not None:
                                        d1 = get_points_dist(intersection, p1)
                                        d2 = get_points_dist(intersection, p2)
                                        v1 = Vector(intersection, p3)
                                        v2 = Vector(intersection, p4)
                                        if d1 < 5 and is_opposite(v1, v2) and d1 < d2:
                                            to_next = True
                                            flow_i_points[-1] = intersection
                                            extend_flow = flow_j_points[i:]
                                            flow_i_points.extend(extend_flow)
                                            if flow_j[2] is not None:
                                                flow_i[2] = flow_j[2]
                                            break
                            if to_next:
                                break
                        if to_next:
                            break

        for arrow_id in range(len(arrows)):
            arrow_flows = flows[arrow_id]
            for flow_id, flow in enumerate(arrow_flows):
                if flow[2] is None:
                    flow_points = flow[1]
                    begin_ele_id = get_begin_ele_id(flow_points[-1], flow_points[-2])
                    complete_flow_points = complete_flow(begin_ele_id, flow_points)
                    flow[1] = complete_flow_points
                    flow[2] = begin_ele_id

        pools_img = draw_pools(pools)
        show_im(pools_img, "pools_img_no_lines")

        for arrow_id in range(len(arrows)):
            arrow_flows = flows[arrow_id]
            arrow_ele = arrow_ele_map.get(arrow_id)
            if len(arrow_flows) == 0:
                arrow = arrows[arrow_id]
                # dilated_arrow = dilate(arrow, 5)
                ele_path = arrow_ele[0]
                end_ele_id = get_element_id(ele_path)
                ele_rec = get_element_rec_by_path(ele_path)
                rec_center = get_rec_center(ele_rec)
                arrow_center = get_rec_center(arrow)

                if rec_center[0] == arrow_center[0] or rec_center[1] == arrow_center[1]:
                    virtual_flow_points = [arrow_center, rec_center]
                else:
                    if arrow_ele[1] == 1 or arrow_ele[1] == 3:
                        virtual_flow_points = [(arrow_center[0], rec_center[1]), arrow_center]
                    else:
                        virtual_flow_points = [(rec_center[0], arrow_center[1]), arrow_center]

                p1 = virtual_flow_points[-1]
                p2 = virtual_flow_points[-2]
                begin_ele_id = get_begin_ele_id(p1, p2)
                points = complete_flow(begin_ele_id, virtual_flow_points)
                if is_same(points[1], arrow_center):
                    points.pop(0)
                else:
                    points[0] = arrow_center
                flows[arrow_id] = [[end_ele_id, points, begin_ele_id]]
                arrow_flows = flows[arrow_id]
            for flow in arrow_flows:
                flow_points = flow[1]
                final_flow_points = get_ele_edge_point(flow[2], flow_points)
                flow[1] = final_flow_points

        # 画顺序流
        for arrow_id in range(len(arrows)):
            arrow_flows = flows.get(arrow_id)
            arrow_ele = arrow_ele_map.get(arrow_id)
            if (arrow_flows is None or len(arrow_flows) == 0) and (arrow_ele is None or len(arrow_ele) == 0):
                helper.draw_one_rect(pools_img, arrows[arrow_id], cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
            elif arrow_flows is None or len(arrow_flows) == 0:
                helper.draw_one_rect(pools_img, arrows[arrow_id], cfg.COLOR_BLUE, cfg.CONTOUR_THICKNESS)
            elif arrow_ele is None or len(arrow_ele) == 0:
                helper.draw_one_rect(pools_img, arrows[arrow_id], cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
            else:
                color = cfg.COLOR_GREEN
                if arrow_ele[0][3] == 1:
                    print("connect sub_process")
                    color = cfg.COLOR_BLUE
                helper.draw_one_rect(pools_img, arrows[arrow_id], color, cfg.CONTOUR_THICKNESS)
                for flow in arrow_flows:
                    show_points = False
                    if flow[2] is None:
                        show_points = True
                        color = cfg.COLOR_RED
                    else:
                        if all_elements[flow[2]][3] == 1:
                            color = cfg.COLOR_BLUE
                        else:
                            color = cfg.COLOR_GREEN
                    flow_points = flow[1]

                    if len(flow_points) >= 2:
                        if show_points:
                            print(flow_points)
                        for i in range(1, len(flow_points)):
                            # try:
                            p1 = (int(flow_points[i - 1][0]), int(flow_points[i - 1][1]))
                            p2 = (int(flow_points[i][0]), int(flow_points[i][1]))
                            cv.line(pools_img, p1, p2, color, cfg.CONTOUR_THICKNESS)
                            # except TypeError:
                            #     print(flow_points[i - 1], flow_points[i])

    show_im(pools_img, name="pools_img")
    # cv.waitKey(0)

    all_seq_flows = []
    if len(flows) > 0:
        for arrow_id in range(len(arrows)):
            arrow_flows = flows.get(arrow_id)
            # print(arrow_id)
            for flow in arrow_flows:
                all_seq_flows.append(flow)

    all_elements_type = []
    print("Classifying begins...")
    helper.print_time()
    for ele_path in all_elements:
        if ele_path[3] == 1:
            all_elements_type.append(["subProcess_expanded", ""])
        else:
            ele_rec = get_element_rec_by_path(ele_path)
            ele_rec = helper.dilate(ele_rec, 10)
            ele_img = helper.truncate(input_img, ele_rec)

            ele_type = classifier.classify_with_vgg_16(ele_img)
            text = ""
            if ele_type.endswith("ask") or ele_type in cfg.TASK_LIKE_LIST:
                text = translator.translate(ele_img)
            all_elements_type.append([ele_type, text])
    print(all_elements_type)
    helper.print_time()
    print("Classifying finished!")

    return all_elements_type, all_seq_flows


def show_im(img_matrix, name="img"):
    pass
    # cv.namedWindow(name, cv.WINDOW_NORMAL)
    # cv.namedWindow(name)
    # cv.imshow(name, img_matrix)
    # file_name = "samples/imgs/example/"+ name+".png"
    # cv.imwrite(file_name, img_matrix)
    # cv.waitKey(0)


def run():
    # sample_dir = "imgs/admission/"
    sample_dir = "samples/imgs/sample_1/"
    # sample_dir = "samples/imgs/"
    images = os.listdir(sample_dir)
    # 5, -1, -4
    selected = images
    for im in selected:
        file_path = sample_dir + im
        print(im)
        if os.path.isfile(file_path):
            all_elements_type, all_seq_flows = parse_img(file_path)
            definitions = model_exporter.create_model(input_img, pools, all_elements, all_elements_type, all_seq_flows)
            model_exporter.export_xml(definitions, "output/{}.bpmn".format(im[0:-4]))
            # cv.waitKey(0)
        # break


if __name__ == '__main__':
    # print(helper.is_in([1, 2, 3, 4], [1, 2, 3, 4]))
    classifier = Classifier()
    run()
