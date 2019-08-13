import cv2 as cv
import numpy as np
import os
from collections import defaultdict
from functools import cmp_to_key

import helper.detector_helper as helper
from helper.utils import Line, Vector, is_parallel, get_point_of_intersection, points_to_line
from classifier import Classifier
import cfg
from img_preprocess import pre_process, get_seq_arrows, get_seq_arrow_direction
import pools_detector
import model_exporter
import translator

input_img = []

pools = []
all_elements = []


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


def remove_elements(element_border):
    # remove elements
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    reverse = np.multiply(np.ones_like(input_img, dtype=np.uint8), 255) - input_img
    mask = np.zeros_like(drawing, dtype=np.uint8)

    for pool in pools:
        pool_rect = pool["rect"]
        # drawing = truncate(drawing, reverse, pool_rect)
        drawing = reverse
        drawing = helper.draw_one_rect(drawing, pool_rect, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
        # drawing = helper.draw_one_rect(reverse, pool_rect, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)

        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        sub_procs = pool.get("sub_procs", {})

        for i, lane in enumerate(lanes):
            drawing = helper.draw_one_rect(drawing, lane, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
            elements_i = elements[i]
            for element in elements_i:
                e_rect = helper.dilate(element, element_border)
                drawing = helper.mask(drawing, mask, e_rect)
            lane_sub_procs = sub_procs.get(i, None)
            if lane_sub_procs is not None:
                for sub_proc in lane_sub_procs:
                    drawing = helper.draw_one_rect(drawing, sub_proc, cfg.COLOR_BLACK, cfg.BOUNDARY_OFFSET)
    return drawing


# def get_arrows(flows_img):
#     arrows = []
#     erode_element = helper.get_structure_ele(cv.MORPH_RECT, 2)
#     erode = cv.erode(flows_img, erode_element)
#     _, erode = cv.threshold(erode, 50, 255, cv.THRESH_BINARY)
#
#     gray = cv.cvtColor(erode, cv.COLOR_BGR2GRAY)
#     _, arrow_contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     for i, arrow_contour in enumerate(arrow_contours):
#         bound = helper.get_one_contour_rec(i, arrow_contours)
#         ratio = bound[0][3] / bound[0][2]
#         # print("area:{}, ratio:{}".format(bound[2], "%.3f" % ratio))
#         # TODO How to detect an arrow
#         if 5 < bound[2] < 200 and 0.39 < ratio < 4.9:
#             bound_rec = helper.dilate(bound[0], cfg.BOUNDARY_OFFSET)
#             arrows.append(bound_rec)
#     # cv.imshow("remove_elements", flows_img)
#     # cv.imshow("arrow_erode", erode)
#     return arrows


def remove_text(flows_only):
    # remove text
    mask = np.zeros_like(input_img, dtype=np.uint8)
    element1 = helper.get_structure_ele(cv.MORPH_RECT, 2)
    morph = cv.morphologyEx(flows_only, cv.MORPH_BLACKHAT, element1)
    element2 = helper.get_structure_ele(cv.MORPH_RECT, 3)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, element2)

    morph.dtype = np.uint8
    gray = cv.cvtColor(morph, cv.COLOR_BGR2GRAY)
    # cv.imshow("morph", morph)
    _, del_contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    text = np.zeros_like(flows_only)

    for i in range(len(del_contours)):
        bound = helper.get_one_contour_rec(i, del_contours)
        bound_rec = helper.dilate(bound[0], 2)
        if bound_rec[2] > 8 and bound_rec[3] > 8:
            helper.draw_one_rect(text, bound_rec, cfg.COLOR_WHITE, 1)
        if bound[1] > 2:
            flows_only = helper.mask(flows_only, mask, bound_rec)
    _, flows_only = cv.threshold(flows_only, 200, 255, cv.THRESH_BINARY)
    # cv.imshow("text", text)
    # cv.waitKey(0)
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
    """相近线段合并"""

    # 确定斜率k
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
        # 选一条线段
        one_line = similar[0]

        # 选出所有可以和它合并的线段, 平行并有重叠部分的线段
        overlap_list = [0]
        for i in range(1, len(similar)):
            line = similar[i]
            if parallel_lines_are_overlap(one_line, line, k):
                overlap_list.append(i)
        if len(overlap_list) == 1:
            # 没有可以合并的线段
            merged_similar.append(one_line)
            similar.pop(0)
        else:
            # 合并可以合并的线段
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
    """判断平行的线段是否有重合部分，或者共线线段是否可以合并"""
    length_sum = line1.length + line2.length
    points = [line1.p1, line1.p2, line2.p1, line2.p2]
    if k is None:
        points.sort(key=lambda p: p[1])
        direct_length = helper.get_points_dist(points[0], points[-1])
    else:
        points.sort(key=lambda p: p[0])
        direct_length = helper.get_points_dist(points[0], points[-1])
    if direct_length < length_sum + 15:
        return True
    else:
        return False


def normalize_all_lines(lines):
    """将 斜率k相同, 截距b相近 的线段合并成一条线段"""
    normalized_lines = list()
    while len(lines) > 0:
        one_line = lines.pop(0)
        k = one_line.k
        b = one_line.b
        borders = get_parallel_lines_with_d(k, b, 3)

        similar = [one_line]
        others = list()

        for i, line in enumerate(lines):
            if is_between(line, borders[0], borders[1]):
                similar.append(line)
            else:
                others.append(line)
        lines = others
        if len(similar) > 1:
            merged_similar = similar_lines_merge(similar)
            normalized_lines.extend(merged_similar)
        else:
            normalized_lines.extend(similar)
    return normalized_lines


def is_between(line, line_border1, line_border2):
    """判断一条线段在两条平行直线之间"""
    if point_is_between(line.p1, line_border1, line_border2) and point_is_between(line.p2, line_border1, line_border2):
        return True
    return False


def point_is_between(point, line_border1, line_border2):
    """判断一个点在两条平行线之间"""
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
    """获得一组平行线, 两直线间距离为2d"""
    borders = []
    if k is None:
        borders.append((k, b - d))
        borders.append((k, b + d))
    else:
        borders.append((k, b - d * ((1 + k ** 2) ** 0.5)))
        borders.append((k, b + d * ((1 + k ** 2) ** 0.5)))
    return borders


def get_line_info(line, p1_in, p2_in):
    """判断与箭头相连的线段与箭头的位置关系，记录线段的起止点"""
    if p1_in or p2_in:  # 判断线段的方向
        info = None
        if line.k is None:  # 线垂直
            if p1_in:  # 下到上
                info = [line.p1[1], line.p2[1], cfg.DOWN]
            elif p2_in:  # 上到下
                info = [line.p1[1], line.p2[1], cfg.TOP]
        else:  # 线不垂直
            if p1_in:  # 右到左
                info = [line.p1[0], line.p2[0], cfg.RIGHT]
            elif p2_in:  # 左到右
                info = [line.p1[0], line.p2[0], cfg.LEFT]
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


def line_is_intersect_rec(line_li, rec):
    """在这里 line是一个长度为4的数组"""
    if helper.is_in(rec, (line_li[0], line_li[1], 0, 0)) or helper.is_in(rec, (line_li[2], line_li[3], 0, 0)):
        return True

    points = helper.get_rect_vertices(rec)
    vectors = list()
    p1 = (line_li[0], line_li[1])
    p2 = (line_li[2], line_li[3])
    line_vector = Vector(p1, p2)
    # print(line_vector)
    for point in points:
        vectors.append(Vector(p1, point))

    # [print(x) for x in vectors]
    vectors.sort(key=cmp_to_key(cross_product))
    # [print(x) for x in vectors]

    return cross_product(line_vector, vectors[0]) * cross_product(line_vector, vectors[-1]) <= 0


def get_line_seg_rec_dist(line, rec):
    """获取线段到矩形的距离，以及线段哪一端离rec更近"""
    rec_center = helper.get_rec_center(rec)

    dist_1 = helper.get_points_dist(line.p1, rec_center)
    dist_2 = helper.get_points_dist(line.p2, rec_center)

    if dist_1 < dist_2:
        return dist_1, 0
    else:
        return dist_2, 1


def get_line_rec_dist(line, rec):
    """获取矩形中心到直线的距离"""
    rec_center = helper.get_rec_center(rec)

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


def extend_line_to_rec(line, rect):
    center = helper.get_rec_center(rect)
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
    """找seqFlow与箭头相连的部分，确定seqFlow的尾端"""
    # arrow_lines:{arrow_id: {(k, b): info}} info: [start_point, end_point, direction]
    arrow_lines = dict()
    discrete_lines = list()
    for line in line_list:
        to_next = False
        intersected_arrows = list()
        for i, arrow in enumerate(arrows):
            p1_in = helper.point_is_in(arrow, line.p1)
            p2_in = helper.point_is_in(arrow, line.p2)
            info = get_line_info(line, p1_in, p2_in)
            if info is not None:  # 线段有一端在箭头中
                # cv.line(flows_cp, line.p1, line.p2, COLOR_GREEN, CONTOUR_THICKNESS)
                arrow_lines = add_info_to_arrow_lines(line, info, arrow_lines, i)
                to_next = True
                break
            else:  # 线段不与箭头直接连接
                if line_is_intersect_rec(line.li, arrow):
                    # 找到线段延长后经过的所有箭头
                    intersected_arrows.append(i)
        if not to_next:
            if len(intersected_arrows) > 0:  # 线段不在箭头中，但所在直线经过箭头
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
                    discrete_lines.append(line)
            else:  # 线段所在直线不经过箭头
                # cv.line(flows_cp, line.p1, line.p2, COLOR_BLUE, CONTOUR_THICKNESS)
                discrete_lines.append(line)

    # 若是独立箭头，连接箭头与部分分离线段，箭头位于线段一端的垂线上，否则合并相似连线
    for i in range(len(arrows)):
        arrow_line = arrow_lines.get(i)
        if arrow_line is None:
            # 没有连接线的箭头

            connected = False
            for discrete_line in discrete_lines:
                # 没有连接线的箭头，但是附近有与它垂直的线段，将它们连接起来

                # 计算箭头中心到线段两端的距离，找到距离较近的那一段
                seg_dist = get_line_seg_rec_dist(discrete_line, arrows[i])
                # 计算箭头中心到线段所在直线的距离
                line_dist = get_line_rec_dist(discrete_line, arrows[i])
                if seg_dist[0] < 60 and 0 <= seg_dist[0] - line_dist < 5:
                    connected = True
                    begin = (discrete_line.li[seg_dist[1] * 2], discrete_line.li[seg_dist[1] * 2 + 1])
                    end = helper.get_rec_center(arrows[i])
                    new_line = create_line((begin[0], begin[1], end[0], end[1]))
                    p1_in = helper.point_is_in(arrows[i], new_line.p1)
                    p2_in = helper.point_is_in(arrows[i], new_line.p2)
                    info = get_line_info(new_line, p1_in, p2_in)
                    arrow_lines[i] = {(new_line.k, new_line.b): info}

            # 没有连接线也没有临近的垂直线段的箭头
            if not connected:
                # 依据箭头图像，判断箭头及其连接线的方向，这里筛选出的孤立箭头，通常情况下只会是一个箭头，不会是多个箭头的融合体
                # print("found_not_connected_arrow")
                arrow = helper.dilate(arrows[i], 1)
                arrow_img = helper.truncate(input_img, arrow)
                arrow_line = get_seq_arrow_direction(arrow_img, arrow)
                arrow_lines[i] = arrow_line

        else:
            # 已有连接线的箭头, 将相似线段合并，不知道什么原因在normalize阶段，有一些线段没合并
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
                    normalized_line = extend_line_to_rec(normalized_line, arrow)
                    temp = get_line_seg_rec_dist(normalized_line, arrow)
                    info = get_line_info(normalized_line, temp[1] == 0, temp[1] == 1)
                arrow_line[(normalized_line.k, normalized_line.b)] = info
            arrow_lines[i] = arrow_line

    return arrow_lines, discrete_lines


def get_arrow_ele_direct(arrow, ele):
    arrow_center = helper.get_rec_center(arrow)
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
            return cfg.LEFT
        else:  # 在右边
            return cfg.RIGHT
    else:  # 竖直
        if arrow[1] < ele_shrink[1]:  # 在上边
            return cfg.TOP
        else:  # 在下边
            return cfg.DOWN


def get_opposite_direct_list(direct):
    if direct == cfg.TOP:
        return [cfg.DOWN]
    elif direct == cfg.DOWN:
        return [cfg.TOP]
    elif direct == cfg.RIGHT:
        return [cfg.TOP, cfg.DOWN, cfg.LEFT]
    elif direct == cfg.LEFT:
        return [cfg.TOP, cfg.RIGHT, cfg.DOWN]


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

    if direct == cfg.TOP:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] + arrow[3]
    elif direct == cfg.DOWN:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] - default_height
    elif direct == cfg.RIGHT:
        p_x = arrow[0] - default_width
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2
    elif direct == cfg.LEFT:
        p_x = arrow[0] + arrow[2]
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2
    else:
        p_x = arrow[0] + arrow[2] // 2 - default_width // 2
        p_y = arrow[1] + arrow[3] // 2 - default_height // 2

    default_element = [int(p_x), int(p_y), default_width, default_height]
    return default_element


def get_possible_element(same_ele_arrows, arrows):
    """通过指向同一元素的多个箭头来确定元素的bounding box"""
    """经过之前的逻辑筛选，输入的多个箭头，不会是在元素的同一条边上"""

    # default_width = 100
    # default_height = 80
    #
    # rec_top = float("inf")
    # rec_bottom = -1
    # rec_left = float("inf")
    # rec_right = -1
    #
    # no_top = False
    # no_bottom = False
    # no_left = False
    # no_right = False
    #
    # top = None
    # bottom = None
    # left = None
    # right = None
    #
    # rec = [-1, -1, -1, -1]
    #
    # ele_arrows = defaultdict(list)
    # for [arrow_id, arrow_direct] in same_ele_arrows:
    #     arrow = arrows[arrow_id]
    #     arrow_center = helper.get_rec_center(arrow)
    #     ele_arrows[arrow_direct].append(arrow_center)
    #
    # top_arrows = ele_arrows[cfg.TOP]
    # if len(top_arrows) > 0:
    #     for arrow in top_arrows:
    #         if arrow[1] < rec_top:
    #             rec_top = arrow[1]
    #             top = arrow
    # else:
    #     no_top = True
    #
    # down_arrows = ele_arrows[cfg.DOWN]
    # if len(down_arrows) > 0:
    #     for arrow in top_arrows:
    #         if arrow[1] > rec_bottom:
    #             rec_bottom = arrow[1]
    #             bottom = arrow
    # else:
    #     no_bottom = True
    #
    # left_arrows = ele_arrows[cfg.LEFT]
    # if len(left_arrows) > 0:
    #     for arrow in left_arrows:
    #         if arrow[0] < rec_left:
    #             rec_left = arrow[0]
    #             left = arrow
    #
    #         if no_bottom and arrow[1] > rec_bottom:
    #             rec_bottom = arrow[1]
    #             bottom = arrow
    #
    #         if no_top and arrow[1] < rec_top:
    #             rec_top = arrow[1]
    #             top = arrow
    # else:
    #     no_left = True
    #
    # right_arrows = ele_arrows[cfg.RIGHT]
    # if len(right_arrows) > 0:
    #     for arrow in right_arrows:
    #         if arrow[0] > rec_right:
    #             rec_right = arrow[0]
    #             right = arrow
    #
    #         if no_bottom and arrow[1] > rec_bottom:
    #             rec_bottom = arrow[1]
    #             bottom = arrow
    #
    #         if no_top and arrow[1] < rec_top:
    #             rec_top = arrow[1]
    #             top = arrow
    # else:
    #     no_right = True
    #
    # print("top:", rec_top)
    # print("down:", rec_bottom)
    # print("left:", rec_left)
    # print("right:", rec_right)
    #
    # if no_left and no_right:
    #     rec[1] = top[1]
    #     rec[3] = bottom[1] - top[1]
    #     width = max(int(rec[3] / 3 * 4), default_width)
    #     x_center = (top[0] + bottom[0]) // 2
    #     rec[0] = max(0, x_center - width // 2)
    #     rec[2] = width
    #     return rec
    # elif no_left:
    #     # only has right
    #     if not no_top and not no_bottom:
    #         # has top and bottom
    #         rec[3] = bottom[1] - top[1]
    #         rec[1] = top[1]
    #         x_center = (top[0] + bottom[0]) // 2
    #         width = max(2 * (right[0] - x_center), default_width)
    #         rec[0] = max(0, x_center - width // 2)
    #         rec[2] = width
    #         return rec
    #     elif top[1] < right[1] and top[0] < right[0] - 5:
    #         # has valid top
    #         rec[1] = top[1]
    #         if bottom[1] > right[1] and bottom[0] < right[0] - 5:
    #             # has valid bottom
    #             rec[3] = bottom[1] - top[1]
    #         else:
    #             # has no valid bottom
    #             rec[3] = max(2 * (right[1] - top[1]), default_height)
    #         width = max(2 * (right[0] - top[0]), default_width)
    #         rec[0] = max(0, right[0] - width)
    #         rec[2] = right[0] - rec[0]
    #         return rec
    #     else:
    #         # has no valid top
    #         if bottom[1] > right[1] and bottom[0] < right[0] - 5:
    #             # has valid bottom
    #             rec[3] = max(2 * (bottom[1] - right[1]), default_height)
    #
    #         else:
    #             print("only right is not possible here")

    # has valid bottom

    x_list = []
    y_list = []
    for [arrow_id, _] in same_ele_arrows:
        arrow = arrows[arrow_id]
        arrow_center = helper.get_rec_center(arrow)
        x_list.append(arrow_center[0])
        y_list.append(arrow_center[1])

    p_x = int(np.mean(x_list))
    p_y = int(np.mean(y_list))
    height = max(y_list) - min(y_list)
    width = max(x_list) - min(x_list)
    length = max(height, width, 80)

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


def extend_line(line, direct, axis_length):
    """延长线段，按照指定方向延长线段，在坐标轴上的投影延长axis_length"""
    """如果direct是cfg.TOP, 则表示，线段从上往下延长"""
    """如果direct是True, 则表示往x轴或y轴坐标较大的方向延长"""

    if line.k is None:
        if direct == cfg.TOP or not direct:
            p1 = line.p1
            p2 = [line.p2[0], line.p2[1] + axis_length]
            return points_to_line(p1, p2), p2
        elif direct == cfg.DOWN or direct:
            p1 = [line.p1[0], max(line.p1[1] - axis_length, 0)]
            p2 = line.p2
            return points_to_line(p1, p2), p1
    else:
        if direct == cfg.LEFT or not direct:
            p1 = line.p1
            p2_x = line.p2[0] + axis_length
            p2_y = helper.get_func_value(p2_x, line.k, line.b)
            p2 = [p2_x, p2_y]
            return points_to_line(p1, p2), p2
        elif direct == cfg.RIGHT or direct:
            p1_x = max(0, line.p1[0] - axis_length)
            p1_y = helper.get_func_value(p1_x, line.k, line.b)
            p1 = [p1_x, p1_y]
            p2 = line.p2
            return points_to_line(p1, p2), p1

    return None


def match_arrows_and_elements(arrow_lines, arrows):
    """匹配 arrow 与 element"""
    # 依据两者位置关系，删除一些线
    # 依据 arrow 确定一些未检出的元素
    # arrow_ele_map :{ arrow_id : [ele_path, arrow_direction]}
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
                    # 确定arrow所在的lane
                    if helper.is_in(lane, arrow):
                        sub_procs_in_lane = sub_procs.get(lane_id, None)
                        elements_in_lane = elements.get(lane_id)
                        dilate_arrow = helper.dilate(arrow, cfg.RECT_DILATION_VALUE)

                        arrow_line = arrow_lines.get(arrow_id)
                        # print(arrow_line)
                        one_key = list(arrow_line.keys())[0]
                        one_line_info = arrow_line[one_key]
                        one_line = info_to_line(one_line_info, one_key[0], one_key[1])
                        _, one_point = extend_line(one_line, one_line_info[2], 10)

                        if elements_in_lane is not None:

                            # 找与arrow匹配的node元素
                            for ele_id, ele in enumerate(elements_in_lane):
                                if helper.point_is_in(helper.dilate(ele, cfg.RECT_DILATION_VALUE), one_point):
                                    found = True
                                    arrow_direct = get_arrow_ele_direct(arrow, ele)
                                    # if arrow_line is not None and len(arrow_line) > 1:
                                    #     arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                                    arrow_ele_map[arrow_id] = [(pool_id, lane_id, ele_id, 0), arrow_direct]
                                    break

                        if not found and sub_procs_in_lane is not None:
                            # 若没有匹配的元素，则判断是否是指向subProcess的arrow
                            for sub_proc_id, sub_proc in enumerate(sub_procs_in_lane):
                                if not helper.is_in(sub_proc, dilate_arrow) and helper.is_overlap(dilate_arrow,
                                                                                                  sub_proc):
                                    found = True
                                    arrow_direct = get_arrow_ele_direct(arrow, sub_proc)
                                    # arrow_line = arrow_lines.get(arrow_id)
                                    # if arrow_line is not None and len(arrow_line) > 1:
                                    #     arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                                    arrow_ele_map[arrow_id] = [(pool_id, lane_id, sub_proc_id, 1), arrow_direct]
                                    break
                        #
                        if not found:  # 剩下的箭头通过连接线来判断方向
                            # print("one arrow can't match element")
                            # print(arrows[arrow_id])
                            arrow_line = arrow_lines.get(arrow_id)
                            # print(arrow_line)
                            if arrow_line is not None:
                                values = list(arrow_line.values())
                                direct_count = [[cfg.TOP, 0], [cfg.RIGHT, 0], [cfg.DOWN, 0], [cfg.LEFT, 0]]
                                for value in values:
                                    direct = value[2]
                                    direct_count[direct][1] += 1
                                # print(direct_count)
                                direct_count.sort(key=lambda x: x[1], reverse=True)
                                # print(direct_count[0][0])
                                arrow_direct = direct_count[0][0]
                                ele_rect = create_virtual_element(arrow, arrow_direct)

                                possible_ele[arrow_id] = [ele_rect, arrow_direct]
                            else:
                                print("one arrow has no arrow lines")
                        break
                break

    if len(possible_ele) > 0:
        keys = list(possible_ele.keys())
        while len(keys) > 0:
            one_arrow_id = keys.pop(0)
            poss_ele = possible_ele[one_arrow_id]
            ele_rec = poss_ele[0]
            same_ele_arrows = [[one_arrow_id, poss_ele[1]]]
            for i, key in enumerate(keys):
                if helper.is_overlap(ele_rec, possible_ele[key][0]):
                    same_ele_arrows.append([key, possible_ele[key][1]])
                    keys[i] = None
            keys = list(filter(lambda x: x is not None, keys))
            if len(same_ele_arrows) > 1:
                ele_rec = get_possible_element(same_ele_arrows, arrows)
            ele_path = add_one_element_to_pool(ele_rec)
            if ele_path is not None:
                # ele = get_element_rec_by_path(ele_path)
                for [arrow_id, arrow_direct] in same_ele_arrows:
                    # arrow = arrows[arrow_id]
                    # arrow_direct = get_arrow_ele_direct(arrow, ele)
                    arrow_lines = filter_arrow_lines(arrow_id, arrow_direct, arrow_lines)
                    arrow_ele_map[arrow_id] = [ele_path, arrow_direct]

    return arrow_lines, arrow_ele_map


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


def is_begin_point(point, line, end_ele_id):
    """判断是否是顺序流的起点"""
    point_rec = helper.dilate((point[0], point[1], 0, 0), 15)
    overlapped_ele = {}
    for i in range(len(all_elements)):
        ele_rec = get_element_rec_by_id(i)
        if all_elements[i][3] == 0:
            if i != end_ele_id and helper.is_overlap(ele_rec, point_rec):
                overlapped_ele[i] = ele_rec
        else:
            if i != end_ele_id and not helper.is_in(ele_rec, point_rec) and helper.is_overlap(ele_rec, point_rec):
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


def point_is_near_line_with_d(line, point, d):
    borders = get_parallel_lines_with_d(line.k, line.b, d)
    return point_is_between(point, borders[0], borders[1])


def is_extension_line(last_begin, curr_begin, line):
    p1 = is_extension_point(last_begin, curr_begin, line.p1)
    p2 = is_extension_point(last_begin, curr_begin, line.p2)
    return p1 and p2


def is_extension_point(last_begin, curr_begin, point):
    # print(last_begin, curr_begin, point)
    curr_line = Line([last_begin[0], last_begin[1], curr_begin[0], curr_begin[1]])
    is_near = point_is_near_line_with_d(curr_line, point, 3)

    if is_near:
        if curr_line.k is None:
            v1 = curr_begin[1] - last_begin[1]
            v2 = point[1] - last_begin[1]
        else:
            v1 = curr_begin[0] - last_begin[0]
            v2 = point[0] - last_begin[0]

        if v1 * v2 > 0:
            return True
    return False


def point_is_on_line_seg(line, point, d):
    """判断点是否在线段附近"""
    """该点在线段所在直线上"""
    """返回值大于等于0表示 交点在线段延长线上，0表示线段的p1离交点较远, 1则对应p2"""
    d1 = helper.get_points_dist(line.p1, point)
    d2 = helper.get_points_dist(line.p2, point)
    length = line.get_points_dist()

    if d1 <= length and d2 <= length:
        if d1 + d2 <= length + 3:
            if d1 < 5:
                return True, 1
            elif d2 < 5:
                return True, 0
            else:
                return True, -1
    elif d1 > length and d2 <= d:
        return True, 2
    elif d1 <= d and d2 > length:
        return True, 3

    return False, -2


def get_ele_rec_pointed_by_curr_line(curr_line, reverse_tag):
    """获取当前flow起始段延长后相交的最近元素"""
    curr_line_ele_rec = []
    curr_line_ele_id = -1
    min_dist = float("inf")
    for ele_id in range(len(all_elements)):
        ele_rec = get_element_rec_by_id(ele_id)
        if line_is_intersect_rec(curr_line.li, ele_rec):
            dist, point_id = get_line_seg_rec_dist(curr_line, ele_rec)
            if (reverse_tag and point_id == 1) or (not reverse_tag and point_id == 0):
                if dist < min_dist:
                    min_dist = dist
                    curr_line_ele_rec = ele_rec
                    curr_line_ele_id = ele_id
    return curr_line_ele_rec, curr_line_ele_id


def get_line_rec_intersection(line, reverse, rec):
    """获取线段延长后与rec的交点"""
    if line_is_intersect_rec(line.li, rec):
        vertices = helper.get_rect_vertices(rec)
        if line.k is None:
            if reverse:
                rec_line = points_to_line(vertices[0], vertices[1])
                intersection = get_point_of_intersection(line, rec_line)
                if is_extension_point(line.p1, line.p2, intersection):
                    return intersection
            else:
                rec_line = points_to_line(vertices[2], vertices[3])
                intersection = get_point_of_intersection(line, rec_line)
                if is_extension_point(line.p2, line.p1, intersection):
                    return intersection
        else:
            if reverse:
                rec_line = points_to_line(vertices[0], vertices[3])
                intersection = get_point_of_intersection(line, rec_line)

                if intersection[1] > vertices[3][1]:
                    rec_line = points_to_line(vertices[2], vertices[3])
                    intersection = get_point_of_intersection(line, rec_line)
                elif intersection[1] < vertices[0][1]:
                    rec_line = points_to_line(vertices[0], vertices[1])
                    intersection = get_point_of_intersection(line, rec_line)

                if is_extension_point(line.p1, line.p2, intersection):
                    return intersection
            else:
                rec_line = points_to_line(vertices[1], vertices[2])
                intersection = get_point_of_intersection(line, rec_line)

                if intersection[1] > vertices[3][1]:
                    rec_line = points_to_line(vertices[2], vertices[3])
                    intersection = get_point_of_intersection(line, rec_line)
                elif intersection[1] < vertices[0][1]:
                    rec_line = points_to_line(vertices[0], vertices[1])
                    intersection = get_point_of_intersection(line, rec_line)

                if is_extension_point(line.p2, line.p1, intersection):
                    return intersection
    return None


def detect_one_flow(flow_points, discrete_line_ids, end_ele_id, discrete_lines, flows, arrow_id, merged_lines):
    curr_begin = flow_points[-1]
    last_begin = flow_points[-2]
    # print(curr_begin, last_begin)
    curr_line = Line([last_begin[0], last_begin[1], curr_begin[0], curr_begin[1]])

    if curr_line.k is None:
        dim = 1
    else:
        dim = 0

    # True表示 flow尾部向x轴变大或y轴变大的方向延长, 坐标系为 左x下y
    reverse_tag = curr_begin[dim] - last_begin[dim] > 0

    curr_begin_rec = helper.dilate([curr_begin[0], curr_begin[1], 0, 0], 3)

    connected_lines = dict()
    rest_lines = []

    for line_id in discrete_line_ids:
        discrete_line = discrete_lines[line_id]

        if not is_parallel(curr_line, discrete_line):
            if helper.point_is_in(curr_begin_rec, discrete_line.p1):
                connected_lines[line_id] = [discrete_line.p2]
            elif helper.point_is_in(curr_begin_rec, discrete_line.p2):
                connected_lines[line_id] = [discrete_line.p1]
            else:
                rest_lines.append(line_id)
        else:
            rest_lines.append(line_id)

    to_extend = True
    if len(connected_lines) > 0:
        to_extend = False

    if to_extend:
        is_begin = is_begin_point(curr_begin, curr_line, end_ele_id)
        if is_begin[0]:
            flow = [end_ele_id, flow_points, is_begin[1]]
            add_one_flow(flows, flow, arrow_id)
            return

        to_extend_lines = []
        other_lines = []

        for line_id in rest_lines:
            discrete_line = discrete_lines[line_id]
            if is_extension_line(last_begin, curr_begin, discrete_line):
                # print(discrete_line)
                d1 = helper.get_points_dist(discrete_line.p1, curr_begin)
                d2 = helper.get_points_dist(discrete_line.p2, curr_begin)
                if min(d1, d2) < 80:
                    to_extend_lines.append(line_id)
                else:
                    other_lines.append(line_id)
            else:
                other_lines.append(line_id)

        # 找curr_line指向的最近的元素，作为延长的边界
        curr_line_ele_rec, _ = get_ele_rec_pointed_by_curr_line(curr_line, reverse_tag)

        curr_line_points = []
        if len(curr_line_ele_rec) > 0:
            # 找到作为curr_line延长边界的元素
            # 从共线的线段中筛选出可以合并到curr_line的线段
            rec_center = helper.get_rec_center(curr_line_ele_rec)
            for line_id in to_extend_lines:
                one_li = discrete_lines[line_id]

                if (reverse_tag and one_li.p1[dim] < rec_center[dim] and one_li.p2[dim] < rec_center[dim]) or \
                        (not reverse_tag and one_li.p1[dim] > rec_center[dim] and one_li.p2[dim] > rec_center[dim]):
                    curr_line_points.append(one_li.p1)
                    curr_line_points.append(one_li.p2)
                    merged_lines[line_id].append(arrow_id)
                else:
                    other_lines.append(line_id)
        else:
            # 没找到作为curr_line延长边界的元素
            # 所有的都可以合并
            for line_id in to_extend_lines:
                one_li = discrete_lines[line_id]
                curr_line_points.append(one_li.p1)
                curr_line_points.append(one_li.p2)
                merged_lines[line_id].append(arrow_id)

        # 合并线段到 curr_line
        if len(curr_line_points) > 0:
            curr_line_points.sort(key=lambda x: x[dim], reverse=reverse_tag)
            temp = curr_line_points[0]
            if curr_line.k is None:
                new_begin = [curr_begin[0], temp[1]]
            else:
                new_begin = [temp[0], helper.get_func_value(temp[0], curr_line.k, curr_line.b)]
            flow_points[-1] = new_begin
            discrete_line_ids = other_lines
            detect_one_flow(flow_points, discrete_line_ids, end_ele_id, discrete_lines, flows, arrow_id, merged_lines)
            return

    curr_line, _ = extend_line(curr_line, reverse_tag, 10)

    next_discrete_lines = []
    for line_id in rest_lines:
        discrete_line = discrete_lines[line_id]
        p1_near = point_is_near_line_with_d(curr_line, discrete_line.p1, 20)
        p2_near = point_is_near_line_with_d(curr_line, discrete_line.p2, 20)

        if (p1_near and not p2_near) or (p2_near and not p1_near):
            # if not (p1_near and p2_near):
            intersection = get_point_of_intersection(discrete_line, curr_line)
            if is_extension_point(last_begin, curr_begin, intersection):
                on_curr = point_is_on_line_seg(curr_line, intersection, 30)
                on_disc = point_is_on_line_seg(discrete_line, intersection, 30)
                if on_curr[0] and on_disc[1] >= 0:
                    # if on_disc[1] >= 0:
                    if on_disc[1] % 2 == 0:
                        connected_lines[line_id] = [discrete_line.p1]
                    else:
                        connected_lines[line_id] = [discrete_line.p2]
                else:
                    next_discrete_lines.append(line_id)
            else:
                next_discrete_lines.append(line_id)
        else:
            next_discrete_lines.append(line_id)

    discrete_line_ids = next_discrete_lines
    if len(connected_lines) == 0:
        flow = [end_ele_id, flow_points, None]
        flows[arrow_id].append(flow)
    else:
        line_ids = list(connected_lines.keys())
        flow_points_list = list(map(lambda x: flow_points.copy(), line_ids))
        for i, line_id in enumerate(line_ids):
            merged_lines[line_id].append(arrow_id)
            # discrete_line_ids.remove(line_id)
            one_line = discrete_lines[line_id]
            intersection = get_point_of_intersection(curr_line, one_line)
            flow_points_list[i][-1] = intersection
            new_begins = connected_lines[line_id]
            if len(new_begins) == 1:
                flow_points_list[i].append(new_begins[0])
                detect_one_flow(flow_points_list[i], discrete_line_ids, end_ele_id, discrete_lines, flows, arrow_id,
                                merged_lines)
            elif len(new_begins) > 1:
                for new_begin in new_begins:
                    flow_points_i = flow_points_list[i].copy()
                    flow_points_i.append(new_begin)
                    detect_one_flow(flow_points_i, discrete_line_ids, end_ele_id, discrete_lines, flows, arrow_id,
                                    merged_lines)


def connect_elements(arrows, arrow_lines, arrow_ele_map, discrete_lines):
    """从seqFlow尾端，递归回溯到连接线的初始位置"""
    # flows: {arrow_id: [[end_ele_id, flow_points, start_ele_id]]}
    flows = defaultdict(list)
    merged_lines = defaultdict(list)

    discrete_line_ids = list(range(len(discrete_lines)))
    for arrow_id in range(len(arrows)):
        ele_map = arrow_ele_map.get(arrow_id)
        arrow_line = arrow_lines.get(arrow_id)
        end_ele_id = get_element_id(ele_map[0])
        if arrow_line is not None:
            for key, info in arrow_line.items():
                line = info_to_line(info, key[0], key[1])
                flow_points = list()

                if info[2] == cfg.TOP or info[2] == cfg.LEFT:
                    flow_points.extend([line.p2, line.p1])
                else:
                    flow_points.extend([line.p1, line.p2])
                detect_one_flow(flow_points, discrete_line_ids, end_ele_id, discrete_lines, flows, arrow_id,
                                merged_lines)

    not_merged_lines = []
    shared_lines = []
    for line_id in discrete_line_ids:
        share_line_arrow_ids = merged_lines[line_id]
        arrows_num = len(share_line_arrow_ids)
        if arrows_num == 0:
            not_merged_lines.append(line_id)
        elif arrows_num > 1:
            shared_lines.append(line_id)

    # 提取 一到多 中共享线段flow的信息
    share_line_flow_info = defaultdict(list)
    to_split_shared_lines = []
    for line_id in shared_lines:
        discrete_line = discrete_lines[line_id]
        share_line_arrow_ids = merged_lines[line_id]
        for arrow_id in share_line_arrow_ids:
            arrow_flows = flows[arrow_id]
            for flow_id, one_flow in enumerate(arrow_flows):
                if one_flow[2] is None:
                    p1_rec = helper.dilate([discrete_line.p1[0], discrete_line.p1[1], 0, 0], 3)
                    p2_rec = helper.dilate([discrete_line.p2[0], discrete_line.p2[1], 0, 0], 3)
                    one_flow_points = one_flow[1]
                    flow_end_part = []
                    for point_id, point in enumerate(one_flow_points):
                        if helper.point_is_in(p1_rec, point) or helper.point_is_in(p2_rec, point):
                            flow_end_part = one_flow_points[:(point_id + 1)]
                            break
                    if len(flow_end_part) > 0:
                        share_line_flow_info[line_id].append([arrow_id, flow_id, flow_end_part])
                        to_split_shared_lines.append(line_id)
                        break

    for arrow_id in range(len(arrows)):
        arrow_flows = flows[arrow_id]
        for flow_id, one_flow in enumerate(arrow_flows):
            one_flow_points = one_flow[1]
            if one_flow[2] is not None:
                # 处理 一到多 多的一边最外围的两条连接线
                for point_id in range(len(one_flow_points) - 1):
                    last_begin = one_flow_points[point_id]
                    curr_begin = one_flow_points[point_id + 1]
                    curr_flow_seg = points_to_line(last_begin, curr_begin)

                    for line_id, share_line_flows in share_line_flow_info.items():
                        discrete_line = discrete_lines[line_id]
                        intersection = get_point_of_intersection(curr_flow_seg, discrete_line)
                        if intersection is not None:
                            on_curr = point_is_on_line_seg(curr_flow_seg, intersection, 5)
                            on_disc = point_is_on_line_seg(discrete_line, intersection, 5)
                            if on_curr[1] == -1 and on_disc[1] == -1:
                                flow_begin_part = one_flow_points[(point_id + 1):]
                                flow_begin_part.insert(0, intersection)
                                for [flow_arrow_id, share_flow_id, flow_end_part] in share_line_flows:
                                    flow_end_part.extend(flow_begin_part)
                                    share_line_flow = flows[flow_arrow_id][share_flow_id]
                                    share_line_flow[1] = flow_end_part
                                    share_line_flow[2] = one_flow[2]
                                break

            if one_flow[2] is None:
                # 将没有找到开始节点的顺序流与共线的离散线段相连
                last_begin = one_flow_points[-2]
                curr_begin = one_flow_points[-1]
                curr_line = Line([last_begin[0], last_begin[1], curr_begin[0], curr_begin[1]])
                if curr_line.k is None:
                    dim = 1
                else:
                    dim = 0
                # True表示 flow尾部向x轴变大或y轴变大的方向延长, 坐标系为 左x下y
                reverse_tag = curr_begin[dim] - last_begin[dim] > 0

                to_extend_lines = []
                curr_line_points = []
                for line_id in not_merged_lines:
                    discrete_line = discrete_lines[line_id]
                    if is_extension_line(last_begin, curr_begin, discrete_line):
                        d1 = helper.get_points_dist(discrete_line.p1, curr_begin)
                        d2 = helper.get_points_dist(discrete_line.p2, curr_begin)
                        if min(d1, d2) < 110:
                            curr_line_points.append(discrete_line.p1)
                            curr_line_points.append(discrete_line.p2)
                            to_extend_lines.append(line_id)

                if len(to_extend_lines) > 0:
                    # print("extended")
                    for line_id in to_extend_lines:
                        not_merged_lines.remove(line_id)
                    curr_line_points.sort(key=lambda x: x[dim], reverse=reverse_tag)
                    temp = curr_line_points[0]
                    if curr_line.k is None:
                        new_begin = [curr_begin[0], temp[1]]
                    else:
                        new_begin = [temp[0], helper.get_func_value(temp[0], curr_line.k, curr_line.b)]
                    one_flow_points[-1] = new_begin

            # 处理 多到一 的情况
            one_flow_points = one_flow[1]
            last_begin = one_flow_points[-2]
            curr_begin = one_flow_points[-1]
            curr_flow_seg = points_to_line(last_begin, curr_begin)

            # new_merged_lines = defaultdict(list)
            connected_lines = dict()
            for line_id in not_merged_lines:
                discrete_line = discrete_lines[line_id]

                intersection = get_point_of_intersection(curr_flow_seg, discrete_line)
                if intersection is not None and is_extension_point(last_begin, curr_begin, intersection):
                    on_curr = point_is_on_line_seg(curr_flow_seg, intersection, 100)
                    on_disc = point_is_on_line_seg(discrete_line, intersection, 20)

                    if on_curr[0]:
                        if on_disc[1] >= 0:
                            if on_disc[1] % 2 == 0:
                                connected_lines[line_id] = [discrete_line.p1]
                            elif on_disc[1] % 2 == 1:
                                connected_lines[line_id] = [discrete_line.p2]
                        elif on_disc[1] == -1:
                            connected_lines[line_id] = [discrete_line.p1, discrete_line.p2]

            if len(connected_lines) > 0:
                new_merged_lines = defaultdict(list)
                line_ids = list(connected_lines.keys())
                flow_points_list = list(map(lambda x: one_flow_points.copy(), line_ids))
                for i, line_id in enumerate(line_ids):
                    not_merged_lines.remove(line_id)
                    one_line = discrete_lines[line_id]
                    intersection = get_point_of_intersection(curr_flow_seg, one_line)
                    flow_points_list[i][-1] = intersection
                    new_begins = connected_lines[line_id]
                    if len(new_begins) == 1:
                        flow_points_list[i].append(new_begins[0])
                        detect_one_flow(flow_points_list[i], not_merged_lines, one_flow[0], discrete_lines, flows,
                                        arrow_id, new_merged_lines)
                    elif len(new_begins) > 1:
                        for new_begin in new_begins:
                            flow_points_i = flow_points_list[i].copy()
                            flow_points_i.append(new_begin)
                            detect_one_flow(flow_points_i, not_merged_lines, one_flow[0], discrete_lines, flows,
                                            arrow_id, new_merged_lines)
                if one_flow[2] is None:
                    arrow_flows.pop(flow_id)

                keys = set(list(new_merged_lines.keys()))
                for key in keys:
                    try:
                        not_merged_lines.remove(key)
                    except ValueError:
                        continue

    # 下面处理 一到多 的剩余情形 complete flows
    not_completed_flows = []
    completed_flows = []

    for arrow_id in range(len(arrows)):
        arrow_flows = flows[arrow_id]
        for flow_id, one_flow in enumerate(arrow_flows):
            if one_flow[2] is None:
                not_completed_flows.append([arrow_id, flow_id])
                # print([arrow_id, flow_id])
                # print(one_flow)
            else:
                completed_flows.append([arrow_id, flow_id])

    not_completed_flows, _ = merge_flows(completed_flows, not_completed_flows, flows)

    merge_flows(not_completed_flows, not_completed_flows, flows, False)

    return flows, discrete_lines


def merge_flows(completed_flows, not_completed_flows, flows, remove=True):
    adjust_flows = []
    for [c_arrow_id, c_flow_id] in completed_flows:
        completed_flow = flows[c_arrow_id][c_flow_id]
        one_flow_points = completed_flow[1]
        for point_id in range(len(one_flow_points) - 1):
            last_begin = one_flow_points[point_id]
            curr_begin = one_flow_points[point_id + 1]
            curr_flow_seg = points_to_line(last_begin, curr_begin)
            for [arrow_id, flow_id] in not_completed_flows:
                if arrow_id != c_arrow_id:
                    not_completed_flow = flows[arrow_id][flow_id]
                    no_begin_line = points_to_line(not_completed_flow[1][-1], not_completed_flow[1][-2])

                    if not is_parallel(curr_flow_seg, no_begin_line):
                        intersection = get_point_of_intersection(curr_flow_seg, no_begin_line)
                        # if intersection is not None:
                        on_curr = point_is_on_line_seg(curr_flow_seg, intersection, 5)
                        on_disc = point_is_on_line_seg(no_begin_line, intersection, 5)

                        if on_disc[0] and on_curr[1] == -1:
                            if on_disc[1] >= 0 and not_completed_flow[0] != completed_flow[2]:
                                not_completed_flow[1].pop()

                                flow_begin_part = one_flow_points[(point_id + 1):]
                                flow_begin_part.insert(0, intersection)

                                not_completed_flow[1].extend(flow_begin_part)
                                not_completed_flow[2] = completed_flow[2]
                                adjust_flows.append([arrow_id, flow_id])
                                if remove:
                                    not_completed_flows.remove([arrow_id, flow_id])
                                break
    return not_completed_flows, adjust_flows


def get_p2_2_p1_direct(p1, p2):
    if p1[0] < p2[0]:  # 右到左
        direct = cfg.RIGHT
    elif p1[0] > p2[0]:  # 左到右
        direct = cfg.LEFT
    else:
        if p1[1] < p2[1]:  # 下到上
            direct = cfg.DOWN
        elif p1[1] > p2[1]:  # 上到下
            direct = cfg.TOP
        else:
            direct = -1
    return direct


def get_end_point_rec_dis(p1, p2, rec, ele_type):
    direct = get_p2_2_p1_direct(p1, p2)
    if ele_type == 0:
        rec_center = helper.get_rec_center(rec)
        valid = False

        if direct == cfg.TOP:
            if rec_center[1] >= (p1[1] - 30):
                valid = True
        elif direct == cfg.DOWN:
            if rec_center[1] <= (p1[1] + 30):
                valid = True
        elif direct == cfg.RIGHT:
            if rec_center[0] <= (p1[0] + 30):
                valid = True
        elif direct == cfg.LEFT:
            if rec_center[0] >= (p1[0] - 30):
                valid = True
        if valid:
            return helper.get_points_dist(rec_center, p1)
        else:
            return -1
    else:
        return -1


def get_begin_ele_id(p1, p2):
    begin_ele_id = -1
    begin_ele_rec = []
    min_dis = float("inf")
    for i in range(len(all_elements)):
        ele_type = all_elements[i][3]
        ele_rec = get_element_rec_by_id(i)
        dist = get_end_point_rec_dis(p1, p2, ele_rec, ele_type)

        if 0 < dist < min_dis:
            min_dis = dist
            begin_ele_id = i
            begin_ele_rec = ele_rec
    return begin_ele_rec, begin_ele_id


def complete_flow(begin_ele_id, flow_points):
    p1 = flow_points[-1]
    p2 = flow_points[-2]

    end_line = points_to_line(p1, p2)
    ele_rec = get_element_rec_by_id(begin_ele_id)
    ele_center = helper.get_rec_center(ele_rec)
    if line_is_intersect_rec(end_line.li, ele_rec):
        if end_line.k is None:
            end_point = (end_line.b, ele_center[1])
        else:
            end_point = (ele_center[0], int(end_line.k * ele_center[0] + end_line.b))
        flow_points[-1] = end_point
    else:
        point_rec = helper.dilate((p1[0], p1[1], 0, 0), 5)
        direct = get_arrow_ele_direct(point_rec, ele_rec)
        if direct == cfg.TOP or direct == cfg.DOWN:
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
    vertices = helper.get_rect_vertices(rec)

    if rec_type == 1 and not line_is_intersect_rec(line.li, rec):
        flow_points = complete_flow(begin_ele_id, flow_points)
        p1 = flow_points[-1]
        p2 = flow_points[-2]
        line = points_to_line(p1, p2)

    intersections = []
    for i, vertex in enumerate(vertices):
        one_border = points_to_line(vertex, vertices[(i + 1) % 4])
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
        intersections.sort(key=lambda x: helper.get_points_dist(p1, x))
        flow_points[-1] = intersections[0]
    return flow_points


def is_same(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]


def parse_img(file_path):
    global pools
    global input_img

    # 轮廓检测
    input_img, layers, contours, contours_rec, partial_elements, arrows = pre_process(file_path)

    # node元素定位(泳池，泳道，活动，事件，网关，子过程)
    pools, type_tag, partial_elements = pools_detector.get_pools(input_img, layers, contours_rec, partial_elements,
                                                                 arrows)
    show_im(input_img, "input")
    # pools_img = draw_pools(pools)
    # show_im(pools_img, "pools_img_no_elements")
    # pools = pools_detector.get_elements(input_img, layers, contours_rec, partial_elements, pools, type_tag)
    pools = pools_detector.get_elements(input_img, layers, contours_rec, partial_elements, pools, type_tag)
    # pools_img = draw_pools(pools)
    # show_im(pools_img, "raw_elements")

    # 移除node元素， 检测seqFlow的箭头
    flows_img = remove_elements(2)
    # arrows = get_arrows(flows_img)
    # show_im(flows_img, "flows_img")
    arrows = get_seq_arrows(flows_img)
    # arrows_img = helper.draw_rects(pools_img, arrows, cfg.COLOR_GREEN, 1)
    # show_im(arrows_img, "arrows_img")

    # 给所有flow-node(活动，事件，网关，子过程 元素编号，记录其所在泳池及泳道编号)
    get_all_elements()
    # print(all_elements)
    flows = defaultdict(list)
    if len(arrows) > 0:
        flows_img = remove_elements(3)
        # show_im(flows_img, "no_elements")

        # 移除文本
        flows_only = remove_text(flows_img)
        # show_im(flows_only, "no_text")

        # 直线检测
        line_list = detect_lines(flows_only)
        # print(line_list[0])
        # all_lines_img = helper.draw_lines(flows_only, [line_list[0]], cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
        # show_im(all_lines_img, "all_lines")
        # cv.waitKey(0)

        # 归一化检测出的直线 (调用两次效果比较好)
        line_list = normalize_all_lines(line_list)
        line_list = normalize_all_lines(line_list)
        background = np.zeros_like(flows_only)
        normalized_lines_img = helper.draw_lines(background, line_list, cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
        show_im(normalized_lines_img, "normalized_lines_img")

        # 获取与箭头相连的线段
        arrow_lines, discrete_lines = get_initial_lines(arrows, line_list)

        # 将箭头与元素匹配，并获取一些之前未检测出的元素
        arrow_lines, arrow_ele_map = match_arrows_and_elements(arrow_lines, arrows)
        #
        # for arrow_id in range(len(arrows)):
        #     print(arrow_id)
        #     print(arrow_ele_map[arrow_id])
        # print("-" * 50)

        # 去除有一端在元素里的离散线段
        for i, line in enumerate(discrete_lines):
            for ele_path in all_elements:
                if ele_path[3] == 0:
                    ele_rec = get_element_rec_by_path(ele_path)
                    if helper.point_is_in(ele_rec, line.p1) or helper.point_is_in(ele_rec, line.p2):
                        discrete_lines[i] = None
        discrete_lines = list(filter(lambda x: x is not None, discrete_lines))
        # discrete_lines_img = helper.draw_lines(flows_only, discrete_lines, cfg.COLOR_RED, cfg.CONTOUR_THICKNESS)
        # show_im(discrete_lines_img, "discrete_lines")
        # cv.waitKey(0)
        # 依据顺序流的末端，递归回溯，找到起点
        flows, discrete_lines = connect_elements(arrows, arrow_lines, arrow_ele_map, discrete_lines)

        for arrow_id in range(len(arrows)):
            arrow_flows = flows[arrow_id]
            for flow_id, flow in enumerate(arrow_flows):
                if flow[2] is None:
                    flow_points = flow[1]
                    _, begin_ele_id = get_begin_ele_id(flow_points[-1], flow_points[-2])
                    if begin_ele_id > 0 and begin_ele_id != flow[0]:
                        complete_flow_points = complete_flow(begin_ele_id, flow_points)
                        flow[1] = complete_flow_points
                        flow[2] = begin_ele_id
                    else:
                        arrow_flows[flow_id] = None
            flows[arrow_id] = list(filter(lambda x: x is not None, arrow_flows))
            for flow in flows[arrow_id]:
                flow_points = flow[1]
                final_flow_points = get_ele_edge_point(flow[2], flow_points)
                flow[1] = final_flow_points

        pools_img = draw_pools(pools)
        # show_im(pools_img, "pools_img_no_lines")

        # 画顺序流
        for arrow_id in range(len(arrows)):

            arrow_flows = flows.get(arrow_id)
            arrow_ele = arrow_ele_map.get(arrow_id)
            # 画箭头
            if (arrow_flows is None or len(arrow_flows) == 0) and (arrow_ele is None or len(arrow_ele) == 0):
                helper.draw_one_rect(pools_img, helper.dilate(arrows[arrow_id], 5), cfg.COLOR_RED,
                                     cfg.CONTOUR_THICKNESS)
            elif arrow_flows is None or len(arrow_flows) == 0:
                helper.draw_one_rect(pools_img, helper.dilate(arrows[arrow_id], 5), cfg.COLOR_BLUE,
                                     cfg.CONTOUR_THICKNESS)
            elif arrow_ele is None or len(arrow_ele) == 0:
                helper.draw_one_rect(pools_img, helper.dilate(arrows[arrow_id], 5), cfg.COLOR_RED,
                                     cfg.CONTOUR_THICKNESS)
            else:
                color = cfg.COLOR_GREEN
                if arrow_ele[0][3] == 1:
                    print("connect sub_process")
                    color = cfg.COLOR_BLUE
                helper.draw_one_rect(pools_img, helper.dilate(arrows[arrow_id], 5), color, cfg.CONTOUR_THICKNESS)

                # print(arrow_id)
                # print(arrows[arrow_id])
                for flow in arrow_flows:
                    # print(flow)

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

                    # pools_img_copy = np.zeros_like(pools_img)
                    if len(flow_points) >= 2:
                        if show_points:
                            print(flow_points)
                        for i in range(1, len(flow_points)):
                            p1 = (int(flow_points[i - 1][0]), int(flow_points[i - 1][1]))
                            p2 = (int(flow_points[i][0]), int(flow_points[i][1]))
                            cv.line(pools_img, p1, p2, color, cfg.CONTOUR_THICKNESS)
                    #         cv.line(pools_img_copy, p1, p2, color, cfg.CONTOUR_THICKNESS)
                    #         show_im(pools_img_copy, name="one_flow")
                    #         show_im(pools_img, name="pools_img")
                    # cv.waitKey(0)

    # show_im(pools_img, name="pools_img")
    # cv.waitKey(0)

    all_seq_flows = []
    if len(flows) > 0:
        for arrow_id in range(len(arrows)):
            arrow_flows = flows.get(arrow_id)
            # print(arrow_id)
            for flow in arrow_flows:
                all_seq_flows.append(flow)

    return all_seq_flows


def show_im(img_matrix, name="img", show=False):
    # pass
    # cv.namedWindow(name, cv.WINDOW_NORMAL)
    if show:
        cv.namedWindow(name)
        cv.imshow(name, img_matrix)
        # cv.waitKey(0)


def classify_elements(classifier, classifier_type):
    all_elements_type = []
    # print("Classifying begins...")
    # helper.print_time()
    # all_elements_images = []
    for ele_path in all_elements:
        if ele_path[3] == 1:
            all_elements_type.append(["subProcess_expanded", ""])
        else:
            ele_rec = get_element_rec_by_path(ele_path)
            ele_rec = helper.dilate(ele_rec, 10)
            ele_img = helper.truncate(input_img, ele_rec)

            ele_type = classifier.classify([ele_img], classifier_type)[0]
            text = ""
            if ele_type.endswith("ask") or ele_type in cfg.TASK_LIKE_LIST:
                text = translator.translate(ele_img)
            all_elements_type.append([ele_type, text])

            # all_elements_images.append(ele_img)

    # 直接分类所有
    # elements_type = classifier.classify(all_elements_images, classifier_type)
    # elements_text = translator.translate_images(all_elements_images)

    # print(all_elements_type)
    # helper.print_time()
    # print("Classifying finished!")

    return all_elements_type


def detect(file_path, classifier, classifier_type):
    all_seq_flows = parse_img(file_path)

    all_elements_type = classify_elements(classifier, classifier_type)
    definitions, all_elements_info = model_exporter.create_model(input_img, pools, all_elements, all_elements_type,
                                                                 all_seq_flows)
    return definitions, all_elements_info, all_seq_flows, all_elements, pools


def run():
    sample_dir = "samples/imgs/test/"
    images = os.listdir(sample_dir)

    classifier = Classifier()
    # [0, 5, 6, 10, 14, 15]
    size = len(images)
    selected = list(range(size))
    for i in selected:

        im = images[i]
        file_path = sample_dir + im
        if os.path.isfile(file_path):
            print("-" * 50)
            print(i)
            print(im)
            # detect(file_path, None, None)
            definitions, _, _, _, _ = detect(file_path, classifier, "vgg16_57")
            # model_exporter.export_xml(definitions, "output_1/{}.bpmn".format(im[0:-4]))


if __name__ == '__main__':
    run()
