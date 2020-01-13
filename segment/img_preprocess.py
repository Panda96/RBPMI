# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from functools import cmp_to_key

import cfg
from helper import detector_helper as helper
from helper.utils import Vector, points_to_line, get_point_of_intersection, get_float_point_of_intersection


def draw_contours_rec(input_img, contors, contours_rec, show_text=True, reverse=False):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        bound = contours_rec[i]
        bound_rect = bound[0]
        drawing = helper.draw_one_rect(drawing, bound_rect, cfg.COLOR_WHITE, cfg.CONTOUR_THICKNESS)

        if show_text:
            # contour index:contour area，area of bounding box， height-to-width-ratio
            text = "{}:{},{},{}".format(i, int(bound[1]), int(bound[2]), "%.2f" % (bound_rect[3] / bound_rect[2]))
            # text = chr(ord('A')+i)
            text_size = cv.getTextSize(text, cv.QT_STYLE_NORMAL, 0.3, 1)
            # put the text in the middle of a rectangle and not beyond the border of the image_mat
            org_x = bound_rect[0] + (bound_rect[2] - text_size[0][0]) // 2
            org_x = max(org_x, 2)
            org_x = min(org_x, drawing.shape[1] - text_size[0][0] - 5)
            cv.putText(drawing, text, (org_x, bound_rect[1] + (bound_rect[3] + text_size[0][1]) // 2),
                       cv.QT_STYLE_NORMAL, 0.3, cfg.COLOR_WHITE)

    if reverse:
        drawing = 255 - drawing

    return drawing


def draw_contours(input_img, contors, contours, reverse=False):
    drawing = np.zeros_like(input_img, dtype=np.uint8)
    for i in contors:
        cv.drawContours(drawing, contours, i, cfg.COLOR_WHITE, cfg.CONTOUR_THICKNESS, cv.LINE_8)

    if reverse:
        drawing = 255 - drawing

    return drawing


def get_layers_img(f, show=True, output=False, split=True, reverse=True, show_text=True):
    # input_img, layers, contours, contours_rec, partial_elements, arrows
    _, input_img, layers, contours, contours_rec, _, _ = pre_process(f, split)

    for model_i in range(len(layers)):
        layer = layers[model_i].keys()
        contours_drawing = draw_contours(input_img, layer, contours, reverse)
        rec_drawing = draw_contours_rec(input_img, layer, contours_rec, show_text, reverse)
        contours_drawing = helper.dilate_drawing(contours_drawing)
        rec_drawing = helper.dilate_drawing(rec_drawing)
        img = helper.dilate_drawing(input_img)

        if show:
            # show_im(img, "input")
            # show_im(contours_drawing, "contour")
            # show_im(rec_drawing, "rect")
            cv.imshow("input", img)
            cv.imshow("contour", contours_drawing)
            cv.imshow("rect", rec_drawing)
            cv.waitKey(0)
        if output:
            output_dir = "samples/layers/test/" + f.split("/")[-1][:-4]
            if split:
                output_dir += "_split"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            contour_file = output_dir + "/" + "layer_{}_contour.png".format(model_i)
            rec_file = output_dir + "/" + "layer_{}_rec.png".format(model_i)
            cv.imwrite(contour_file, contours_drawing)
            cv.imwrite(rec_file, rec_drawing)
    cv.destroyAllWindows()


def get_contours(image, split=True):
    # 形态学变化，将图中箭头连接处断开
    morph_size = 2
    morph_elem = cv.MORPH_RECT
    operation = cv.MORPH_BLACKHAT

    reverse = 255 - image

    # op_element = helper.get_structure_ele(cv.MORPH_RECT, 1)
    # reverse = cv.morphologyEx(reverse, cv.MORPH_CLOSE, op_element)
    # reverse = cv.dilate(reverse, op_element)
    # cv.imshow("reverse", reverse)

    arrows = get_seq_arrows(reverse)

    if split:
        element = helper.get_structure_ele(morph_elem, morph_size)
        morph = cv.morphologyEx(image, operation, element)
        # cv.imshow("morph", morph)
        # cv.imwrite("img_output/black_hat.png", morph)
    else:
        morph = 255 - image
    # show(morph, "morph")

    # 获取粗边框的元素的位置
    erosion_element = helper.get_structure_ele(morph_elem, 1)
    erosion = cv.erode(morph, erosion_element)
    # show(erosion, "erosion")

    erosion_gray = cv.cvtColor(erosion, cv.COLOR_BGR2GRAY)
    _, erosion_binary = cv.threshold(erosion_gray, 100, 255, cv.THRESH_BINARY)
    _, erosion_contours, erosion_hierarchy = cv.findContours(erosion_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 获取粗边框元素边界矩形
    partial_elements_rec = []
    for i, contour in enumerate(erosion_contours):
        contour_rec = helper.get_one_contour_rec(i, erosion_contours)
        if contour_rec[2] > cfg.CONTOUR_AREA_THRESHOLD and erosion_hierarchy[0][i][3] == -1:
            partial_elements_rec.append(contour_rec[0])

    # 获取细边框元素轮廓
    morph_gray = cv.cvtColor(morph, cv.COLOR_BGR2GRAY)
    _, morph_binary = cv.threshold(morph_gray, 50, 255, cv.THRESH_BINARY)
    _, morph_contours, morph_hierarchy = cv.findContours(morph_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return morph_contours, morph_hierarchy, partial_elements_rec, arrows


# remove the contours with small area
def get_contours_bt(contours, area_threshold, contour_list):
    return list(filter(lambda i: cv.contourArea(contours[i]) > area_threshold, contour_list))
    # return list(filter(lambda i: helper.get_one_contour_rec(i, contours)[2] > 2000, contour_list))


def divide_layers(contours, hierarchy):
    res = np.where(hierarchy[0, :, 3] == -1)
    layer = list(res[0])
    layer = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, layer)

    layer_dic = {}
    layer_count = -1
    while len(layer) > 0:
        layer_count += 1
        # get the contours in the next layer
        next_layer = []
        curr_layer = {}
        for c in layer:
            res = np.where(hierarchy[0, :, 3] == c)
            c_children = list(res[0])
            c_children = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, c_children)
            curr_layer[c] = c_children
            next_layer.extend(c_children)

        next_layer = get_contours_bt(contours, cfg.CONTOUR_AREA_THRESHOLD, next_layer)

        layer_dic[layer_count] = curr_layer
        layer = next_layer
    # {layer_num:{c_i:c_children}}
    return layer_dic


def get_contours_rec(contours, layers):
    contours_rec = [None] * len(contours)
    layers_num = len(layers)
    for i in range(layers_num):
        layer = layers[i].keys()
        for c_i in layer:
            contours_rec[c_i] = helper.get_one_contour_rec(c_i, contours)
    return contours_rec


def locate_objects(input_img):
    """检测二值图片中元素的轮廓及其bounding box"""

    # 检测轮廓
    _, contours, hierarchy = cv.findContours(input_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 获取轮廓bounding box
    objects = list(map(helper.get_one_contour_box, contours))

    return objects


def normalize_arrow(arrow):
    if arrow[2] < 8 <= arrow[3] or arrow[3] < 8 <= arrow[2] or (arrow[2] < 8 and arrow[3] < 8):
        center_x = arrow[0] + arrow[2] // 2
        center_y = arrow[1] + arrow[3] // 2
        x = max(0, center_x - 4)
        y = max(0, center_y - 4)

        arrow = (x, y, 8, 8)
    arrow = helper.dilate(arrow, 2)
    return arrow


def get_seq_arrows(input_img):
    """获取sequence flow的箭头 实心箭头"""
    # detect_base = 255 - input_img
    detect_base = input_img.copy()
    # cv.imshow("detect_base", detect_base)

    # Opening, kernel:2, element:ellipse
    open_struct = helper.get_structure_ele(cv.MORPH_ELLIPSE, 2)
    detect_base = cv.morphologyEx(detect_base, cv.MORPH_OPEN, open_struct)
    # cv.imshow("seq_opening", detect_base)

    # Threshold
    detect_base = cv.cvtColor(detect_base, cv.COLOR_BGR2GRAY)
    _, detect_base = cv.threshold(detect_base, 125, 255, cv.THRESH_BINARY)
    # cv.imshow("seq_open_binary", detect_base)

    seq_arrows = locate_objects(detect_base)

    # Filter
    seq_arrows = list(map(normalize_arrow, seq_arrows))

    # seq_arrows_img = helper.draw_rects(input_img, seq_arrows, cfg.COLOR_GREEN, 1)
    # cv.imshow("seq_arrows_img", seq_arrows_img)
    # cv.waitKey(0)
    return seq_arrows


def abs_cross(p1, p2, p3):
    """获取向量p1p2,与p1p3的叉积的绝对值"""
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p3[0] - p1[0], p3[1] - p1[1]]
    return abs(v1[0] * v2[1] - v2[0] * v1[1])


def get_bisector(seg1, seg2):
    """获取两等长相交线段的角平分线"""
    """坐标系：右x下y"""

    # 计算两条线段的交点
    l1 = points_to_line(seg1[0], seg1[1])
    l2 = points_to_line(seg2[0], seg2[1])

    # p = get_point_of_intersection(l1, l2)
    p = get_float_point_of_intersection([seg1[0][0], seg1[0][1], seg1[1][0], seg1[1][1]],
                                        [seg2[0][0], seg2[0][1], seg2[1][0], seg2[1][1]])

    if not (l1.p1[0] <= p[0] <= l1.p2[0] and l2.p1[0] <= p[0] <= l2.p2[0]):
        print("invalid segs")

    # 计算两线段向量的角平分线斜率
    v1 = Vector(seg1[0], seg1[1])
    v2 = Vector(seg2[0], seg2[1])

    v1 = np.array([v1.x, v1.y])
    v2 = np.array([v2.x, v2.y])

    length_seq = helper.get_points_dist_square(l1.p1, l1.p2)

    cos_angle = v1.dot(v2) / length_seq

    # 因为两个向量模相等，相加后得角平分线向量, 得到的是锐角的角平分线
    if cos_angle < 0:
        v3 = v1 + v2
    else:
        v3 = v1 - v2

    v4 = np.array([1, 0])

    cos_theta = v3.dot(v4) / np.sqrt(v3.dot(v3))

    if cos_theta == 0:
        k = None
        b = int(p[0])
        return k, b
    else:
        theta = np.arccos(cos_theta)
        k = np.tan(theta)
        b = p[1] - k * p[0]
        return k, b


# def is_border_point(point, width, height):
#     if point[0] == 0 or point[0] == width - 1 or point[1] == 0 or point[1] == height - 1:
#         return True
#     return False

def get_convex_hull_points(points):
    points = np.array(points)

    # 获取箭头的凸包
    hull = cv.convexHull(points)
    hull_points = list()
    for point in hull:
        p = list(point[0])
        hull_points.append(p)
    return hull_points


def get_diam_of_convex_hull(hull, width, height):
    """获取凸包的长轴(距离最远的两个点连成的线) 可用旋转卡壳算法，但点不多，不必要"""
    """输入:逆时针排序的凸包上的点"""
    """输出:线段的序列，每条线段的端点按横坐标，纵坐标排序"""

    length = 0
    diams = list()
    size = len(hull)

    for i in range(size):
        for j in range(i + 1, size):
            dist = helper.get_points_dist_square(hull[i], hull[j])
            if length < dist:
                length = dist
                diams = [[hull[i], hull[j]]]
            elif dist == length:
                diams.append([hull[i], hull[j]])

    for diam in diams:
        # print("diam")
        diam.sort(key=cmp_to_key(helper.points_cmp))
        # print(diam)
        # print("------")

    line = None
    if len(diams) == 1:
        line = points_to_line(diams[0][0], diams[0][1])
        # return line
    elif len(diams) == 2:
        k, b = get_bisector(diams[0], diams[1])
        if k is not None:
            p1 = (0, helper.get_func_value(0, k, b))
            p2 = (width, helper.get_func_value(width, k, b))
        else:
            p1 = (int(b), 0)
            p2 = (int(b), height)
        line = points_to_line(p1, p2)
    else:
        # for diam in diams:
        #     print(diam)
        print("凸包存在多条直径, 或凸包点数少于2")

    return line


def get_arrow_points(points, center, arrow):
    """通过聚类，去除噪声点即不属于箭头的点"""

    # if filter:
    real_points = []
    for p in points:
        if (p[1] == 0 or p[1] == arrow[3] - 1) and \
                (0 < p[0] < center[0] - 1 or center[0] + 1 < p[0] < arrow[2] - 1) \
                or \
                (p[0] == 0 or p[0] == arrow[2] - 1) and \
                (0 < p[1] < center[1] - 1 or center[1] + 1 < p[1] < arrow[1] - 1):
            pass
        else:
            real_points.append(p)
    points = real_points

    arrow_points = []
    curr_cores = [center]
    next_cores = []
    rest_points = []

    while len(curr_cores) > 0:
        for p in points:
            is_valid = False
            for core in curr_cores:
                if helper.get_points_dist_square(core, p) < 1.1:
                    next_cores.append(p)
                    is_valid = True
                    break
            if not is_valid:
                rest_points.append(p)

        arrow_points.extend(next_cores)
        curr_cores = next_cores
        next_cores = []
        points = rest_points
        rest_points = []

    return arrow_points


def get_seq_arrow_direction(binary_input, arrow, input_img):
    """从箭头图片判断箭头方向"""
    # print(arrow)

    width = arrow[2]
    height = arrow[3]
    center = [width // 2, height // 2]
    arrow_img = helper.truncate(binary_input, arrow)

    # a_img = helper.truncate(input_img, arrow)

    # cv.imshow("arrow_img", helper.dilate_drawing(arrow_img))
    # cv.waitKey(0)

    # 获取箭头所有点的坐标
    # 坐标系：右x下y
    points = []
    vertical_lines = []
    horizontal_lines = []

    for i in range(width):
        is_possible_diam = True
        count = 0
        for j in range(height):
            if arrow_img[j][i] > 125:
                points.append([i, j])
            else:
                count += 1
                if count > 1:
                    is_possible_diam = False
        if is_possible_diam and center[0] - 3 < i < center[0] + 3:
            vertical_lines.append(i)

    for j in range(height):
        is_possible_diam = True
        count = 0
        for i in range(width):
            if arrow_img[j][i] <= 125:
                count += 1
                if count > 1:
                    is_possible_diam = False
                    break
        if is_possible_diam and center[1] - 3 < j < center[1] + 3:
            horizontal_lines.append(j)

    points_copy = points.copy()
    # points = np.array(points)
    # #
    # # 坐标轴翻转
    # ax = plt.gca()
    # ax.xaxis.set_ticks_position("top")
    #
    # # 设置网格线
    # plt.ylim(-1, height)
    # plt.xlim(-1, width)
    # maloc = plt.MultipleLocator(2)
    # miloc = plt.MultipleLocator(1)
    # ax.xaxis.set_major_locator(maloc)
    # ax.yaxis.set_major_locator(maloc)
    # ax.xaxis.set_minor_locator(miloc)
    # ax.yaxis.set_minor_locator(miloc)
    # ax.grid(which="minor", axis="x", linewidth=1)
    # ax.grid(which="minor", axis="y", linewidth=1)
    # ax.invert_yaxis()
    #
    # # 坐标轴刻度的字体
    # plt.xticks(size=16)
    # plt.yticks(size=16)
    #
    # plt.scatter(points[:, 0], points[:, 1], c="black", s=120)
    # plt.show()

    core_points = []
    for p in points_copy:
        all_in = True
        for i in range(-1, 2):
            for j in range(-1, 2):
                if [p[0] + i, p[1] + j] not in points_copy:
                    all_in = False
                    break
        if all_in:
            core_points.append(p)

    # core_points = np.array(core_points)
    # plt.ylim(-1, height)
    # plt.xlim(-1, width)
    # plt.scatter(core_points[:, 0], core_points[:, 1])
    # plt.show()

    diam = None
    if len(vertical_lines) > 1 >= len(horizontal_lines):
        # if max(vertical_lines) - min(vertical_lines)
        mean_v_line = np.mean(vertical_lines)
        if center[0] - 2 < mean_v_line < center[0] + 2:
            x_value = int(mean_v_line)
            diam = points_to_line([x_value, 0], [x_value, height - 1])
        else:
            for v_line in vertical_lines:
                for j in range(height):
                    points.remove([v_line, j])
    elif len(vertical_lines) <= 1 < len(horizontal_lines):
        mean_h_line = np.mean(horizontal_lines)
        if center[1] - 2 < mean_h_line < center[0] + 2:
            y_value = int(mean_h_line)
            diam = points_to_line([0, y_value], [width - 1, y_value])
        # else:
    elif len(vertical_lines) > 1 and len(horizontal_lines) > 1:
        mean_v_line = np.mean(vertical_lines)
        mean_h_line = np.mean(horizontal_lines)
        v_center_dist = abs(center[0] - mean_v_line)
        h_center_dist = abs(center[1] - mean_h_line)

        if v_center_dist < h_center_dist and v_center_dist < 2:
            x_value = int(mean_v_line)
            diam = points_to_line([x_value, 0], [x_value, height - 1])
        elif h_center_dist < v_center_dist and h_center_dist < 2:
            y_value = int(mean_h_line)
            diam = points_to_line([0, y_value], [width - 1, y_value])

    # arrow_points = get_arrow_points(core_points, center, arrow)
    hull_points = get_convex_hull_points(core_points)

    if diam is None:
        # 获取箭头的轴线，即凸包中最远的两个点
        diam = get_diam_of_convex_hull(hull_points, width, height)

    if diam is None:
        print("can't find valid diam")
        # cv.waitKey(0)
        return None

    # 获取箭头前端三角形的底
    pos_max = -1
    pos_polar_points = []
    neg_min = 1
    neg_polar_points = []
    for p in core_points:
        if diam.k is None:
            val = p[0] - diam.b
        else:
            val = (p[1] - diam.k * p[0] - diam.b) / np.sqrt(1 + diam.k ** 2)

        if val > 0:
            if val > pos_max:
                pos_max = val
                pos_polar_points = [[p[0], p[1]]]
            elif val == pos_max:
                pos_polar_points.append([p[0], p[1]])
        else:
            if val < neg_min:
                neg_min = val
                neg_polar_points = [[p[0], p[1]]]
            elif val == neg_min:
                neg_polar_points.append([p[0], p[1]])

    if len(pos_polar_points) == 0 or len(neg_polar_points) == 0:
        print("invalid diam. The diam can't divide the arrow")
        return None

    pos_pts = np.array(pos_polar_points)
    neg_pts = np.array(neg_polar_points)

    pos_polar_point = [int(np.mean(pos_pts[:, 0])), int(np.mean(pos_pts[:, 1]))]
    neg_polar_point = [int(np.mean(neg_pts[:, 0])), int(np.mean(neg_pts[:, 1]))]

    line_cut = points_to_line(pos_polar_point, neg_polar_point)

    # 箭头前端三角的底与轴线的交点
    float_cut_point = get_float_point_of_intersection(diam.li, line_cut.li)
    diam_center = [np.mean([diam.li[0], diam.li[2]]), np.mean([diam.li[1], diam.li[3]])]

    # cv.line(a_img, diam.p1, diam.p2, cfg.COLOR_RED, 1)
    # cv.imshow("arrow", helper.dilate_drawing(a_img))
    #
    real_p1 = [diam.p1[0] + arrow[0], diam.p1[1] + arrow[1]]
    real_p2 = [diam.p2[0] + arrow[0], diam.p2[1] + arrow[1]]

    diam = points_to_line(real_p1, real_p2)

    if diam.k is None:
        if float_cut_point[1] < diam_center[1]:
            direction = cfg.TOP
        else:
            direction = cfg.DOWN
        info = [diam.p1[1], diam.p2[1], direction]
    else:
        if float_cut_point[0] < diam_center[0]:
            direction = cfg.LEFT
        else:
            direction = cfg.RIGHT
        info = [diam.p1[0], diam.p2[0], direction]
    # print(info, cfg.DIRECTIONS[info[-1]])
    arrow_line = {(diam.k, diam.b): info}

    # print(arrow_line)
    # cv.waitKey(0)
    return arrow_line


def get_msg_arrows(input_img, seq_arrows):
    """获取message flows的箭头 空心箭头"""

    black_mask = np.zeros_like(input_img)

    detect_base = 255 - input_img
    # detect_base = input_img.copy()

    for seq_arrow in seq_arrows:
        detect_base = helper.mask(detect_base, black_mask, seq_arrow)

    # Closing 2, ellipse
    closing_struct = helper.get_structure_ele(cv.MORPH_ELLIPSE, 2)
    detect_base = cv.morphologyEx(detect_base, cv.MORPH_CLOSE, closing_struct)
    # cv.imshow("close", detect_base)

    # Opening 3, ellipse
    opening_struct = helper.get_structure_ele(cv.MORPH_ELLIPSE, 3)
    detect_base = cv.morphologyEx(detect_base, cv.MORPH_ELLIPSE, opening_struct)
    # cv.imshow("open", detect_base)

    # Threshold
    detect_base = cv.cvtColor(detect_base, cv.COLOR_BGR2GRAY)
    _, detect_base = cv.threshold(detect_base, 140, 255, cv.THRESH_BINARY)
    # cv.imshow("binary", detect_base)

    msg_arrows = locate_objects(detect_base)
    #
    # for msg_arrow in msg_arrows:
    #     print(msg_arrow)
    #
    msg_arrows_img = helper.draw_rects(input_img, msg_arrows, cfg.COLOR_GREEN, 1)
    cv.imshow("msg_arrows_img", msg_arrows_img)
    cv.waitKey(0)
    return msg_arrows


def threshold_one_dim(one_dim):
    # cv.imshow("one_dim", one_dim)
    # _, dst = cv.threshold(one_dim, 100, 255, cv.THRESH_TOZERO)
    # cv.imshow("dst", dst)
    _, final = cv.threshold(one_dim, 210, 255, cv.THRESH_BINARY)
    # cv.imshow("final", final)

    op_element = helper.get_structure_ele(cv.MORPH_RECT, 3)
    dilation = cv.dilate(final, op_element)
    # cv.imshow("one_dim_dilation", dilation)
    final = final - dilation + 255
    # cv.imshow("final_op", final)
    return final


def convert_to_black_white(input_img):
    """将彩色图片变为黑白图片"""
    # cv.imshow("input", input_img)
    b = input_img[:, :, 0]
    g = input_img[:, :, 1]
    r = input_img[:, :, 2]

    input_copy = input_img.copy()
    b = threshold_one_dim(b)
    # cv.waitKey(0)
    g = threshold_one_dim(g)
    # cv.waitKey(0)
    r = threshold_one_dim(r)
    # cv.waitKey(0)
    input_copy[:, :, 0] = b
    input_copy[:, :, 1] = g
    input_copy[:, :, 2] = r

    for i in range(input_copy.shape[0]):
        for j in range(input_copy.shape[1]):
            a = input_copy[i, j, :]
            if a[0] > 0 or a[1] > 0 or a[2] > 0:
                input_copy[i, j, :] = [255, 255, 255]

            black = input_img[i, j, :]
            if black[0] == 0 or black[1] == 0 or black[2] == 0:
                input_copy[i, j, :] = black

    # op_element = helper.get_structure_ele(cv.MORPH_RECT, 1)
    # input_copy = cv.erode(input_copy, op_element)

    return input_copy


def pre_process(file_path, split=True):
    raw_img = cv.imread(file_path)
    # cv.imshow("raw_img", raw_img)
    if file_path.endswith(".jpeg"):
        project_dir = file_path[:file_path.rindex("/")]
        project = project_dir[project_dir.rindex("/") + 1:]
        convert_img_file = "{}/jpg_convert/{}.jpeg".format(project_dir, project)
        # print(convert_img_file)
        input_img = cv.imread(convert_img_file)
    elif file_path.endswith(".png"):
        input_img = raw_img.copy()
        # _, input_img = cv.threshold(raw_img, 254, 255, cv.THRESH_BINARY)
        # cv.imshow("input_img", input_img)
    else:
        input_img = 255 - raw_img
        # op_element = helper.get_structure_ele(cv.MORPH_CROSS, 1)
        # input_img = cv.morphologyEx(input_img, cv.MORPH_TOPHAT, op_element)
        # input_img = cv.erode(input_img, op_element)
        cv.imshow("input_img", input_img)
        cv.waitKey(0)
    contours, hierarchy, partial_elements, arrows = get_contours(input_img, split)
    layers = divide_layers(contours, hierarchy)
    contours_rec = get_contours_rec(contours, layers)
    return raw_img, input_img, layers, contours, contours_rec, partial_elements, arrows


def convert_jpg_projects():
    # root_dir = "E:/master/data_1031/merge_info_validate"
    root_dir = "gen_my_data_jpg/projects"
    projects = os.listdir(root_dir)
    for project in projects:
        print(project)
        project_dir = "{}/{}".format(root_dir, project)
        files = os.listdir(project_dir)
        jpg_file = ""
        for file in files:
            if file.endswith("jpeg"):
                jpg_file = "{}/{}".format(project_dir, file)
                break
        # root_dir = "gen_my_data_jpg/imgs"
        # imgs = os.listdir(root_dir)
        # for img in imgs[::2]:
        #     jpg_file = "{}/{}".format(root_dir, img)
        input_img = cv.imread(jpg_file)
        input_img = convert_to_black_white(input_img)
        convert_dir = "{}/{}".format(project_dir, "jpg_convert")
        if not os.path.exists(convert_dir):
            os.mkdir(convert_dir)
        convert_img_file = "{}/{}.jpeg".format(convert_dir, project)

        # cv.imshow("convert", input_img)
        # cv.waitKey(0)
        cv.imwrite(convert_img_file, input_img)


def main():
    convert_jpg_projects()

    # sample_dir = "gen_my_data_jpg/imgs/"
    # images = os.listdir(sample_dir)
    #
    # selected = images[1:2]
    # for im in selected:
    #     file_path = sample_dir + im
    #     if os.path.isfile(file_path):
    #         # pass
    #         get_layers_img(file_path, reverse=False)
    #         # _, input_img, _, _, _, _, _ = pre_process(file_path)
    #         # cv.imshow("input", input_img)
    #         # # input_img = cv.imread(file_path)
    #         # # convert_to_black_white(input_img)
    #     cv.waitKey(0)


if __name__ == '__main__':
    main()
