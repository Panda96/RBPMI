# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import time


def points_cmp(p1, p2):
    if p1[0] < p2[0]:
        return -1
    elif p1[0] > p2[0]:
        return 1
    else:
        if p1[1] < p2[1]:
            return -1
        elif p1[1] > p2[1]:
            return 1
        else:
            return 0


def get_func_value(x, k, b):
    return int(k * x + b)


def get_points_dist_square(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def get_points_dist(p1, p2):
    dist_sq = get_points_dist_square(p1, p2)
    return dist_sq ** 0.5


def print_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


def get_structure_ele(morph_elem, morph_size):
    return cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))


def get_one_contour_box(contour):
    contour_poly = cv.approxPolyDP(contour, 3, True)
    bound_rect = cv.boundingRect(contour_poly)
    return bound_rect


def get_one_contour_rec(contour_index, contours_list):
    contour_poly = cv.approxPolyDP(contours_list[contour_index], 3, True)
    bound_rect = cv.boundingRect(contour_poly)
    contour_area = cv.contourArea(contours_list[contour_index])
    rec_area = bound_rect[2] * bound_rect[3]
    # bound_rect (x, y, width, height)(x,y) is the coordinate of the left-top point
    # bound_rect = [int(x) for x in bound_rect]
    bound = (bound_rect, contour_area, rec_area)
    return bound


def draw_one_rect(draw_img, bound_rect, color, thickness, show_text=False):
    cv.rectangle(draw_img, (int(bound_rect[0]), int(bound_rect[1])),
                 (int(bound_rect[0] + bound_rect[2]), int(bound_rect[1] + bound_rect[3])), color,
                 thickness)
    if show_text:
        text = "{}".format(bound_rect[2] * bound_rect[3])
        text_size = cv.getTextSize(text, cv.QT_FONT_NORMAL, 0.3, 1)
        # put the text in the middle of a rectangle and not beyond the border of the image_mat
        org_x = bound_rect[0] + (bound_rect[2] - text_size[0][0]) // 2
        org_x = max(org_x, 2)
        org_x = min(org_x, draw_img.shape[1] - text_size[0][0] - 5)
        cv.putText(draw_img, text, (org_x, bound_rect[1] + (bound_rect[3] + text_size[0][1]) // 2),
                   cv.QT_FONT_BLACK, 0.3, (255, 255, 255))

    return draw_img


def draw_rects(draw_img, rects, color, thickness, show_text=False):
    base_img = draw_img.copy()
    for rect in rects:
        base_img = draw_one_rect(base_img, rect, color, thickness, show_text)
    return base_img


def draw_lines(draw_img, line_list, color, thickness):
    base_img = np.copy(draw_img)
    for line in line_list:
        cv.line(base_img, line.p1, line.p2, color, thickness, cv.LINE_AA)
    return base_img


def dilate(rect, dilation_value):
    # may be I need to consider whether if the rect will beyond the border of the image_mat after dilation
    x = max(0, rect[0] - dilation_value)
    y = max(0, rect[1] - dilation_value)
    width = rect[2] + 2 * dilation_value
    height = rect[3] + 2 * dilation_value
    rect = (x, y, width, height)
    return rect


def shrink(rect, shrink_value):
    x = rect[0] + shrink_value
    y = rect[1] + shrink_value
    width = rect[2] - 2 * shrink_value
    height = rect[3] - 2 * shrink_value
    rect = (x, y, width, height)
    return rect


def is_adjacent(rect1, rect2, dilation_value=5):
    rect1_dilation = dilate(rect1, dilation_value)
    return (not is_overlap(rect1, rect2)) and is_overlap(rect1_dilation, rect2)


def is_overlap(rec1, rec2):
    return rec1[1] + rec1[3] > rec2[1] and rec1[1] < rec2[1] + rec2[3] \
           and rec1[0] + rec1[2] > rec2[0] and rec1[0] < rec2[0] + rec2[2]


def get_rect_area(rect):
    return rect[2] * rect[3]


def get_overlap_area(rec1, rec2):
    if is_overlap(rec1, rec2):
        p1_x = max(rec1[0], rec2[0])
        p1_y = max(rec1[1], rec2[1])
        # p2 bottom down point of overlap area-
        p2_x = min(rec1[0] + rec1[2], rec2[0] + rec2[2])
        p2_y = min(rec1[1] + rec1[3], rec2[1] + rec2[3])

        overlap_area = (p2_x - p1_x) * (p2_y - p1_y)
        return overlap_area
    else:
        return -1


# rec2 is in rec1
def is_in(rec1, rec2):
    """判断 rec2 是否在 rec1 里"""
    return rec1[0] <= rec2[0] and rec1[1] <= rec2[1] \
           and rec1[0] + rec1[2] >= rec2[0] + rec2[2] and rec1[1] + rec1[3] >= rec2[1] + rec2[3]


def point_is_in(rec, point):
    return is_in(rec, [point[0], point[1], 0, 0])


def get_rect_vertices(rec):
    # the four vertices of a rectangle in clockwise turn
    points = list()
    points.append((rec[0], rec[1]))
    points.append((rec[0] + rec[2], rec[1]))
    points.append((rec[0] + rec[2], rec[1] + rec[3]))
    points.append((rec[0], rec[1] + rec[3]))
    return points


def get_rec_center(rec):
    rec_center = (rec[0] + rec[2] // 2, rec[1] + rec[3] // 2)
    return rec_center


def truncate(base, roi_rec):
    """从图片中截取Region of Interest"""
    if len(base.shape) == 3:
        return base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
               max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1]), :]
    elif len(base.shape) == 2:
        return base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
               max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1])]
    else:
        print("The image_mat's channels are not 2 or 3.")


def mask(base, mask, roi_rec):
    """替换图片中的 ROI 为相应mask中的内容"""

    if len(base.shape) == 3:
        base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
        max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1]), :] = \
            mask[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
            max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1]), :]
        return base
    elif len(base.shape) == 2:
        base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
        max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1])] = \
            mask[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
            max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1])]
        return base
    else:
        print("The image_mat's channels are not 2 or 3.")


def rotate_90_clockwise(image):
    rotated_img = cv.transpose(image)
    rotated_img = cv.flip(rotated_img, 1)
    return rotated_img


# place rec2 at the center of rec1
def get_center_position(rec1, rec2_shape):
    x_begin = rec1[0] + (rec1[2] - rec2_shape[1]) // 2
    y_begin = rec1[1] + (rec1[3] - rec2_shape[0]) // 2

    return x_begin, y_begin


def dilate_drawing(drawing, background=0):
    """确保图片的最小尺寸为300*300, 图片在窗口中居中显式"""
    drawing_shape = drawing.shape
    if drawing_shape[0] > 300 and drawing_shape[1] > 300:
        return drawing
    else:
        height = max(300, drawing_shape[0]) + 20
        width = max(300, drawing_shape[1]) + 20
        (x_begin, y_begin) = get_center_position((0, 0, width, height), drawing_shape)
        if len(drawing_shape) == 2:
            base = np.zeros((height, width), dtype=np.uint8)
            base = background + base
            base[y_begin:y_begin + drawing_shape[0], x_begin:x_begin + drawing_shape[1]] = drawing
        elif len(drawing_shape) == 3:
            base = np.zeros((height, width, 3), dtype=np.uint8)
            base = background + base
            base[y_begin:y_begin + drawing_shape[0], x_begin:x_begin + drawing_shape[1], :] = drawing
        return base
