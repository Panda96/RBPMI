# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import time


def print_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


def get_structure_ele(morph_elem, morph_size):
    return cv.getStructuringElement(morph_elem, (2 * morph_size + 1, 2 * morph_size + 1), (morph_size, morph_size))


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
        # put the text in the middle of a rectangle and not beyond the border of the image
        org_x = bound_rect[0] + (bound_rect[2] - text_size[0][0]) // 2
        org_x = max(org_x, 2)
        org_x = min(org_x, draw_img.shape[1] - text_size[0][0] - 5)
        cv.putText(draw_img, text, (org_x, bound_rect[1] + (bound_rect[3] + text_size[0][1]) // 2),
                   cv.QT_FONT_BLACK, 0.3, (255, 255, 255))

    return draw_img


def dilate(rect, dilation_value):
    # may be I need to consider whether if the rect will beyond the border of the image after dilation
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
    return rec1[0] <= rec2[0] and rec1[1] <= rec2[1] \
           and rec1[0] + rec1[2] >= rec2[0] + rec2[2] and rec1[1] + rec1[3] >= rec2[1] + rec2[3]


def point_is_in(rec, point):
    return is_in(rec, [point[0], point[1], 0, 0])


def truncate(base, roi_rec):
    if len(base.shape) == 3:
        return base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
               max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1]), :]
    elif len(base.shape) == 2:
        return base[max(0, roi_rec[1]): min(roi_rec[1] + roi_rec[3], base.shape[0]),
               max(0, roi_rec[0]): min(roi_rec[0] + roi_rec[2], base.shape[1])]
    else:
        print("The image's channels are not 2 or 3.")


# place rec2 at the center of rec1
def get_center_position(rec1, rec2_shape):
    x_begin = rec1[0] + (rec1[2] - rec2_shape[1]) // 2
    y_begin = rec1[1] + (rec1[3] - rec2_shape[0]) // 2

    return x_begin, y_begin


def dilate_drawing(drawing):
    drawing_shape = drawing.shape
    if drawing_shape[0] > 300 and drawing_shape[1] > 300:
        return drawing
    else:
        height = max(300, drawing_shape[0]) + 20
        width = max(300, drawing_shape[1]) + 20
        (x_begin, y_begin) = get_center_position((0, 0, width, height), drawing_shape)
        if len(drawing_shape) == 2:
            base = np.zeros((height, width), dtype=np.uint8)
            base = 123 + base
            base[y_begin:y_begin + drawing_shape[0], x_begin:x_begin + drawing_shape[1]] = drawing
        elif len(drawing_shape) == 3:
            base = np.zeros((height, width, 3), dtype=np.uint8)
            base = 123 + base
            base[y_begin:y_begin + drawing_shape[0], x_begin:x_begin + drawing_shape[1], :] = drawing
        return base
