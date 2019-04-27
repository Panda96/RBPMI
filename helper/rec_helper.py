# -*- coding:utf-8 -*-
import cv2 as cv


def draw_one_rect(draw_img, bound_rect, color, thickness):
    cv.rectangle(draw_img, (int(bound_rect[0]), int(bound_rect[1])),
                 (int(bound_rect[0] + bound_rect[2]), int(bound_rect[1] + bound_rect[3])), color,
                 thickness)

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
