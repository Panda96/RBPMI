# -*- coding:utf-8 -*-
import cfg
import numpy as np
import cv2 as cv
from helper import detector_helper as helper
from collections import defaultdict


def draw_pools(pools_list, input_img):
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
                    drawing = helper.draw_one_rect(drawing, element, cfg.COLOR_BLUE, cfg.CONTOUR_THICKNESS, True)

    return drawing


def connected_by_arrows(pool_rec, arrows):
    connected = False
    for arrow in arrows:
        dilate_arrow = helper.dilate(arrow, 10)
        if not helper.is_in(pool_rec, dilate_arrow) and helper.is_overlap(pool_rec, dilate_arrow):
            connected = True
            break
    return connected


def get_potential_pools(layer_0, contours_rec, layers, arrows):
    potential_pools = []
    for c_i in layer_0:
        bound_i = contours_rec[c_i]
        is_pool = False
        if bound_i[1] > cfg.POOL_AREA_THRESHOLD:
            # print(len(layers[0][c_i]))
            is_pool = True
        elif bound_i[1] > 20000 and len(layers[0][c_i]) <= 2 and bound_i[1] / bound_i[2] > 0.8:
            print("detect small pool")
            is_pool = True

        if is_pool and not connected_by_arrows(bound_i[0], arrows):
            potential_pools.append(c_i)
    return potential_pools


def get_pools_in_one_rec(c_children, contours_rec, pool_bound, c_i):
    pools_in_one_rec = []


    # 判断排好序的子轮廓列表中的第一个轮廓是否是pool_header
    contour_bound = contours_rec.pop(0)

    [contour_x, contour_y, contour_width, contour_height] = contour_bound[0]
    [pool_x, pool_y, pool_width, pool_height] = pool_bound[0]
    pool_top_left = helper.dilate([pool_x, pool_y, 0, 0], 3)

    is_header = False
    if helper.point_is_in(pool_top_left, [contour_x, contour_y]):
        if 15 <= contour_width <= 30 and pool_height - 10 < contour_height < pool_height + 10:
            is_header = True




    # children_rest = c_children.copy()
    # for child_id in c_children:
    #     # print(child_id)
    #     # get normal pools
    #     if child_id in children_rest:
    #         child_bound = contours_rec[child_id]
    #
    #         if is_pool_header(child_bound):
    #             # print("header:{}".format(child_bound))
    #             # found a header means found a pool
    #             pool = dict()
    #             # remove header contour id
    #             children_rest.remove(child_id)
    #             # a rect with appropriate height-to-width ratio is very likely to be a header bounding box
    #             header_rect = child_bound[0]
    #             pool["header_rect"] = header_rect
    #
    #             pool_rect = (header_rect[0], header_rect[1], pool_bound[0][2], header_rect[3])
    #             pool["rect"] = pool_rect
    #
    #             lane_width = pool_bound[0][2] - header_rect[2]
    #             pool_lanes_rect = [header_rect[0] + header_rect[2], header_rect[1], lane_width,
    #                                header_rect[3]]
    #             pool["lanes_rect"] = pool_lanes_rect
    #
    #             pool_lanes = []
    #
    #             lane_y_begin = pool_lanes_rect[1]
    #             while lane_y_begin < pool_lanes_rect[1] + pool_lanes_rect[3] - 20:
    #                 # print(lane_y_begin)
    #                 # 按水平方向寻找lane
    #                 next_y_begin = lane_y_begin
    #                 for c_id in children_rest:
    #                     lane_seg_bound = contours_rec[c_id]
    #                     lane_seg_rec = lane_seg_bound[0]
    #                     # print(lane_seg_rec)
    #                     if helper.is_in(pool_lanes_rect, lane_seg_rec) and \
    #                             lane_y_begin <= lane_seg_rec[1] < lane_y_begin + 5:
    #                         # one_lane_seg.append(c_id)
    #                         # print(lane_seg_rec[1])
    #                         if lane_seg_rec[2] > 32:
    #                             seg_y_end = lane_seg_rec[1] + lane_seg_rec[3]
    #                             if seg_y_end > next_y_begin:
    #                                 next_y_begin = seg_y_end
    #                         # else:
    #                 if next_y_begin == lane_y_begin:
    #                     break
    #                 # print("next:",next_y_begin)
    #                 lane = [pool_lanes_rect[0], lane_y_begin, pool_lanes_rect[2],
    #                         next_y_begin - lane_y_begin]
    #                 # print(lane)
    #                 pool_lanes.append(lane)
    #
    #                 one_lane_seg = []
    #                 for c_id in children_rest:
    #                     lane_seg_rec = contours_rec[c_id][0]
    #                     if helper.is_overlap(lane, lane_seg_rec):
    #                         one_lane_seg.append(c_id)
    #
    #                 for lane_seg in one_lane_seg:
    #                     children_rest.remove(lane_seg)
    #                 lane_y_begin = next_y_begin
    #             if len(pool_lanes) > 0:
    #                 pool["lanes"] = pool_lanes
    #                 pools_in_one_rec.append(pool)
    #
    # if len(pools_in_one_rec) == 0:
    #     print("has one pool only lanes")
    #     # if len(c_children) > 1:
    #     default_pool = create_default_pool([c_i], contours_rec, 0, layer1=c_children)
    #     pools_in_one_rec.append(default_pool)
    #
    # else:
    #     for c_id in children_rest:
    #         child_bound = contours_rec[c_id]
    #         if child_bound[1] > cfg.POOL_AREA_THRESHOLD:
    #             print("has blank pool")
    #             blank_pool = create_default_pool([c_id], contours_rec, 0)
    #             pools_in_one_rec.append(blank_pool)
    return pools_in_one_rec


def get_pools(layers, contours_rec, partial_elements, arrows):
    layer_0 = layers[0].keys()
    potential_pools = get_potential_pools(layer_0, contours_rec, layers, arrows)

    # print(potential_pools)

    pools_list = []

    pools_num = len(potential_pools)
    print("potential_pools:{}".format(pools_num))
    if pools_num == 0:
        # no lanes and no pools, just one process
        print("only elements")
        default_pool = create_default_pool(layer_0, contours_rec, 5)
        pools_list.append(default_pool)
        tag = 0

    else:
        print("has pools or lanes")
        for c_i in potential_pools:
            pool_bound = contours_rec[c_i]
            # print("pool_bound:{}".format(pool_bound))

            c_children = layers[0][c_i]

            if len(c_children) <= 1:
                print("potential pool has no or only one child")
            else:
                children_recs = [[child_id, contours_rec[child_id]] for child_id in c_children]
                # Sort the child contours by x-coordinate and y-coordinate
                children_recs.sort(key=lambda x: (x[1][0][0], x[1][0][1]))
                # print(children_recs)

                pools_in_one_rec = []
                children_rest = c_children.copy()
                for child_id in c_children:
                    # print(child_id)
                    # get normal pools
                    if child_id in children_rest:
                        child_bound = contours_rec[child_id]

                        if is_pool_header(child_bound):
                            # print("header:{}".format(child_bound))
                            # found a header means found a pool
                            # a rect with appropriate height-to-width ratio is very likely to be a header bounding box
                            pool = dict()
                            pool_rect = pool_bound[0]
                            # remove header contour id
                            children_rest.remove(child_id)

                            header_rect = child_bound[0]
                            pool["header_rect"] = header_rect

                            pool_width = pool_rect[2] + pool_rect[0] - header_rect[0]
                            pool_rect = (header_rect[0], header_rect[1], pool_width, header_rect[3])
                            pool["rect"] = pool_rect

                            lane_width = pool_width - header_rect[2]
                            pool_lanes_rect = [header_rect[0] + header_rect[2], header_rect[1], lane_width,
                                               header_rect[3]]
                            pool["lanes_rect"] = pool_lanes_rect

                            pool_lanes = []
                            lane_y_begin = pool_lanes_rect[1]
                            last_y_begin = lane_y_begin
                            next_y_begin = lane_y_begin
                            while lane_y_begin < pool_lanes_rect[1] + pool_lanes_rect[3] - 20:
                                # 按水平方向寻找lane

                                for c_id in children_rest:
                                    # print(next_y_begin)
                                    lane_seg_bound = contours_rec[c_id]
                                    lane_seg_rec = lane_seg_bound[0]

                                    if helper.is_in(helper.dilate(pool_lanes_rect, 5), lane_seg_rec) and \
                                            lane_y_begin - 5 <= lane_seg_rec[1] < lane_y_begin + 5:
                                        # one_lane_seg.append(c_id)
                                        if lane_seg_rec[2] > 35:
                                            seg_y_end = lane_seg_rec[1] + lane_seg_rec[3]
                                            if seg_y_end > next_y_begin:
                                                next_y_begin = seg_y_end
                                        # else:
                                if next_y_begin <= last_y_begin:
                                    break
                                else:
                                    last_y_begin = lane_y_begin
                                    lane_y_begin = next_y_begin
                                lane = [pool_lanes_rect[0], last_y_begin, pool_lanes_rect[2],
                                        lane_y_begin - last_y_begin]
                                # print(lane)
                                pool_lanes.append(lane)

                                one_lane_seg = []
                                for c_id in children_rest:
                                    lane_seg_rec = contours_rec[c_id][0]
                                    if helper.is_in(helper.dilate(lane, 5), lane_seg_rec):
                                        one_lane_seg.append(c_id)

                                for lane_seg in one_lane_seg:
                                    children_rest.remove(lane_seg)

                            if len(pool_lanes) > 0:
                                pool["lanes"] = pool_lanes
                                pools_in_one_rec.append(pool)

                if len(pools_in_one_rec) == 0:
                    print("has one pool only lanes")
                    # if len(c_children) > 1:
                    default_pool = create_default_pool([c_i], contours_rec, 0, layer1=c_children)
                    pools_in_one_rec.append(default_pool)

                else:
                    for c_id in children_rest:
                        child_bound = contours_rec[c_id]
                        if child_bound[1] > cfg.POOL_AREA_THRESHOLD:
                            print("has blank pool")
                            blank_pool = create_default_pool([c_id], contours_rec, 0)
                            pools_in_one_rec.append(blank_pool)

                pools_list.extend(pools_in_one_rec)
        # print("pools_object:", len(pools_object))
        if len(pools_list) == 0:
            print("only elements")
            default_pool = create_default_pool(layer_0, contours_rec, 5)
            pools_list.append(default_pool)
            tag = 0

        else:
            tag = 1
        # pools_img = draw_pools(pools_object, input_img)
        # # cv.namedWindow("pools", cv.WINDOW_NORMAL)
        # cv.imshow("pools", pools_img)
        # cv.waitKey(0)

    for pool in pools_list:
        pool["sub_procs"] = defaultdict(list)
        pool["elements"] = defaultdict(list)

    # pools_img = draw_pools(pools_object, input_img)
    # # # cv.namedWindow("pools", cv.WINDOW_NORMAL)
    # cv.imshow("pools", pools_img)
    # cv.waitKey(0)
    return pools_list, tag, partial_elements


def is_pool_header(bound):
    bound_rect = bound[0]

    if 20 <= bound_rect[2] <= 40:
        ratio = bound_rect[3] / bound_rect[2]
        # print(ratio)
        if cfg.POOL_HEADER_H_W_RATIO_FLOOR < ratio < cfg.POOL_HEADER_H_W_RATIO_CEILING:
            return True
    return False


def create_default_pool(layer0, contours_rec, pool_dilate_value, layer1=None):
    print("create_default_pool")
    all_left_top_x = map(lambda c: contours_rec[c][0][0], layer0)
    all_left_top_y = map(lambda c: contours_rec[c][0][1], layer0)
    all_right_bottom_x = map(lambda c: contours_rec[c][0][0] + contours_rec[c][0][2], layer0)
    all_right_bottom_y = map(lambda c: contours_rec[c][0][1] + contours_rec[c][0][3], layer0)

    left_top_x = min(all_left_top_x) - pool_dilate_value
    left_top_y = min(all_left_top_y) - pool_dilate_value
    right_bottom_x = max(all_right_bottom_x) + pool_dilate_value
    right_bottom_y = max(all_right_bottom_y) + pool_dilate_value

    # print([[left_top_x, left_top_y], [right_bottom_x, right_bottom_y]])

    x = left_top_x - cfg.DEFAULT_POOL_HEADER_WIDTH
    width = right_bottom_x - x
    height = right_bottom_y - left_top_y
    pool_rect = (x, left_top_y, width, height)
    header_rect = (x, left_top_y, cfg.DEFAULT_POOL_HEADER_WIDTH, height)
    lane_width = right_bottom_x - left_top_x
    pool_lanes_rect = (left_top_x, left_top_y, lane_width, height)

    pool_lanes = []
    if layer1 is None:
        pool_lanes = [(left_top_x, left_top_y, lane_width, height)]
    else:
        recs = [contours_rec[c_id][0] for c_id in layer1]
        potential_lanes = defaultdict(list)
        for rec in recs:
            potential_lanes[rec[1]].append(rec)

        lane_keys = list(potential_lanes.keys())
        lane_keys.sort()
        next_lane_begin = lane_keys[0]
        for k in lane_keys:
            if k >= next_lane_begin:
                lane_segs = potential_lanes[k]
                max_height = 0
                if len(lane_segs) == 1 and lane_segs[0][2] * lane_segs[0][3] < 10000:
                    continue
                for lane_seg in lane_segs:
                    if lane_seg[3] > max_height:
                        max_height = lane_seg[3]
                lane = [left_top_x, k, lane_width, max_height]
                pool_lanes.append(lane)
                next_lane_begin = k + max_height

    pool = {"rect": pool_rect, "lanes_rect": pool_lanes_rect, "header_rect": header_rect, "lanes": pool_lanes,
            "name": "pool"}
    return pool


def get_recs_bound(adjacent_recs):
    min_x = float("inf")
    min_y = float("inf")
    max_x = -1
    max_y = -1
    for [x, y, width, height] in adjacent_recs:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + width)
        max_y = max(max_y, y + height)

    return min_x, min_y, max_x - min_x, max_y - min_y


def get_elements(input_img, layers, contours_rec, partial_elements, pools_list, model_tag):
    layers_num = len(layers)
    upper_limit = min(model_tag + 3, layers_num)
    k = model_tag

    for ele_rec in partial_elements:
        for pool in pools_list:
            if helper.is_in(pool["lanes_rect"], ele_rec):
                lanes = pool["lanes"]
                for lane_i, lane in enumerate(lanes):
                    if helper.is_in(lane, ele_rec):
                        elements = pool.get("elements", defaultdict(list))
                        elements[lane_i].append(ele_rec)

    while k < upper_limit:
        layer = layers[k]
        k += 1

        # show layers
        # print(k)
        # layer_contour_rec = list(map(lambda x: contours_rec[x][0], layer))
        # base = np.zeros_like(input_img)
        # layer_img = helper.draw_rects(base, layer_contour_rec, cfg.COLOR_WHITE, cfg.CONTOUR_THICKNESS, show_text=True)
        # # cv.namedWindow("one_layer", cv.WINDOW_NORMAL)
        # cv.imshow("one_layer", layer_img)
        # cv.waitKey(0)

        boundary_fake_elements = []
        for c_i in layer:
            bound = contours_rec[c_i]
            bound_rect = bound[0]
            bound_rect_center = helper.get_rec_center(bound_rect)

            # filter the rec formed by lane boundary lines and sequence flows
            is_fake = False
            for fake_ele in boundary_fake_elements:
                if helper.is_adjacent(fake_ele, bound_rect):
                    is_fake = True
                    boundary_fake_elements.append(bound_rect)
                    break
            if is_fake:
                continue

            pool_id = -1
            lane_id = -1
            if bound_rect[2] < 15 or bound_rect[3] < 15:
                continue
            for pool_i, pool in enumerate(pools_list):
                if helper.is_in(pool["lanes_rect"], bound_rect):
                    pool_id = pool_i
                    lanes = pool["lanes"]
                    for lane_i, lane in enumerate(lanes):
                        # filter the rec formed by lane boundary lines and sequence flows
                        if helper.is_in(lane, bound_rect):

                            dilate_rec = helper.dilate([bound_rect_center[0], bound_rect_center[1], 0, 0], 3)
                            if helper.is_in(lane, dilate_rec):
                                lane_id = lane_i
                                break
                            else:
                                boundary_fake_elements.append(bound_rect)
                            break
            # print("pool_id:{},lane_id:{}".format(pool_id, lane_id))
            if pool_id < 0 or lane_id < 0:
                continue
            else:
                pool = pools_list[pool_id]
                elements = pool.get("elements", defaultdict(list))
                sub_procs = pool.get("sub_procs", defaultdict(list))

                found = False

                for j, ele_i_j in enumerate(elements[lane_id]):
                    if helper.is_in(ele_i_j, bound_rect):
                        found = True

                        if bound[2] < 800:
                            bound_rect = helper.dilate(bound_rect, cfg.RECT_DILATION_VALUE)

                        if bound[2] > 1050:
                            elements[lane_id][j] = bound_rect

                        if ele_i_j[2] * ele_i_j[3] > 3000:
                            elements[lane_id][j] = bound_rect

                if not found:
                    if bound[2] < cfg.SUB_PROC_AREA_THRESHOLD:
                        elements[lane_id].append(bound_rect)
                    else:
                        upper_limit = layers_num
                        sub_p_list = sub_procs.get(lane_id, [])
                        existed = False
                        for p_id, p in enumerate(sub_p_list):
                            if helper.is_in(p, bound_rect):
                                sub_p_list[p_id] = bound_rect
                                existed = True
                                break
                        if not existed:
                            # print("found sub_proc")
                            sub_procs[lane_id].append(bound_rect)
        # pools_img = draw_pools(pools_list, input_img)
        # cv.namedWindow("elements", cv.WINDOW_NORMAL)
        # cv.imshow("elements", pools_img)
        # cv.waitKey(0)

    # remove invalid sub_process
    for pool in pools_list:
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        sub_procs = pool.get("sub_procs", defaultdict(list))
        for lane_id, lane in enumerate(lanes):
            valid_sub_p = []
            for sub_p in sub_procs[lane_id]:
                # print(sub_p)
                if not helper.is_in(helper.shrink(lane, 5), sub_p):
                    continue
                sub_p_elements_num = 0
                valid = True
                for ele in elements[lane_id]:
                    dilate_ele = helper.dilate(ele, cfg.RECT_DILATION_VALUE)
                    if helper.is_in(sub_p, dilate_ele):
                        sub_p_elements_num += 1
                    elif helper.is_overlap(sub_p, dilate_ele):
                        valid = False
                        break
                if valid:
                    if sub_p_elements_num > 1:
                        valid_sub_p.append(sub_p)
                    elif sub_p_elements_num == 0:
                        elements[lane_id].append(sub_p)

            sub_procs[lane_id] = valid_sub_p

            # pools_img = draw_pools(pools_list, input_img)
            # # cv.namedWindow("elements", cv.WINDOW_NORMAL)
            # cv.imshow("elements", pools_img)
            # cv.waitKey(0)

    # remove blank element
    for pool in pools_list:
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        for lane_id in range(len(lanes)):
            non_blank = []
            for ele in elements[lane_id]:
                # ele = helper.shrink(ele, 1)
                roi = helper.truncate(input_img, ele)
                vertex_sum = sum(roi[0, 0]) + sum(roi[0, -1]) + sum(roi[-1, 0]) + sum(roi[-1, -1])
                edge_point_sum = sum(roi[0, ele[2] // 2]) + sum(roi[ele[3] // 2, 0]) + sum(roi[-1, ele[2] // 2]) + sum(
                    roi[ele[3] // 2, -1])

                if vertex_sum == 0 and edge_point_sum == 0:
                    print("has blank element")
                else:
                    non_blank.append(ele)
            elements[lane_id] = non_blank


    # remove embeded element
    for pool in pools_list:
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        for lane_id in range(len(lanes)):
            to_remove_set = set()
            one_lane_elements = elements[lane_id]
            for i in range(len(one_lane_elements)):
                if i in to_remove_set:
                    continue
                for j in range(i+1, len(one_lane_elements)):
                    if j in to_remove_set:
                        continue
                    if helper.is_in(helper.dilate(one_lane_elements[i], 1), one_lane_elements[j]):
                        to_remove_set.add(j)
                    elif helper.is_in(helper.dilate(one_lane_elements[j], 1), one_lane_elements[i]):
                        to_remove_set.add(i)

            to_remove_list = list(to_remove_set)
            to_remove_list.sort(reverse=True)
            for each in to_remove_list:
                one_lane_elements.pop(each)

    for pool_id, pool in enumerate(pools_list):
        # print("pool_{}:".format(pool_id))
        lanes = pool["lanes"]
        elements = pool["elements"]

        for lane_id in range(len(lanes)):
            unique_eles = []
            for ele in elements[lane_id]:
                if ele not in unique_eles:
                    unique_eles.append(ele)
            elements[lane_id] = unique_eles
            # print(len(unique_eles))

    for pool in pools_list:
        lanes = pool["lanes"]
        elements = pool.get("elements", defaultdict(list))
        for lane_id in range(len(lanes)):
            independent_eles = []
            one_lane_elements = elements[lane_id]
            to_merge_index = []

            for i in range(len(one_lane_elements)):
                if i in to_merge_index:
                    continue
                adjacent_eles = []
                for j in range(i+1, len(one_lane_elements)):
                    if helper.is_adjacent(one_lane_elements[i], one_lane_elements[j], dilation_value=3):
                        adjacent_eles.append(j)
                        to_merge_index.append(i)
                if len(adjacent_eles) == 0:
                    independent_eles.append(one_lane_elements[i])
                else:
                    adjacent_eles.append(i)
                    to_merge_index.extend(adjacent_eles)

                    adjacent_recs = [one_lane_elements[x] for x in adjacent_eles]

                    one_rec = get_recs_bound(adjacent_recs)
                    independent_eles.append(one_rec)
            elements[lane_id] = independent_eles

    return pools_list
