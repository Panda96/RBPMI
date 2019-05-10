# -*- coding:utf-8 -*-
import cfg
from helper import detector_helper as helper
from collections import defaultdict


def get_pools(layers, contours_rec):
    # tag = -1
    potential_pools = []
    # potential_elements = []
    layer_0 = layers[0].keys()
    for c_i in layer_0:
        bound_i = contours_rec[c_i]
        if bound_i[1] > cfg.POOL_AREA_THRESHOLD:
            # print(len(layers[0][c_i]))
            potential_pools.append(c_i)
        elif bound_i[1] > 20000 and len(layers[0][c_i]) <= 2:
            print("detect small pool")
            potential_pools.append(c_i)
        # else:
        #     potential_elements.append(bound_i[0])
            # else:
            #     blank_pools.append(bound_i[0])

    pools_object = []

    pools_num = len(potential_pools)
    print("pools_num:", pools_num)
    if pools_num == 0:
        # no lanes and no pools, just one process
        print("only elements")
        default_pool = create_default_pool(layer_0, contours_rec, 5)
        pools_object.append(default_pool)
        tag = 0

    else:
        print("has pools or lanes")
        for c_i in potential_pools:
            pool_bound = contours_rec[c_i]
            # c_children = layers[0][c_i]
            # Sort the child contours by x-coordinate
            children_recs = [[child_id, contours_rec[child_id]] for child_id in layers[0][c_i]]
            children_recs.sort(key=lambda x: x[1][0][0])
            c_children = [child_rec[0] for child_rec in children_recs]

            pools_in_one_rec = []

            children_rest = c_children.copy()
            for child_id in c_children:
                # get normal pools
                if child_id in children_rest:
                    child_bound = contours_rec[child_id]
                    if is_pool_header(child_bound):
                        # found a header means found a pool
                        pool = dict()
                        # remove header contour id
                        children_rest.remove(child_id)
                        # a rect with appropriate height-to-width ratio is very likely to be a header bounding box
                        header_rect = child_bound[0]

                        pool_rect = (header_rect[0], header_rect[1], pool_bound[0][2], header_rect[3])
                        pool["rect"] = pool_rect

                        lane_width = pool_bound[0][2] - header_rect[2]
                        pool_lanes_rect = [header_rect[0] + header_rect[2], header_rect[1], lane_width, header_rect[3]]
                        pool["lanes_rect"] = pool_lanes_rect

                        pool_lanes = []

                        lane_y_begin = pool_lanes_rect[1]
                        while lane_y_begin < pool_lanes_rect[1] + pool_lanes_rect[3] - 20:
                            # print(lane_y_begin)
                            # 按水平方向寻找lane
                            next_y_begin = lane_y_begin
                            for c_id in children_rest:
                                lane_seg_bound = contours_rec[c_id]
                                lane_seg_rec = lane_seg_bound[0]
                                # print(lane_seg_rec)
                                if helper.is_in(pool_lanes_rect, lane_seg_rec) and \
                                        lane_y_begin <= lane_seg_rec[1] < lane_y_begin + 5 and lane_seg_rec[2] > 30:
                                    # one_lane_seg.append(c_id)
                                    # print(lane_seg_rec[1])
                                    seg_y_end = lane_seg_rec[1] + lane_seg_rec[3]
                                    if seg_y_end > next_y_begin:
                                        next_y_begin = seg_y_end
                            # print("next:",next_y_begin)
                            lane = [pool_lanes_rect[0], lane_y_begin, pool_lanes_rect[2],
                                    next_y_begin - lane_y_begin]
                            # print(lane)
                            pool_lanes.append(lane)

                            one_lane_seg = []
                            for c_id in children_rest:
                                lane_seg_rec = contours_rec[c_id][0]
                                if helper.is_in(lane, lane_seg_rec):
                                    one_lane_seg.append(c_id)

                            for lane_seg in one_lane_seg:
                                children_rest.remove(lane_seg)
                            lane_y_begin = next_y_begin
                        pool["lanes"] = pool_lanes
                        pools_in_one_rec.append(pool)

            if len(pools_in_one_rec) == 0:
                print("has one pool only lanes")
                default_pool = create_default_pool([c_i], contours_rec, 0, layer1=c_children)
                pools_in_one_rec.append(default_pool)
            else:
                for c_id in children_rest:
                    child_bound = contours_rec[c_id]
                    if child_bound[1] > cfg.POOL_AREA_THRESHOLD:
                        print("has blank pool")
                        blank_pool = create_default_pool([c_id], contours_rec, 0)
                        pools_in_one_rec.append(blank_pool)

            pools_object.extend(pools_in_one_rec)

        # layer_1 = layers[1].keys()
        # if len(pools_object) == 0:
        #     print("no pools but lanes")
        #     default_pool = create_default_pool(layer_0, contours_rec, layer1=layer_1)
        #     pools_object.append(default_pool)
        tag = 1

    return pools_object, tag


def is_pool_header(bound):
    bound_rect = bound[0]
    ratio = bound_rect[3] / bound_rect[2]
    return cfg.POOL_HEADER_H_W_RATIO_FLOOR < ratio < cfg.POOL_HEADER_H_W_RATIO_CEILING


def create_default_pool(layer0, contours_rec, pool_dilate_value, layer1=None):
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

    pool = {"rect": pool_rect, "lanes_rect": pool_lanes_rect, "lanes": pool_lanes, "sub_procs": {}}
    return pool


def get_elements(layers, contours_rec, partial_elements, pools_list, model_tag):
    layers_num = len(layers)
    upper_limit = min(model_tag + 3, layers_num)
    k = model_tag

    potential_sub_procs = defaultdict(list)

    while k < upper_limit:
        layer = layers[k]
        k += 1
        for c_i in layer:
            bound = contours_rec[c_i]
            bound_rec = bound[0]

            pool_id = -1
            lane_id = -1
            for pool_i, pool in enumerate(pools_list):
                if helper.is_in(pool["lanes_rect"], bound_rec):
                    pool_id = pool_i
                    lanes = pool["lanes"]
                    for lane_i, lane in enumerate(lanes):
                        if helper.is_in(lane, bound_rec):
                            lane_id = lane_i
                            break
                    break

            if pool_id < 0 or lane_id < 0:
                continue
            else:
                # found pool and lane
                pass



def get_elements_raw(layers, contours_rec, partial_elements, pools_list, model_tag):
    layers_num = len(layers)
    upper_limit = min(model_tag + 3, layers_num)
    k = model_tag
    # 对后三层的轮廓进行遍历
    while k < upper_limit:
        layer = layers[k]
        # 遍历单层轮廓
        for c_i in layer:
            bound = contours_rec[c_i]
            bound_rect = bound[0]
            pool_pivot = None
            # 遍历泳池，确定该轮廓所属的泳池
            for pool in pools_list:
                if helper.is_in(pool["lanes_rect"], bound_rect):
                    pool_pivot = pool
                    break

            if pool_pivot is not None:
                # 找到所属泳池
                lanes = pool_pivot["lanes"]
                elements = pool_pivot.get("elements")
                if elements is None:
                    elements = defaultdict(list)
                    pool_pivot["elements"] = elements
                # 遍历泳道，确定该轮廓所属的泳道
                for i, lane in enumerate(lanes):
                    if helper.is_in(lane, bound_rect):
                        elements_i = elements[i]
                        num = len(elements_i)
                        found = False
                        # 找到所属泳道后判断是否与泳道中已有的元素重叠，
                        # 若重叠选择边界矩形面积小于930的那个作为元素边界矩形
                        for j in range(num):
                            if helper.is_in(elements_i[j], bound_rect):
                                found = True
                                if bound[2] < 930:
                                    bound_rect = helper.dilate(bound_rect, cfg.RECT_DILATION_VALUE)
                                elements_i[j] = bound_rect
                                break
                        if not found:
                            # filter the blank rectangle formed by element border and lane border
                            if bound_rect[3] < lane[3] - 2 * cfg.BOUNDARY_OFFSET:
                                sub_procs = pool_pivot.get("sub_procs", {})
                                if bound[2] < cfg.POOL_AREA_THRESHOLD:
                                    # 如果不重叠，且面积不是太大，不是子进程，则加入泳道元素列表
                                    elements_i.append(bound_rect)
                                else:
                                    # found subprocesses
                                    # 若是子进程，则遍历层数加深，将子进程轮廓加入所属泳池
                                    # 将子进程中的元素轮廓加入所属泳道
                                    upper_limit = layers_num

                                    sub_proc = sub_procs.get(i, None)
                                    if sub_proc is None:
                                        sub_procs[i] = [bound_rect]
                                    else:
                                        existed = False
                                        for proc_id, proc in enumerate(sub_proc):
                                            if helper.is_in(proc, bound_rect):
                                                sub_proc[proc_id] = bound_rect
                                                existed = True
                                                break
                                        if not existed:
                                            sub_proc.append(bound_rect)
        k += 1

    # 将粗边框元素合并到泳池元素中
    for ele_rect in partial_elements:
        for pool_id, pool in enumerate(pools_list):
            pool_lanes_rect = pool["lanes_rect"]
            if helper.is_overlap(pool_lanes_rect, ele_rect):
                elements = pool["elements"]
                lanes = pool["lanes"]
                for lane_id, lane in enumerate(lanes):
                    if helper.is_overlap(lane, ele_rect):
                        elements_in_lane = elements.get(lane_id)
                        existed = False
                        # 若与已检测出的元素冲突，选择重叠面积占比大的那一个
                        for ele_id, element in enumerate(elements_in_lane):
                            if helper.is_overlap(element, ele_rect):
                                existed = True
                                area1 = element[2] * element[3]
                                area2 = ele_rect[2] * ele_rect[3]
                                if area2 < area1:
                                    elements_in_lane[ele_id] = ele_rect
                        if not existed:
                            elements_in_lane.append(ele_rect)
                        break
                break

    # 筛选子进程，若元素横穿或者靠近子进程边界，则不是子进程
    for pool in pools_list:
        elements = pool["elements"]
        for i, elements_i in elements.items():
            num = len(elements_i)
            remove_set = set()
            for j in range(num):
                sub_procs = pool.get("sub_procs", {})
                for lane_id, lane_sub_procs in sub_procs.items():
                    for proc_id, one_proc in enumerate(lane_sub_procs):
                        shrink_proc = helper.shrink(one_proc, cfg.RECT_DILATION_VALUE)
                        if not helper.is_in(shrink_proc, elements_i[j]) and helper.is_overlap(shrink_proc,
                                                                                              elements_i[j]):
                            lane_sub_procs[proc_id] = None
                    lane_sub_procs = list(filter(lambda x: x is not None, lane_sub_procs))
                    if len(lane_sub_procs) == 0:
                        sub_procs.pop(lane_id)
                        break
                    else:
                        sub_procs[lane_id] = lane_sub_procs
                        break

                # 这里很奇怪，可能是针对特定情况的元素筛选
                for m in range(j + 1, num):
                    if helper.is_adjacent(elements_i[j], elements_i[m]):
                        # two elements adjacent
                        remove_set.add(m)
                        remove_set.add(j)
                    else:
                        # two elements intersect
                        overlap_area = helper.get_overlap_area(elements_i[j], elements_i[m])
                        if overlap_area > 0:
                            area_j = elements_i[j][2] * elements_i[j][3]
                            area_m = elements_i[m][2] * elements_i[m][3]
                            if area_m < 1200 or area_j < 1200:
                                if area_m < area_j:
                                    remove_set.add(m)
                                else:
                                    remove_set.add(j)
                            else:
                                ratio_j = overlap_area / area_j
                                ratio_m = overlap_area / area_m
                                if ratio_j > ratio_m:
                                    remove_set.add(m)
                                else:
                                    remove_set.add(j)

            for j in remove_set:
                elements_i[j] = None
            elements_i = list(filter(lambda x: x is not None, elements_i))
            elements[i] = elements_i

    return pools_list
