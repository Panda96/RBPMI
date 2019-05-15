# -*- coding:utf-8 -*-
import json
import os
import numpy as np


def show_res(total, each):
    print(total)
    print(total[1] / total[0])
    print(total[2] / total[0])
    print(total[2] / total[1])

    each = np.array(each)
    print(np.mean(each[:, 0]))
    print(np.mean((each[:, 1])))
    print(np.mean(each[:, 2]))


v_res_dir = "validate_results/"
json_file = "3_bcf_gen_data_valid_res.json"

data_dir = "../gen_data_valid/"
labels = os.listdir("../57_622data/train/")

json_path = v_res_dir + json_file

# res_list = []
with open(json_path, "r") as fj:
    res_list = json.load(fj)

labels_count = {}

for label in labels:
    labels_count[label] = [0, 0, 0, 0]

shape_total = [0, 0, 0, 0, 0]
seq_total = [0, 0, 0, 0, 0]
shape_each = []
seq_each = []

for res in res_list:
    shapes = res["shapes"]
    true_shape_num = 0
    detect_shape_num = 0
    type_true_num = 0
    type_wrong_num = 0
    for label in labels:
        shape_label = shapes.get(label, None)
        # print(shape_label)
        if shape_label is not None:
            # print("{}\t{},{},{},{}".format(label, shape_label[0], shape_label[1], shape_label[2], shape_label[3]))
            true_shape_num += shape_label[0]
            detect_shape_num += shape_label[1]
            type_true_num += shape_label[2]
            type_wrong_num += shape_label[3]
            labels_count[label][0] += shape_label[0]
            labels_count[label][1] += shape_label[1]
            labels_count[label][2] += shape_label[2]
            labels_count[label][3] += shape_label[3]
    #
    fake_shapes = res["fake_elements"]
    # print("fake_shapes\t{}".format(fake_shapes))

    # print("shapes\t{},{},{},{},{}".format(true_shape_num, detect_shape_num, type_true_num, type_wrong_num, fake_shapes))

    seq = res["seq_record"]
    seq = seq.split("\t")[1]
    [seq_num, seq_detect_wrong, target_match_num, detect_true_num, others_num] = [int(x) for x in seq.split(",")]
    # print("seqs\t{},{},{},{},{}".format(seq_num, target_match_num, detect_true_num, seq_detect_wrong, others_num))

    shape_total[0] += true_shape_num
    shape_total[1] += detect_shape_num
    shape_total[2] += type_true_num
    shape_total[3] += type_wrong_num
    shape_total[4] += fake_shapes
    if true_shape_num != 0:
        if detect_shape_num != 0:
            shape_each.append(
                [detect_shape_num / true_shape_num, type_true_num / true_shape_num, type_true_num / detect_shape_num])
        else:
            shape_each.append([detect_shape_num / true_shape_num, type_true_num / true_shape_num, 0])
    seq_total[0] += seq_num
    seq_total[1] += target_match_num
    seq_total[2] += detect_true_num
    seq_total[3] += seq_detect_wrong
    seq_total[4] += others_num
    if seq_num != 0:
        if target_match_num != 0:
            seq_each.append([target_match_num/seq_num, detect_true_num/seq_num, detect_true_num/target_match_num])
        else:
            seq_each.append([target_match_num / seq_num, detect_true_num / seq_num, 0])
# for label in labels:
#     print(label)
#     label_c = labels_count[label]
#     print("{}\t{},{},{},{}".format(label, label_c[0], label_c[1], label_c[2], label_c[3]))

show_res(shape_total, shape_each)

show_res(seq_total, seq_each)
