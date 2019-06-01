# -*- coding:utf-8 -*-
import json
import os
import numpy as np


def get_element_ratio(shape_count):
    [true_shape_num, detect_shape_num, type_true_num, type_wrong_num, fake_shapes] = shape_count

    if true_shape_num == 0:
        # print("no elements")
        # exit(0)
        return None
    else:
        detection_acc = detect_shape_num / true_shape_num
        overall_acc = type_true_num / true_shape_num


        if detect_shape_num == 0:
            # print("detect elements all fake")
            # detection_acc = 0
            classify_acc = 0
            adjust_ratio = 1
            if fake_shapes == 0:
                # print("detect no elements")
                fake_ratio = 0
            else:
                fake_ratio = 1

            # detect_valid = 0

        else:
            # detection_acc = detect_shape_num / true_shape_num
            classify_acc = type_true_num / detect_shape_num
            fake_ratio = fake_shapes / detect_shape_num
            adjust_ratio = (type_wrong_num + fake_shapes) / (detect_shape_num + fake_shapes)
            # detect_valid = type_true_num/
    return [detection_acc, classify_acc, overall_acc, fake_ratio, adjust_ratio]


def analyse(count_list):
    shape_ratios = []
    shape_none_count = 0
    for shape_e in count_list:
        ratio = get_element_ratio(shape_e)
        if ratio is None:
            shape_none_count += 1
        else:
            shape_ratios.append(ratio)
    return shape_ratios, shape_none_count


def show_one_ratio(one_ratio):
    print("detection_acc:{}".format(one_ratio[0]))
    print("classify_acc:{}".format(one_ratio[1]))
    print("overall_acc:{}".format(one_ratio[2]))
    print("fake_ratio:{}".format(one_ratio[3]))
    print("adjust_ratio:{}".format(one_ratio[4]))


def get_average_ratio(ratios):
    ratios = np.array(ratios)
    print("avg_detection_acc:{}".format(np.mean(ratios[:, 0])))
    print("avg_classify_acc:{}".format(np.mean(ratios[:, 1])))
    print("avg_overall_acc:{}".format(np.mean(ratios[:, 2])))
    print("avg_fake_acc:{}".format(np.mean(ratios[:, 3])))
    print("avg_adjust_acc:{}".format(np.mean(ratios[:, 4])))


# def get_seq_ratio(seq_count):
#     [seq_num, target_match_num, detect_true_num, seq_detect_wrong, others_num] = seq_count
#
#     if seq_num == 0:
#         print("no seq flows")
#         return None
#     else:
#         detection_acc = target_match_num / seq_num
#         overal_acc = detect_true_num / seq_num
#         if detect_shape_num == 0:
#             print("detect shapes all fake")
#             # detection_acc = 0
#             classify_acc = 0
#             adjust_ratio = 1
#             if fake_shapes == 0:
#                 print("detect no shapes")
#                 fake_ratio = 0
#             else:
#                 fake_ratio = 1
#
#         else:
#             # detection_acc = detect_shape_num / true_shape_num
#             classify_acc = type_true_num / detect_shape_num
#             fake_ratio = fake_shapes / detect_shape_num
#             adjust_ratio = (type_wrong_num + fake_shapes) / (detect_shape_num + fake_shapes)
#     return [detection_acc, classify_acc, overal_acc, fake_ratio, adjust_ratio]


def main():
    v_res_dir = "validate_results/"
    bcf_56_1_json_file = "1_bcf_56_gen_data_valid_res.json"
    bcf_56_2_json_file = "4_bcf_56_gen_data_valid_res.json"
    bcf_57_1_json_file = "2_bcf_57_gen_data_valid_res.json"
    bcf_57_2_json_file = "5_bcf_57_gen_data_valid_res.json"

    vgg16_56_1_json_file = "1_vgg16_56_gen_data_valid_res.json"
    vgg16_56_2_json_file = "7_vgg16_56_gen_data_valid_res.json"
    vgg16_57_1_json_file = "2_vgg16_57_gen_data_valid_res.json"
    vgg16_57_2_json_file = "8_vgg16_57_gen_data_valid_res.json"

    data_dir = "../gen_data_valid/"
    labels = ["task", "userTask", "serviceTask", "sendTask", "scriptTask", "receiveTask", "manualTask",
              "businessRuleTask", "exclusiveGateway", "exclusiveGateway_fork", "parallelGateway", "inclusiveGateway",
              "eventBasedGateway", "complexGateway", "startEvent", "startEvent_message", "startEvent_timer",
              "startEvent_conditional", "startEvent_signal", "startEvent_isInterrupting", "startEvent_escalation",
              "startEvent_compensate", "startEvent_error", "endEvent", "endEvent_terminate", "endEvent_message",
              "endEvent_cancel", "endEvent_error", "endEvent_signal", "endEvent_compensate", "endEvent_escalation",
              "endEvent_link", "callActivity", "subProcess", "transaction", "adHocSubProcess",
              "subProcess_triggeredByEvent", "dataObjectReference", "dataStoreReference",
              "intermediateCatchEvent_message", "intermediateCatchEvent_timer", "intermediateCatchEvent_conditional",
              "intermediateCatchEvent_signal", "intermediateCatchEvent_link", "intermediateThrowEvent",
              "intermediateThrowEvent_message", "intermediateThrowEvent_link", "intermediateThrowEvent_signal",
              "intermediateThrowEvent_escalation", "intermediateThrowEvent_compensate", "boundaryEvent_error",
              "boundaryEvent_escalation", "boundaryEvent_compensate", "boundaryEvent_cancel",
              "boundaryEvent_timer_cancelActivity"]

    # json_path = v_res_dir + bcf_57_1_json_file
    # json_path = v_res_dir + bcf_57_2_json_file
    # json_path = v_res_dir + vgg16_57_1_json_file
    json_path = v_res_dir + vgg16_57_2_json_file

    # json_path = v_res_dir + bcf_56_1_json_file
    # json_path = v_res_dir + bcf_56_2_json_file
    # json_path = v_res_dir + vgg16_56_1_json_file
    # json_path = v_res_dir + vgg16_56_2_json_file

    # res_list = []
    with open(json_path, "r") as fj:
        res_list = json.load(fj)

    print("res_list:{}".format(len(res_list)))
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

        shape_each.append([true_shape_num, detect_shape_num, type_true_num, type_wrong_num, fake_shapes])

        seq_total[0] += seq_num
        seq_total[1] += target_match_num
        seq_total[2] += detect_true_num
        seq_total[3] += seq_detect_wrong
        seq_total[4] += others_num

        seq_each.append([seq_num, target_match_num, detect_true_num, seq_detect_wrong, others_num])

    print("analyse shapes")
    shape_ratios, shape_none_count = analyse(shape_each)
    print(len(shape_ratios))
    print(shape_none_count)
    print("-"*100)
    print("analyse seqs")
    seq_ratios, seq_none_count = analyse(seq_each)
    print(len(seq_ratios))
    print(seq_none_count)
    print("-"*100)
    shape_all_ratios, shape_all_none_count = analyse([shape_total])
    seq_all_ratios, seq_all_non_count = analyse([seq_total])

    print("shape")
    show_one_ratio(shape_all_ratios[0])
    get_average_ratio(shape_ratios)
    print("seqs")
    show_one_ratio(seq_all_ratios[0])
    get_average_ratio(seq_ratios)


if __name__ == '__main__':
    main()
