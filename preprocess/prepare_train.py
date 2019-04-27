# -*- coding:utf-8 -*-
import os
import cv2 as cv

import count
import rec_helper as rh


flow_ele_type_list = ["complexGateway", "eventBasedGateway", "exclusiveGateway", "inclusiveGateway", "parallelGateway",
                      "dataObject", "boundEvent_cancel", "boundEvent_compensate", "boundEvent_conditional",
                      "boundEvent_error", "boundEvent_escalation", "boundEvent_message", "boundEvent_signal",
                      "boundEvent_timer", "endEvent", "endEvent_cancel", "endEvent_compensate", "endEvent_error",
                      "endEvent_escalation", "endEvent_link", "endEvent_message", "endEvent_signal",
                      "endEvent_terminate", "interCatchEvent",
                      "interCatchEvent_conditional", "interCatchEvent_link", "interCatchEvent_message",
                      "interCatchEvent_signal", "interCatchEvent_timer", "interThrowEvent",
                      "interThrowEvent_compensate",
                      "interThrowEvent_escalation", "interThrowEvent_link", "interThrowEvent_message",
                      "interThrowEvent_signal",
                      "interThrowEvent_timer", "startEvent", "startEvent_compensate", "startEvent_conditional",
                      "startEvent_error",
                      "startEvent_escalation", "startEvent_message", "startEvent_signal", "startEvent_timer",
                      "textAnnotation", "dataStore", "busiRuleTask", "busiRuleTask_mulInsL_seq", "manualTask",
                      "manualTask_mulInsL_seq",
                      "receiveTask", "receiveTask_mulInsL_seq", "scriptTask", "scriptTask_mulInsL_seq",
                      "scriptTask_stdL_seq", "sendTask", "sendTask_mulInsL_seq", "serviceTask",
                      "serviceTask_mulInsL_seq",
                      "serviceTask_stdL_seq", "task", "task_mulInsL_seq", "task_stdL_seq", "userTask",
                      "userTask_mulInsL_seq", "userTask_stdL_seq"]


def make_type_dirs():
    train_data = "E:/diagrams/bpmn-io/bpmn2image/data0423/type_data/"

    for one_type in flow_ele_type_list:
        one_type_dir = train_data + one_type
        if not os.path.exists(one_type_dir):
            os.mkdir(one_type_dir)


def get_file_id(file_name):
    return "_".join(file_name.split("_")[0:2])


def distribute_shape_data():
    # data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/ele_type_data/"
    # files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/bpmn/"
    # imgs_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/imgs/"
    data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_type_data/"
    files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
    imgs_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images/"
    print("Start Counting ...")
    count.count(files_dir)
    print("Counting finished!")
    # count.statistic()
    all_shapes_label = count.all_shapes_label

    imgs = os.listdir(imgs_dir)

    i = 0
    file_id = get_file_id(imgs[i])
    file_path = imgs_dir + imgs[i]
    cur_img = 255 - cv.imread(file_path, cv.COLOR_BGR2GRAY)
    for shape_id, shape_label in enumerate(all_shapes_label):
        while shape_label[0] != file_id:
            i += 1
            file_id = get_file_id(imgs[i])
            print(file_id)
            file_path = imgs_dir + imgs[i]
            cur_img = 255 - cv.imread(file_path)

        shape_rec = shape_label[2]
        if shape_rec[2] * shape_rec[3] == 0:
            print(file_id)
            continue
        shape_rec = rh.dilate(shape_rec, 5)
        # print("{}:{}".format(str(shape_id)+"_"+file_id, shape_rec))
        shape_img = rh.truncate(cur_img, shape_rec)
        shape_path = data_dir+shape_label[1]+"/"+str(shape_id)+"_"+file_id+".png"
        cv.imwrite(shape_path, shape_img)


if __name__ == '__main__':
    # imgs_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/imgs/"
    # imgs = os.listdir(imgs_dir)
    # img = 255 - cv.imread(imgs_dir+imgs[0])
    #
    # roi = rh.truncate(img, [100, 100, 300, 300])
    # cv.imshow("roi", roi)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # files_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_3/bpmn/"
    # count.count(files_dir)
    # count.statistic()
    distribute_shape_data()
    # print(len(flow_ele_type_list))

