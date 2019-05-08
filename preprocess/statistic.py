# -*- coding:utf-8 -*-
import os

data_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/ele_enough_data/"

type_list = os.listdir(data_dir)
count = []
for each in type_list:
    imgs = os.listdir(data_dir + each)
    count.append([each, len(imgs)])

count.sort(key=lambda x: x[1])

for each in count:
    print(each)
