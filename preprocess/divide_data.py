# -*- coding:utf-8 -*-
import os
import numpy as np
import shutil

bpmn_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/files/"
img_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/images/"

all_bpmn = os.listdir(bpmn_dir)
all_img = os.listdir(img_dir)
all_bpmn.sort()
all_img.sort()

size = len(all_bpmn)
index = np.arange(size)
np.random.seed(423)
np.random.shuffle(index)

num = size // 700 + 1

for i in range(num):
    data_set_dir = "E:/diagrams/bpmn-io/bpmn2image/data0423/data700_{}/".format(i)
    for j in index[i*700:(i+1)*700]:
        shutil.copy(bpmn_dir+all_bpmn[j], data_set_dir+"bpmn")
        shutil.copy(img_dir+all_img[j], data_set_dir+"imgs")


