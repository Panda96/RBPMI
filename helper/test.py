# -*- coding:utf-8 -*-
import os


def get_type_num(dir_path):
    res = []
    type_dirs = os.listdir(dir_path)
    for type_dir in type_dirs:
        files = os.listdir(dir_path+type_dir)
        res.append([type_dir, str(len(files))])
    return res


dir_1 = "E:/diagrams/bpmn-io/bpmn2image/data0423/98_type_data/ele_type_data/"
dir_2 = "E:/diagrams/bpmn-io/bpmn2image/data0423/98_type_data/ele_little_data/"
dir_3 = "E:/diagrams/bpmn-io/bpmn2image/data0423/70_type_data/ele_type_data/"

res_1 = get_type_num(dir_1)
res_2 = get_type_num(dir_2)
res_3 = get_type_num(dir_3)

# for res in res_1:
#     print(",".join(res))

# for res in res_2:
#     print(",".join(res))
#
for res in res_3:
    print(",".join(res))
