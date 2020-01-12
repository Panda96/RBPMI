import os
import re
import numpy as np


def get_categories():
    cate_name_list = []
    cate_abbrev_list = []

    with open(categories_file, mode="r", encoding="utf-8") as f:
        for line in f:
            [cate_name, cate_abbr] = line.split("\t")
            cate_name_list.append(cate_name)
            cate_abbrev_list.append(cate_abbr[:-1])

    return cate_name_list, cate_abbrev_list


def read_one_log(log_file, log_name):
    log_res = dict()
    with open(log_file, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Mistook"):
                continue
            # if "endEvent_terminate" in line:
            #     print(line)
            line = line[:-1]
            start = re.search("\d", line).start()
            cate_name = line[:start].strip()
            number_res = line[start:]
            each_acc = number_res.split(",")[-1]
            log_res[cate_name] = each_acc

    log_dict[log_name] = log_res


def main():
    logs = os.listdir(log_dir)
    for log in logs:
        if not log.endswith(".txt"):
            continue
        segments = log.split("_")
        log_name = "_".join(segments[1:5])

        read_one_log("{}/{}".format(log_dir, log), log_name)

    log_names = list(log_dict.keys())
    log_names.sort()
    all_res = []
    header = ["cate_name", "cate_abbrev", "bcf_jpg", "bcf_png", "vgg_jpg_mean", "vgg_jpg_max", "vgg_png_mean", "vgg_png_max"]

    for each_id, each in enumerate(cate_list):
        each_res_list = [each, abbr_list[each_id]]
        temp_res = []
        for log_name in log_names:
            log_res = log_dict[log_name]
            try:
                temp_res.append(log_res[each])
            except KeyError:
                print(log_name, each)

        # bcf_jpg
        bcf_jpg = temp_res[0]
        # bcf_png
        bcf_png = temp_res[1]

        vgg_jpg = temp_res[2:6]
        vgg_jpg = [float(x) for x in vgg_jpg]
        vgg_jpg_mean = str(np.mean(vgg_jpg))
        vgg_jpg_max = str(np.max(vgg_jpg))
        vgg_png = temp_res[6:]
        vgg_png = [float(x) for x in vgg_png]
        vgg_png_mean = str(np.mean(vgg_png))
        vgg_png_max = str(np.max(vgg_png))

        each_res_list.extend([bcf_jpg, bcf_png, vgg_jpg_mean, vgg_jpg_max, vgg_png_mean, vgg_png_max])

        all_res.append(each_res_list)

    all_res.insert(0, header)
    # all_res = list(np.transpose(np.array(all_res)))
    with open("output/all_res.txt", encoding="utf-8", mode="w") as f:
        for each_res in all_res:
            f.write(",".join(each_res) + "\n")


if __name__ == '__main__':
    log_dir = "E:/master/TSE2020/classifiers"
    categories_file = "categories.txt"
    cate_list, abbr_list = get_categories()
    # print(cate_list)
    # print(abbr_list)
    log_dict = dict()
    main()
    # a = "endEvent_terminate123"
    # print(re.search("\d", a).start())


