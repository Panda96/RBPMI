import os
import sys

sys.path.append("..")

from collections import defaultdict
from functools import reduce

from scipy.spatial.distance import cdist
import numpy as np
import sklearn
import sklearn.cluster
import pickle
import time

from evolution import evolution
from shape_context import shape_context
from llc import llc_coding_approx
import image_parser as image_parser


class BCF:
    def __init__(self, ):
        self.DATA_DIR = "/home/plw/workplace/python/segmentation/training_data_jpg/train/"
        self.CODEBOOK_FILE = "model/code_book_52_jpg.data"
        self.CLASSIFIER_FILE = "model/classifier_52_jpg"
        # self.LABEL_TO_CLASS_MAPPING_FILE = "model/labels_to_classes.data"
        self.classes = defaultdict(list)
        self.data = defaultdict(dict)
        self.counter = defaultdict(int)
        self.kmeans = None
        self.clf = None
        self.label_to_class_mapping = None

    # def save_label_to_class_mapping(self):
    #     self.label_to_class_mapping = {hash(cls): cls for cls in os.listdir(self.DATA_DIR)}
    #     with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'wb') as out_file:
    #         pickle.dump(self.label_to_class_mapping, out_file, -1)
    #
    # def load_label_to_class_mapping(self):
    #     if self.label_to_class_mapping is None:
    #         with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'rb') as in_file:
    #             self.label_to_class_mapping = pickle.load(in_file)
    #     return self.label_to_class_mapping

    def get_image_shape_feats(self, image):
        # print(image)
        shapes_feature = []
        if type(image) == str:
            input_img, _, contours = image_parser.get_layers_by_file_name(image)
        else:
            input_img, _, contours = image_parser.get_layers_by_img(image)
        selected_contours_id = image_parser.filter_contours(input_img.shape, contours)
        sz = input_img.shape
        for contour_id in selected_contours_id:
            contour = contours[contour_id]
            points = []
            for point in contour:
                points.append([point[0][0], point[0][1]])
            max_curvature = 1.5
            n_contsamp = 50
            n_pntsamp = 10
            if len(points) > 1:
                cfs = self.extr_raw_points(np.array(points), max_curvature, n_contsamp, n_pntsamp)

                num_cfs = len(cfs)
                # print("Extracted %s points" % num_cfs)
                contour_feature = np.zeros((300, num_cfs))
                xy = np.zeros((num_cfs, 2))
                for i in range(num_cfs):
                    cf = cfs[i]
                    sc, _, _, _ = shape_context(cf)
                    # shape context is 60x5 (60 bins at 5 reference points)
                    sc = sc.flatten(order='F')
                    sc /= np.sum(sc)  # normalize
                    contour_feature[:, i] = sc
                    # shape context descriptor sc for each cf is 300x1
                    # save a point at the midpoint of the contour fragment
                    xy[i, 0:2] = cf[np.round(len(cf) / 2. - 1).astype('int32'), :]

                shapes_feature.append([contour_feature, xy])
        shape_feats = [shapes_feature, sz]
        return shape_feats

    def extract_cf(self, upper):
        type_dirs = os.listdir(self.DATA_DIR)
        print("Extracting CF...")
        self.print_time()
        for type_dir in type_dirs:
            print(type_dir)
            images = os.listdir(self.DATA_DIR + type_dir)
            size = len(images)
            index = np.arange(size)
            np.random.shuffle(index)

            for i in index[:upper]:
                image = images[i]
                image_key = (type_dir, image)
                # print(image_key)
                image_path = self.DATA_DIR + type_dir + "/" + image
                self.data[image_key]['cfs'] = self.get_image_shape_feats(image_path)
        self.print_time()
        print("Extracting finished!")

    def print_time(self):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    def learn_codebook(self):
        MAX_CFS = 800  # max number of contour fragments per image_mat; if above, sample randomly
        CLUSTERING_CENTERS = 1500
        feats_sc = []
        for image in self.data.values():
            feats = image['cfs']
            shapes_feature = feats[0]
            for shape_feature in shapes_feature:
                contour_feature = shape_feature[0]
                if contour_feature.shape[1] > MAX_CFS:
                    # Sample MAX_CFS from contour fragments
                    rand_indices = np.random.permutation(contour_feature.shape[1])
                    contour_feature = contour_feature[:, rand_indices[:MAX_CFS]]
                feats_sc.append(contour_feature)

        feats_sc = np.concatenate(feats_sc, axis=1).transpose()
        print("feats_sc size:{}".format(len(feats_sc)))
        print("Running KMeans...")
        self.print_time()
        self.kmeans = sklearn.cluster.KMeans(min(CLUSTERING_CENTERS, feats_sc.shape[0]), n_jobs=-1,
                                             algorithm='elkan').fit(feats_sc)
        self.print_time()
        print("Saving codebook...")

        self.save_kmeans(self.kmeans)
        return self.kmeans

    def save_kmeans(self, kmeans):
        with open(self.CODEBOOK_FILE, 'wb') as out_file:
            pickle.dump(kmeans, out_file, -1)

    def load_kmeans(self):
        if self.kmeans is None:
            with open(self.CODEBOOK_FILE, 'rb') as in_file:
                self.kmeans = pickle.load(in_file)
        return self.kmeans

    def encode_shape_feats(self, shape_feats, kmeans, k_nn):
        shapes_feature = shape_feats[0]
        encoded_shape_feats = []
        for shape_feature in shapes_feature:
            contour_feature = shape_feature[0]
            # print("raw:{}".format(contour_feature.shape))
            contour_feature = np.array(contour_feature)
            # print("after:{}".format(contour_feature.shape))
            encoded_shape_feats.append(
                llc_coding_approx(kmeans.cluster_centers_, contour_feature.transpose(), k_nn))
        return encoded_shape_feats

    def encode_cf(self):
        print("Encoding ...")
        self.print_time()
        k_nn = 5
        kmeans = self.load_kmeans()
        # Represent each contour fragment shape descriptor as a combination of K_NN of the
        # clustering centers
        for image in self.data.values():
            image["llc_coding"] = self.encode_shape_feats(image["cfs"], kmeans, k_nn)
        self.print_time()
        print("Encoding finished!")

    def spp_llc_code(self, pyramid, shape_feats, encoded_shape_feats):
        shapes_feature = shape_feats[0]
        shapes_spp_feature = []
        for shape_id, shape_feature in enumerate(shapes_feature):
            shape_feas = self.pyramid_pooling(pyramid, shape_feats[1], shape_feature[1], encoded_shape_feats[shape_id])
            shape_spp_fea = shape_feas.flatten()
            shape_spp_fea /= np.sqrt(np.sum(shape_spp_fea ** 2))
            shapes_spp_feature.append(shape_spp_fea)

        spp_fea_size = len(shapes_spp_feature[0])
        spp_feature = np.zeros(spp_fea_size * 30)

        if len(shapes_spp_feature) > 30:
            head = shapes_spp_feature[0:15]
            tail = shapes_spp_feature[-15:]
            head.extend(tail)
            shapes_spp_feature = head

        for i in range(len(shapes_spp_feature)):
            spp_feature[i * spp_fea_size:(i + 1) * spp_fea_size] = shapes_spp_feature[i]
        return spp_feature

    def spp(self):
        print("SPP Begin...")
        self.print_time()
        pyramid = np.array([1, 2, 4])
        for image_key, image in self.data.items():
            feat = image['cfs']
            image['spp_descriptor'] = self.spp_llc_code(pyramid, image['cfs'], image["llc_coding"])
            # print(image_key)
            # print(image_mat["spp_descriptor"].shape)
        self.print_time()
        print("SPP Finished!")

    def pyramid_pooling(self, pyramid, sz, xy, encoded_shape_descriptors):
        feas = np.zeros((encoded_shape_descriptors.shape[1], np.sum(pyramid ** 2)))
        counter = 0
        height = sz[0]
        width = sz[1]
        x = xy[:, 0]  # midpoint for each contour fragment
        y = xy[:, 1]
        for p in range(len(pyramid)):
            for i in range(pyramid[p]):
                for j in range(pyramid[p]):
                    yrang = height * np.array([float(i), float(i + 1)]) / pyramid[p]
                    xrang = width * np.array([float(j), float(j + 1)]) / pyramid[p]
                    c = encoded_shape_descriptors[reduce(np.logical_and, [x >= xrang[0], x < xrang[1], y >= yrang[0],
                                                                          y < yrang[1]])]  # get submatrix
                    if c.shape[0] == 0:
                        f = np.zeros(encoded_shape_descriptors.shape[1])
                    else:
                        f = np.amax(c, axis=0)  # max vals in submatrix
                    feas[:len(f), counter] = f
                    counter += 1
        return feas

    def save_classifier(self, clf):
        with open(self.CLASSIFIER_FILE, 'wb') as out_file:
            pickle.dump(clf, out_file, -1)

    def load_classifier(self):
        if self.clf is None:
            with open(self.CLASSIFIER_FILE, 'rb') as in_file:
                self.clf = pickle.load(in_file)
        return self.clf

    def svm_train(self):
        # print("SVM Training...")
        # self.print_time()
        # self.save_label_to_class_mapping()
        clf = sklearn.svm.LinearSVC(multi_class='crammer_singer')
        training_data = []
        labels = []
        for (cls, idx) in self.data.keys():
            training_data.append(self.data[(cls, idx)]['spp_descriptor'])
            labels.append(cls)
        print("Training SVM...")
        self.print_time()
        self.clf = clf.fit(training_data, labels)
        self.print_time()
        print("Saving classifier...")
        self.save_classifier(self.clf)
        return self.clf

    # def show(self, image_mat):
    #     cv2.imshow('image_mat', image_mat)
    #     _ = cv2.waitKey()

    def extr_raw_points(self, c, max_value, N, nn):
        # -------------------------------------------------------
        # [SegmentX, SegmentY,NO]=GenSegmentsNew(a,b,maxvalue,nn)
        # This function is used to generate all the segments
        # vectors of the input contour
        # a and b are the input contour sequence
        #  maxvalue is the stop condition of DCE, usually 1~1.5
        #  nn is the sample points' number on each segment, in super's method,n=25
        # SegmentX,SegmentY denotes all the coordinates of all the segments of input contour
        # NO denotes the number of segments of input contour
        # -------------------------------------------------------
        kp, _, _ = evolution(c, N, max_value, 0, 0, 0)  # critical points
        n2 = cdist(kp, c)

        i_kp = np.argmin(n2.transpose(), axis=0)  # column-wise min
        n_kp = len(i_kp)
        # n_cf = (n_kp - 1) * n_kp + 1
        pnts = []

        # s = 0
        for i in range(n_kp):
            for j in range(n_kp):
                if i == j:
                    continue
                if i < j:
                    cf = c[i_kp[i]:i_kp[j] + 1, :]
                if i > j:
                    cf = np.append(c[i_kp[i]:, :], c[:i_kp[j] + 1, :], axis=0)

                if cf.shape[0] > 1:
                    pnts.append(self.sample_contour(cf, nn))
                # s += 1
        # if c.shape[0] > 1:
        pnts.append(self.sample_contour(c, nn))
        return pnts

    def sample_contour(self, cf, nn):
        # Sample points from contour fragment
        _len = cf.shape[0]
        # try:
        ii = np.round(np.arange(0, _len - 0.9999, float(_len - 1) / (nn - 1))).astype('int32')
        # except ZeroDivisionError:
        #     print("_len:{}".format(_len))
        #     print("nn-1:{}".format(nn - 1))
        #     exit(0)

        cf = cf[ii, :]
        return cf

    def train_code_book(self, code_book_num):
        self.extract_cf(code_book_num)
        self.learn_codebook()

    def train(self, classifier_num):
        self.extract_cf(classifier_num)
        self.encode_cf()
        self.spp()
        self.svm_train()

    def get_one_image_feature(self, image):
        shape_feats = self.get_image_shape_feats(image)
        k_nn = 5
        kmeans = self.load_kmeans()
        encoded_shape_feats = self.encode_shape_feats(shape_feats, kmeans, k_nn)
        pyramid = np.array([1, 2, 4])
        spp_feature = self.spp_llc_code(pyramid, shape_feats, encoded_shape_feats)
        return spp_feature

    def get_images_type(self, images):
        clf = self.load_classifier()
        testing_data = []
        invalid_data = []
        for image_id, image in enumerate(images):
            try:
                spp_feature = self.get_one_image_feature(image)
            except IndexError:
                invalid_data.append(image_id)

            testing_data.append(spp_feature)
        testing_data = np.array(testing_data)
        predictions = clf.predict(testing_data)
        return predictions, invalid_data

    def test_dir(self, test_data):

        type_dirs = os.listdir(test_data)
        # {type:[[test_num, correct_num], ["mistook info"...]]}
        test_res = {}
        for each_type in type_dirs:
            print(each_type)
            test_res[each_type] = [[0, 0], []]
            type_dir = test_data + each_type + "/"
            image_names = os.listdir(type_dir)
            images = [type_dir + name for name in image_names]
            predictions, _ = self.get_images_type(images[0:1])
            test_res[each_type][0][0] = len(images)
            for i, prediction in enumerate(predictions):
                if prediction == each_type:
                    test_res[each_type][0][1] += 1
                else:
                    test_res[each_type][1].append("Mistook {} {} for {}".format(image_names[i], each_type, prediction))

        all_total = 0
        all_correct = 0
        for label in type_dirs:
            total = test_res[label][0][0]
            correct_num = test_res[label][0][1]
            all_total += total
            all_correct += correct_num
            print("{}\t{},{},{}".format(label, total, correct_num, correct_num / total))
            for info in test_res[label][1]:
                print(info)
        print("{}\t{},{},{}".format("all", all_total, all_correct, all_correct / all_total))
        return test_res


if __name__ == "__main__":
    # sys.path.append("../helper")
    # sys.path.append("..")
    # sys.path.append("../bcf")
    # print(sys.path)
    code_book_train_num = 30
    classifier_train_num = 50
    bcf = BCF()

    # print("--------train code book-----------")
    # bcf.train_code_book(code_book_train_num)

    print("---------train model----------")
    bcf.train(classifier_train_num)

