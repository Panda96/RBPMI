import os
import sys
from collections import defaultdict
from functools import reduce

from scipy.spatial.distance import cdist
import numpy as np
import cv2
import sklearn
import sklearn.cluster
import pickle

from evolution import evolution
from shape_context import shape_context
from llc import llc_coding_approx
import image_parser


class BCF:
    def __init__(self):
        self.DATA_DIR = "data/train/"
        self.CODEBOOK_FILE = "model/codebook.data"
        self.CLASSIFIER_FILE = "model/classifier"
        self.LABEL_TO_CLASS_MAPPING_FILE = "model/labels_to_classes.data"
        self.classes = defaultdict(list)
        self.data = defaultdict(dict)
        self.counter = defaultdict(int)
        self.kmeans = None
        self.clf = None
        self.label_to_class_mapping = None

    def _save_label_to_class_mapping(self):
        self.label_to_class_mapping = {hash(cls): cls for cls in self.classes}
        with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'wb') as out_file:
            pickle.dump(self.label_to_class_mapping, out_file, -1)

    def _load_label_to_class_mapping(self):
        if self.label_to_class_mapping is None:
            with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'rb') as in_file:
                self.label_to_class_mapping = pickle.load(in_file)
        return self.label_to_class_mapping

    def _extract_cf(self):
        type_dirs = os.listdir(self.DATA_DIR)
        for type_dir in type_dirs:
            images = os.listdir(self.DATA_DIR + type_dir)
            for image in images[0:5]:
                image_key = (type_dir, image)
                print(image_key)
                image_path = self.DATA_DIR + type_dir + "/" + image
                input_img, _, contours = image_parser.get_layers(image_path)
                selected_contours_id = image_parser.filter_contours(input_img.shape, contours)
                sz = input_img.shape

                shapes_feature = []
                for contour_id in selected_contours_id:
                    contour = contours[contour_id]
                    points = []
                    for point in contour:
                        points.append([point[0][0], point[0][1]])
                    max_curvature = 1.5
                    n_contsamp = 50
                    n_pntsamp = 10
                    if len(points) > 1:
                        cfs = self._extr_raw_points(np.array(points), max_curvature, n_contsamp, n_pntsamp)

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
                self.data[image_key]['cfs'] = [np.array(shapes_feature), sz]

    def _learn_codebook(self):
        MAX_CFS = 800  # max number of contour fragments per image; if above, sample randomly
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
        self.kmeans = sklearn.cluster.KMeans(min(CLUSTERING_CENTERS, feats_sc.shape[0]), n_jobs=-1,
                                             algorithm='elkan').fit(feats_sc)
        print("Saving codebook...")
        self._save_kmeans(self.kmeans)
        return self.kmeans

    def _save_kmeans(self, kmeans):
        with open(self.CODEBOOK_FILE, 'wb') as out_file:
            pickle.dump(kmeans, out_file, -1)

    def _load_kmeans(self):
        if self.kmeans is None:
            with open(self.CODEBOOK_FILE, 'rb') as in_file:
                self.kmeans = pickle.load(in_file)
        return self.kmeans

    def _encode_cf(self):
        K_NN = 5
        kmeans = self._load_kmeans()
        # Represent each contour fragment shape descriptor as a combination of K_NN of the
        # clustering centers
        for image in self.data.values():
            shapes_feature = image['cfs'][0]
            encoded_shape_features = []
            for shape_feature in shapes_feature:
                contour_feature = shape_feature[0]
                encoded_shape_features.append(
                    llc_coding_approx(kmeans.cluster_centers_, contour_feature.transpose(), K_NN))
            image["llc_coding"] = encoded_shape_features

    def _spp(self):
        PYRAMID = np.array([1, 2, 4])
        for image_key, image in self.data.items():
            feat = image['cfs']
            shapes_feature = feat[0]
            shapes_spp_feature = []
            for shape_id, shape_feature in enumerate(shapes_feature):
                shape_feas = self._pyramid_pooling(PYRAMID, feat[1], shape_feature[1], image["llc_coding"][shape_id])
                # feas = self._pyramid_pooling(PYRAMID, feat[3], feat[2], image['encoded_shape_descriptors'])
                shape_spp_fea = shape_feas.flatten()
                shape_spp_fea /= np.sqrt(np.sum(shape_spp_fea ** 2))
                shapes_spp_feature.append(shape_spp_fea)

            image['spp_descriptor'] = np.array(shapes_spp_feature)
            print(image_key)
            print(image["spp_descriptor"].shape)

    def _pyramid_pooling(self, pyramid, sz, xy, encoded_shape_descriptors):
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

    def _save_classifier(self, clf):
        with open(self.CLASSIFIER_FILE, 'wb') as out_file:
            pickle.dump(clf, out_file, -1)

    def _load_classifier(self):
        if self.clf is None:
            with open(self.CLASSIFIER_FILE, 'rb') as in_file:
                self.clf = pickle.load(in_file)
        return self.clf

    def _svm_train(self):
        self._save_label_to_class_mapping()
        clf = sklearn.svm.LinearSVC(multi_class='crammer_singer')
        training_data = []
        labels = []
        for (cls, idx) in self.data.keys():
            training_data.append(self.data[(cls, idx)]['spp_descriptor'])
            labels.append(hash(cls))
        print("Training SVM...")
        self.clf = clf.fit(training_data, labels)
        print("Saving classifier...")
        self._save_classifier(self.clf)
        return self.clf

    def show(self, image):
        cv2.imshow('image', image)
        _ = cv2.waitKey()

    def _extr_raw_points(self, c, max_value, N, nn):
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
                    pnts.append(self._sample_contour(cf, nn))
                # s += 1
        # if c.shape[0] > 1:
        pnts.append(self._sample_contour(c, nn))
        return pnts

    def _sample_contour(self, cf, nn):
        # Sample points from contour fragment
        _len = cf.shape[0]
        try:
            ii = np.round(np.arange(0, _len - 0.9999, float(_len - 1) / (nn - 1))).astype('int32')
        except ZeroDivisionError:
            print("_len:{}".format(_len))
            print("nn-1:{}".format(nn - 1))
            exit(0)

        cf = cf[ii, :]
        return cf

    def train(self):
        self._extract_cf()
        self._learn_codebook()
        # self._encode_cf()
        # self._spp()
        # self._svm_train()


if __name__ == "__main__":
    bcf = BCF()
    bcf.train()
