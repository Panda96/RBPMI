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


class BCF:
    def __init__(self):
        self.DATA_DIR = "data/cuauv"
        self.PERC_TRAINING_PER_CLASS = 0.5
        self.CODEBOOK_FILE = "codebook.data"
        self.CLASSIFIER_FILE = "classifier"
        self.LABEL_TO_CLASS_MAPPING_FILE = "labels_to_classes.data"
        self.classes = defaultdict(list)
        self.data = defaultdict(dict)
        self.counter = defaultdict(int)
        self.kmeans = None
        self.clf = None
        self.label_to_class_mapping = None

    def _load_classes(self):
        for dir_name, subdir_list, file_list in os.walk(self.DATA_DIR):
            if subdir_list:
                continue
            class_key = dir_name.split(os.path.sep)[-1]
            for f in sorted(file_list, key=hash):
                self.classes[class_key].append(os.path.join(dir_name, f))

    def _load_training(self):
        for cls in self.classes:
            images = self.classes[cls]
            for image in images[:int(len(images) * self.PERC_TRAINING_PER_CLASS)]:
                image_id = self._get_image_identifier(cls)
                self.data[image_id]['image'] = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if self.data[image_id]['image'] is None:
                    print("Failed to load " + image)

    def _load_testing(self):
        for cls in self.classes:
            images = self.classes[cls]
            for image in images[int(len(images) * self.PERC_TRAINING_PER_CLASS):]:
                image_id = self._get_image_identifier(cls)
                self.data[image_id]['image'] = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                if self.data[image_id]['image'] is None:
                    print("Failed to load " + image)

    def _load_single(self, image):
        # Load single image data
        self.data.clear()
        image_id = self._get_image_identifier(None)
        self.data[image_id]['image'] = image

    def _save_label_to_class_mapping(self):
        self.label_to_class_mapping = {hash(cls): cls for cls in self.classes}
        with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'wb') as out_file:
            pickle.dump(self.label_to_class_mapping, out_file, -1)

    def _load_label_to_class_mapping(self):
        if self.label_to_class_mapping is None:
            with open(self.LABEL_TO_CLASS_MAPPING_FILE, 'rb') as in_file:
                self.label_to_class_mapping = pickle.load(in_file)
        return self.label_to_class_mapping

    def _normalize_shapes(self):
        for (cls, idx) in self.data.keys():
            image = self.data[(cls, idx)]['image']
            # Remove void space
            y, x = np.where(image > 50)
            max_y = y.max()
            min_y = y.min()
            max_x = x.max()
            min_x = x.min()
            trimmed = image[min_y:max_y, min_x:max_x] > 50
            trimmed = trimmed.astype('uint8')
            trimmed[trimmed > 0] = 255
            self.data[(cls, idx)]['normalized_image'] = trimmed

    def _extract_cf(self):
        for (cls, idx) in self.data.keys():
            image = self.data[(cls, idx)]['normalized_image']
            _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = sorted(contours, key=len)[-1]
            mat = np.zeros(image.shape, np.int8)
            cv2.drawContours(mat, [contour], -1, (255, 255, 255))
            # self.show(mat)
            MAX_CURVATURE = 1.5
            N_CONTSAMP = 50
            N_PNTSAMP = 10
            C = None
            for pnt in contour:
                if C is None:
                    C = np.array([[pnt[0][0], pnt[0][1]]])
                else:
                    C = np.append(C, [[pnt[0][0], pnt[0][1]]], axis=0)
            cfs = self._extr_raw_points(C, MAX_CURVATURE, N_CONTSAMP, N_PNTSAMP)

            # self.show(mat)
            # for cf in cfs:
            #     tmp = np.zeros(image.shape, np.int8)
            #     for pnt in cf:
            #         cv2.circle(tmp, (pnt[0], pnt[1]), 2, (255, 0, 0))
            #     self.show(tmp)
            num_cfs = len(cfs)
            print("Extracted %s points" % (num_cfs))
            feat_sc = np.zeros((300, num_cfs))
            xy = np.zeros((num_cfs, 2))

            for i in range(num_cfs):
                cf = cfs[i]
                sc, _, _, _ = shape_context(cf)
                # shape context is 60x5 (60 bins at 5 reference points)
                sc = sc.flatten(order='F')
                sc /= np.sum(sc)  # normalize
                feat_sc[:, i] = sc
                # shape context descriptor sc for each cf is 300x1
                # save a point at the midpoint of the contour fragment
                xy[i, 0:2] = cf[np.round(len(cf) / 2. - 1).astype('int32'), :]
            sz = image.shape
            self.data[(cls, idx)]['cfs'] = (cfs, feat_sc, xy, sz)

    def _learn_codebook(self):
        MAX_CFS = 800  # max number of contour fragments per image; if above, sample randomly
        CLUSTERING_CENTERS = 1500
        feats_sc = []
        for image in self.data.values():
            feats = image['cfs']
            feat_sc = feats[1]
            if feat_sc.shape[1] > MAX_CFS:
                # Sample MAX_CFS from contour fragments
                rand_indices = np.random.permutation(feat_sc.shape[1])
                feat_sc = feat_sc[:, rand_indices[:MAX_CFS]]
            feats_sc.append(feat_sc)

        feats_sc = np.concatenate(feats_sc, axis=1).transpose()
        # print("feats_sc size:{}".format(len(feats_sc)))
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
            feat_sc = image['cfs'][1]
            a = llc_coding_approx(kmeans.cluster_centers_, feat_sc.transpose(), K_NN)
            # print("-" * 50)
            # for b in a:
            #     print(len(list(b)))
            # print(a)
            image['encoded_shape_descriptors'] = llc_coding_approx(kmeans.cluster_centers_, feat_sc.transpose(), K_NN)

    def _spp(self):

        PYRAMID = np.array([1, 2, 4])
        for image in self.data.values():
            feat = image['cfs']
            feas = self._pyramid_pooling(PYRAMID, feat[3], feat[2], image['encoded_shape_descriptors'])
            fea = feas.flatten()
            fea /= np.sqrt(np.sum(fea ** 2))
            image['spp_descriptor'] = fea

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

    def _svm_classify_test(self):
        clf = self._load_classifier()
        label_to_cls = self._load_label_to_class_mapping()
        testing_data = []
        labels = []
        for (cls, idx) in self.data.keys():
            testing_data.append(self.data[(cls, idx)]['spp_descriptor'])
            labels.append(hash(cls))
        predictions = clf.predict(testing_data)
        correct = 0
        for (i, label) in enumerate(labels):
            if predictions[i] == label:
                correct += 1
            else:
                print("Mistook %s for %s" % (label_to_cls[label], label_to_cls[predictions[i]]))
        print(
            "Correct: %s out of %s (Accuracy: %.2f%%)" % (correct, len(predictions), 100. * correct / len(predictions)))

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
        n_cf = (n_kp - 1) * n_kp + 1
        pnts = [None] * n_cf

        s = 0
        for i in range(n_kp):
            for j in range(n_kp):
                if i == j:
                    continue
                if i < j:
                    cf = c[i_kp[i]:i_kp[j] + 1, :]
                if i > j:
                    cf = np.append(c[i_kp[i]:, :], c[:i_kp[j] + 1, :], axis=0)
                pnts[s] = self._sample_contour(cf, nn)
                s += 1
        pnts[s] = self._sample_contour(c, nn)
        return pnts

    def _sample_contour(self, cf, nn):
        # Sample points from contour fragment
        _len = cf.shape[0]
        ii = np.round(np.arange(0, _len - 0.9999, float(_len - 1) / (nn - 1))).astype('int32')
        cf = cf[ii, :]
        return cf

    def _next_count(self, cls):
        self.counter[cls] += 1
        return self.counter[cls]

    def _get_image_identifier(self, cls):
        return (cls, self._next_count(cls))

    def _predict(self):
        with open(self.CLASSIFIER_FILE, 'rb') as in_file:
            clf = pickle.load(in_file)
        label_to_cls = self._load_label_to_class_mapping()
        testing_data = []
        for (cls, idx) in self.data.keys():
            testing_data.append(self.data[(cls, idx)]['spp_descriptor'])
        predictions = clf.predict(testing_data)
        return [label_to_cls[label] for label in predictions]

    def classify_single(self, image):
        '''
        Classifies a single image
        Example usage:
            mat = cv2.imread("data/cuauv/lightning/lightning-10.jpg", cv2.IMREAD_GRAYSCALE)
            print(bcf.classify_single(mat))
        '''
        self._load_single(image)
        self._normalize_shapes()
        self._extract_cf()
        self._encode_cf()
        self._spp()
        return self._predict()[0]

    def train(self):
        self._load_classes()
        self._load_training()
        self._normalize_shapes()
        self._extract_cf()
        self._learn_codebook()
        self._encode_cf()
        self._spp()
        self._svm_train()

    def test(self):
        self._load_classes()
        self._load_testing()
        self._normalize_shapes()
        self._extract_cf()
        self._encode_cf()
        self._spp()
        self._svm_classify_test()


if __name__ == "__main__":
    bcf = BCF()
    action = "train"
    img = "data/cuauv/lightning/lightning-22.jpg"
    if action == "train":
        print("Training mode")
        bcf.train()
    elif action == "test":
        print("Testing mode")
        bcf.test()
    elif action == "single":
        print("Single classification mode")
        mat = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if mat is None:
            print("Failed to load: " + img)
            sys.exit(1)
        print(bcf.classify_single(mat))
    else:
        print("Usage: bcf.py [train | test | single <image file>]")
        sys.exit(1)
