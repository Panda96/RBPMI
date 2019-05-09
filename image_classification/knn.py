# -*- coding:utf-8 -*-
import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        # X是NXD的数组，其中每一行代表一个样本，Y是N行的一维数组，对应X的标签
        # 最近邻分类器就是简单的记住所有的数据
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        # X是NXD的数组，其中每一行代表一个图片样本
        # 看一下测试数据有多少行
        num_test = X.shape[0]
        # 确认输出的结果类型符合输入的类型
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # 循环每一行，也就是每一个样本
        for i in range(num_test):
            # 找到和第i个测试图片距离最近的训练图片
            # 计算他们的L1距离
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # 拿到最小那个距离的索引
            Ypred[i] = self.ytr[min_index]  # 预测样本的标签，其实就是跟他最近的训练数据样本的标签
        return Ypred
