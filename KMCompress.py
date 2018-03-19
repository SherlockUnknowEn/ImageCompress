# -*- coding: utf-8 -*-
# @Time    : 2017/8/28 下午6:18
# @Author  : fj
# @Site    : 
# @File    : KMCompress.py
# @Software: PyCharm

import numpy as np
from PIL import Image
import random

class KMCompress():

    def __init__(self):
        self.centroids = None #簇中心 (K, c)
        self.data = None #压缩后的数据 (h*w, c)
        self.cen_idx = None #压缩后的数据对应簇中心的下标(h*w, 1)
        self.shape = None #原图的shape

    def compress(self, path, K=16, iter=3):
        '''
        压缩图像
        :param path: 图像路径
        :param K: K簇，压缩后图像使用K个像素点表示全图
        :param iter: 迭代计算次数
        :return: None
        '''
        X, self.shape = self.__image_to_matrix(path)
        centroids = self.__init_random_centroids(X, K)
        for i in range(iter):
            print(' ====> ' + str((i+1.0)*100 / (iter + 1)) + '%')
            cen_idx = self.__find_closest_centroids(X, centroids)
            centroids = self.__move_centroids(X, cen_idx, K)
        print(' ====> 99%')
        cen_idx = self.__find_closest_centroids(X, centroids)

        self.cen_idx = cen_idx
        self.centroids = centroids
        self.__fill_data()


    def __fill_data(self):
        m = self.cen_idx.shape[0]
        channel = self.shape[2]
        self.data = np.zeros((m, channel))
        for i in range(m):
            self.data[i] = self.centroids[int(self.cen_idx[i])]

    def __image_to_matrix(self, path):
        '''
        加载图片
        :param path:
        :return: 图像矩阵 (h * w, c)
        '''
        img = Image.open(path)
        w, h = img.size
        if img.mode == 'RGB':
            shape = (h, w, 3)
        elif img.mode == 'L':
            shape = (h, w, 1)
        return np.asmatrix(img.getdata(), dtype='float'), shape


    def __init_random_centroids(self, X, K):
        '''
        初始化簇中心
        :param X: 图像矩阵 (h*w, c)
        :param K:
        :return: 簇中心(K, c)
        '''
        m, n = X.shape
        sh = list(range(m))
        random.shuffle(sh)
        centroids = []
        for i in range(K):
            centroids.append(X[sh[i]])
        return (np.asarray(centroids))[:,0,:] #list 转换成Matrix


    def __find_closest_centroids(self, img, centroids):
        '''
        计算图像每个像素点对应的簇中心下标
        :param img:
        :param centroids: (h*w, 1)
        :return: 图像每个像素点对应的簇中心下标
        '''
        m, n = img.shape
        K, _ = centroids.shape
        cen_idx = np.zeros([m, 1],  dtype='int')
        for i in range(m):
            M = np.full([K, n], img[i])
            err = M - centroids
            distance = np.multiply(err, err).sum(axis=1)
            cen_idx[i] = distance.argmin() #最小列索引
        return cen_idx


    def __move_centroids(self, img, cen_idx, K):
        '''
        更新簇中心
        :param img:
        :param cen_idx: 图像每个像素点对应的簇中心下标
        :param K: K簇
        :return: 簇中心矩阵 (K, c)
        '''
        m, n = img.shape
        times = np.zeros([K, 1], dtype='int')
        centroids = np.zeros([K, n], dtype='float')
        for i in range(m):
            idx = int(cen_idx[i])
            centroids[idx] = centroids[idx] + img[i]
            times[idx] = times[idx] + 1
        while times.min() == 0: #避免除零异常
            times[times.argmin()] = 1
        centroids = centroids / times
        return centroids


    def get_img(self):
        '''
        返回图像文件
        :param M:
        :param shape:
        :return:
        '''
        img = np.reshape(np.asarray(self.data), self.shape)
        return Image.fromarray(img.astype(np.uint8))



    def save(self, path):
        '''
        #TODO 字典+二进制压缩存储图片
        :param path:
        :return:
        '''
        pass


    def load(self, path):
        '''
        TODO 读取压缩后的图片
        :param path:
        :return:
        '''
        pass

a = KMCompress()
a.compress('./b.jpg', iter=10)
a.get_img().save('./compress_b.jpg')
print('done...')