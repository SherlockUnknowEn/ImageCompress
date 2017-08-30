# -*- coding: utf-8 -*-
# @Time    : 2017/8/28 下午6:18
# @Author  : fj
# @Site    : 
# @File    : kmeans.py
# @Software: PyCharm

import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import time


def image_to_matrix(path):
    img = Image.open(path)
    w, h = img.size
    if img.mode == 'RGB':
        shape = (h, w, 3)
    elif img.mode == 'L':
        shape = (h, w, 1)
    return np.asmatrix(img.getdata(), dtype='int'), shape


def compress(path, K, iter=3):
    X, shape = image_to_matrix(path)
    m, n = X.shape
    a = time.time()
    centroids = init_random_centroids(X, K)
    print(time.time() - a)
    for i in range(iter):
        print('iter = ' + str(i+1))
        cen_idx = find_closest_centroids(X, centroids)
        centroids = move_centroids(X, cen_idx, K)

    cen_idx = find_closest_centroids(X, centroids)
    for i in range(m):
        X[i] = centroids[int(cen_idx[i])]
    save_compressed_img(X, './compress.jpg', shape)


def init_random_centroids(X, K):
    m, n = X.shape
    sh = list(range(m))
    random.shuffle(sh)
    centroids = []
    for i in range(K):
        centroids.append(X[sh[i]])
    return (np.asarray(centroids))[:,0,:] #list 转换成Matrix



def find_closest_centroids(img, centroids):
    m, n = img.shape
    K, _ = centroids.shape
    cen_idx = np.zeros([m, 1],  dtype='int')
    for i in range(m):
        M = np.full([K, n], img[i])
        err = M - centroids
        distance = np.multiply(err, err).sum(axis=1)
        cen_idx[i] = distance.argmin() #最小列索引
    return cen_idx


def move_centroids(img, cen_idx, K):
    m, n = img.shape
    times = np.zeros([K, 1], dtype='int')
    centroids = np.zeros([K, n], dtype='int')
    for i in range(m):
        idx = int(cen_idx[i])
        centroids[idx] = centroids[idx] + img[i]
        times[idx] = times[idx] + 1
    while times.min() == 0: #避免除零异常
        times[times.argmin()] = 1
    centroids = centroids / times
    return centroids


def save_compressed_img(M, path, shape):
    img = np.reshape(np.asarray(M), shape)
    Image.fromarray(img.astype(np.uint8)).save(path)


def matrixToImage(matrix, shape):
    pass


compress('./a.jpg', K=16)
print('done...')