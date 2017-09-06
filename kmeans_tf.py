# -*- coding: utf-8 -*-
# @Time    : 2017/9/5 下午10:44
# @Author  : fj
# @Site    : 
# @File    : kmeans_net.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import time
from PIL import Image


def TFKmeans(vect, K, iter=3):
    K = int(K)
    assert K < len(vect)
    m, n =len(vect), len(vect[0])
    init_idx = list(range(m))
    np.random.shuffle(init_idx)
    centroids = [vect[init_idx[i]] for i in range(K)]
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        op_centroids = tf.placeholder(dtype=tf.float32, shape=[K, n])
        op_cent_idx = tf.Variable(dtype=tf.int64, initial_value=init_idx)

        # 单个像素点组成的[K, n]矩阵
        v1 = tf.placeholder(dtype=tf.float32, shape=[K, n])

        # 使用v1矩阵计算单个像素点到所有簇中心的距离
        op_distances = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, op_centroids), 2), axis=1))
        op_idx = tf.argmin(op_distances, axis=0)

        # 所有像素点分簇后  重新计算出簇中心的算子
        cluster_input = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        reduce_new_cent = tf.reduce_mean(cluster_input, axis=0)

        ################################
        # 到此为止，已经构建好图中所有需要的算子
        ################################
        sess.run(tf.global_variables_initializer())

        for it in range(iter):
            print('iter = ' + str(it + 1))
            for j in range(m):
                print(j)
                v = vect[j]
                # 对于当前像素  计算它到每一个簇中心的距离
                M = np.full(shape=[K, n], fill_value=v)
                idx = sess.run(op_idx, feed_dict={v1: M, op_centroids: centroids})
                # 取距离最近的那个簇中心的索引
                # idx = sess.run(op_idx, feed_dict={each_centroid_distances: distances})
                # 将索引保存在cent_idx[j]中
                value = sess.run(op_cent_idx)
                value[j] = idx
                sess.run(tf.assign(ref=op_cent_idx, value=value))
                # sess.run(cent_idx[j], feed_dict={cent_idx_value: idx})

            #聚类完毕后, 更新centroids
            for j in range(K):
                input = [vect[i] for i in range(m) if sess.run(op_cent_idx)[i] == j]
                new_centroid_value = sess.run(reduce_new_cent, feed_dict={cluster_input: input})
                c = sess.run(op_centroids)
                c[j] = new_centroid_value
                sess.run(tf.assign(ref=op_centroids, value=c))

    return sess.run(op_centroids), sess.run(op_cent_idx)


def image_to_vect(path):
    img = np.array(Image.open(path), dtype=np.float32)
    m, n, channel = img.shape
    return img.reshape(m*n, channel), img.shape


def save_compressed_img(M, path, shape):
    img = M.reshape(shape)
    Image.fromarray(img.astype(np.uint8)).save(path)


def compress(path, K=16, iter=3):
    vect, shape = image_to_vect(path=path)
    tim = time.time()
    print(tim)
    centroids, cent_idx = TFKmeans(vect, K=16)
    print(tim - time.time())
    for i in range():
        vect[i] = centroids[int(cent_idx[i])]
    save_compressed_img(vect, path='compress.jpg', shape=shape)


compress('./a.jpg')
