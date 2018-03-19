# ImageCompress
K-means聚类算法实现图片的压缩，Python实现

### 更新
* 2018-03-19 修改代码格式，封装成对象
* 2017-09-06 增加使用TensorFlow实现（实测速度比纯python+numpy慢）

### 参数
* K 全图用多少种颜色表示，默认值 K=16，设置K后将会对所有图像进行聚类，并只使用K种颜色表示全图
* iter 迭代计算次数, 默认值 iter=3

### 未实现
* 未实现压缩后的存储（降低图像文件大小），只能存为一般jpg格式预览压缩后效果

### 效果
默认参数下的效果

![第一张图片](https://github.com/SherlockUnknowEn/ImageCompress/blob/master/a.jpg "第一张图片")

![第一张图片压缩](https://github.com/SherlockUnknowEn/ImageCompress/blob/master/compress_a.jpg "第一张图片压缩")

![第二张图片](https://github.com/SherlockUnknowEn/ImageCompress/blob/master/b.jpg "第二张图片")

![第二张图片压缩](https://github.com/SherlockUnknowEn/ImageCompress/blob/master/compress_b.jpg "第二张图片压缩")