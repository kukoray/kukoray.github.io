# PCA

解决什么问题？

::: tip 作用

主成分分析算法（PCA）是最常用的线，。性降维方法，它的目标是通过某种线性投影，将高维的数据映射到低维的空间中，并期望在所投影的维度上数据的信息量最大（方差最大），以此使用较少的数据维度，同时保留住较多的原数据点的特性。

PCA降维的目的，就是为了在尽量保证“信息量不丢失”的情况下，对原始特征进行降维，也就是尽可能将原始特征往具有最大投影信息量的维度上进行投影。将原特征投影到这些维度上，使降维后信息量损失最小。

:::

## 主成分分析

PCA是一种统计方法，通过**正交变换**将一组可能存在相关性的变量转换为一组线性无关的变量，转换后的这组变量叫主成分。

本质上就是一种**降维**的方法



**方差最大，误差最小**



PCA本质就是找一些投影方向，使得数据在这些投影方向上的方差（投影点到原点的距离）最大；

也可以是：找一些投影方向，使得数据在这些投影方向上的高（投影点到数据点的距离）最小；

<img src="https://s2.loli.net/2022/05/04/IJhl8HOfazVw5uc.png" alt="image-20220504181528322" style="zoom:40%;" />





## 基于协方差矩阵的特征值分解算法

（1）均值归一化。计算各个特征数据均值，然后令Xj=Xj-Uj。如果特征在不同的数量级上，还需要将其除以标准差。

eigen特征的意思：



均值归一化，计算各个特征均值，然后令 。如果特征在不同的数量积上。还需要将其除以标准差计算协方差矩阵急速那协方差 

基于协方差的特征值分解是缘分分解算法去u的前k维矩阵，用Userducee表式