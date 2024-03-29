# MAE

<img src="https://pic3.zhimg.com/80/v2-ac59a762ee712410c3d08f9ec2e11aa6_720w.webp" alt="img" style="zoom: 150%;" />



**论文标题：**Masked Autoencoders Are Scalable Vision Learners

**论文地址：**[https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)

**代码地址：**[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)

**论文动机：**

![img](https://s2.loli.net/2023/05/02/G2VTf49uCrn7XOy.webp)

随着 BERT 的出现，Mask Language Modeling（MLM）的自监督学习方法逐渐进入人们的视野，这一方法在 NLP 领域中得到了广泛的应用。受到 MLM 的启发，一些工作也尝试在图像上进行 Mask Modeling（即，mask 图片的部分区域，然后对区域的内容进行重建），并且也取得了不错的效果。但目前的方法通常都采用对称的 encoder 和 decoder 结构，在 encoder 中，mask token 也需要消耗大量的计算，因此作者提出了一个非对称 encoder-decoder 的结构——masked autoencoders（MAE）。

MAE 方法很简单：mask 输入图像的随机 patch，并重建缺失的像素（上图展示了不同 mask 比率的重建结果）。它基于两个核心设计。首先，作者开发了一种非对称编码器-解码器结构，其中的编码器仅对可见的 patch 子集（不带 mask token）进行操作，而轻量级解码器则从潜在表示和 mask token 重建原始图像。其次，作者发现对输入图像的高比例（例如 75%）进行 mask 会产生一项困难且有意义的自监督任务。将这两种设计结合起来，能够高效地训练大型模型：加快训练速度（3 倍或更多）并提高精度。



## 原理简介

Masked Autoencoder是一种无监督学习算法，它可以用于数据降维、特征提取和数据重建等任务。MAE的基本思想是将输入数据进行编码，然后再进行解码，以重建原始数据。与传统的自编码器不同，MAE使用了掩码（masking）技术，以限制自编码器只能使用部分数据进行重建。

具体来说，MAE会随机地掩盖一些输入数据，只允许自编码器使用未掩盖的数据进行重建。这样做的目的是强制模型学习数据的局部结构，从而提高模型的泛化能力。MAE还可以通过使用不同的掩码方式来学习不同的数据特征，从而提高模型的表达能力。

<img src="https://pic4.zhimg.com/80/v2-4f76163d5cf81ff232992f08547dea9f_720w.webp" alt="img" style="zoom:150%;" />