(window.webpackJsonp=window.webpackJsonp||[]).push([[57],{476:function(t,s,a){"use strict";a.r(s);var n=a(65),r=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"cnn卷积神经网络"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cnn卷积神经网络"}},[t._v("#")]),t._v(" CNN卷积神经网络")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/05/17/tzLbE6uHT1QGM7D.png",alt:"这里写图片描述"}})]),t._v(" "),a("h2",{attrs:{id:"cnn"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cnn"}},[t._v("#")]),t._v(" CNN")]),t._v(" "),a("p",[t._v("卷积层还有一个特性就是“权值共享”原则。")]),t._v(" "),a("h3",{attrs:{id:"卷积层"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#卷积层"}},[t._v("#")]),t._v(" 卷积层")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2023/07/24/GBgez9ambAVN713.webp",alt:"img"}})]),t._v(" "),a("p",[t._v("卷积层：输入的数据也叫特征feature map，维度一般是 channel * long * width 例如 256x224x224")]),t._v(" "),a("p",[t._v("卷积层输出的数据特征维度是  512x111x111")]),t._v(" "),a("p",[t._v("其中这里的512是指filter的个数，这里的filter在多通道的卷积中是卷积核的集合，例如此处，一个filter的大小就是256x3x3")]),t._v(" "),a("p",[t._v("个数是512个，所以整个卷积层的参数是 512x256x3x3")]),t._v(" "),a("p",[a("strong",[t._v("卷积的计算公式")])]),t._v(" "),a("p",[t._v("卷积神将网络的计算公式为：\n"),a("strong",[t._v("N=(W-F+2P)/S+1")])]),t._v(" "),a("blockquote",[a("p",[t._v("其中N：输出大小\nW：输入大小\nF："),a("a",{attrs:{href:"https://so.csdn.net/so/search?q=%E5%8D%B7%E7%A7%AF%E6%A0%B8&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[t._v("卷积核"),a("OutboundLink")],1),t._v("大小\nP：填充值的大小\nS：步长大小")])]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2023/07/24/oQLzIy1xewZ3b6F.png",alt:"image-20230724170016656"}})]),t._v(" "),a("h3",{attrs:{id:"池化层"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#池化层"}},[t._v("#")]),t._v(" 池化层")]),t._v(" "),a("p",[t._v("也叫下采样层，缩小feature map 的大小")]),t._v(" "),a("p",[a("strong",[t._v("该层没有任何参数")]),t._v("，只有size（3x3）和stride（步长为2）；")]),t._v(" "),a("p",[t._v("常用:")]),t._v(" "),a("blockquote",[a("p",[t._v("maxPooling：最大池化，把3*3的区域里最大值作为该区域的代表，起到突出的效果")]),t._v(" "),a("p",[t._v("averagePooling：平均池化，把3*3区域的平均值作为该区域代表，起到模糊的效果")])]),t._v(" "),a("p",[t._v("图中不同颜色代表不同的特征，需要学习对应数量的卷积核进行"),a("a",{attrs:{href:"https://so.csdn.net/so/search?q=%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[t._v("特征提取"),a("OutboundLink")],1),t._v("。")]),t._v(" "),a("p",[t._v("对于"),a("a",{attrs:{href:"https://so.csdn.net/so/search?q=%E7%81%B0%E5%BA%A6%E5%9B%BE%E5%83%8F&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[t._v("灰度图像"),a("OutboundLink")],1),t._v("，图像为2D\n例如一个图像大小是5×5，\n有一个3×3的卷积核对着图像进行卷积，步长为1，卷积结束后生成一个3×3的矩阵。\n如果有2组卷积核对着图像卷积，就会生成2个3×3的矩阵。\n"),a("strong",[t._v("同理有多少组卷积核对图像卷积就有多少个矩阵。")]),t._v("\n这个叫做通道。")]),t._v(" "),a("p",[t._v("对于RGB图像，图像为3维\n若要提取2个特征，可以设置2个3维卷积核进行特征提取，提取结果为2通道的feature map，2个通道互相独立，代表着不同卷积核提取的不同特征。")]),t._v(" "),a("img",{staticStyle:{zoom:"70%"},attrs:{src:"https://img-blog.csdnimg.cn/20210715191700154.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2tyaXNfcGF1bA==,size_16,color_FFFFFF,t_70",alt:"sad"}}),t._v(" "),a("p",[a("a",{attrs:{href:"https://cs231n.github.io/assets/conv-demo/index.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("上图的动图链接"),a("OutboundLink")],1)]),t._v(" "),a("p",[a("strong",[t._v("一般来说一个卷积核对应着一个特征的提取")]),t._v("（例如：一个卷积核用来提取边缘特征，另外一个卷积核用来提取x方向的边缘特征等）")]),t._v(" "),a("p",[a("strong",[t._v("进行卷积处理的卷积通道数默认和输入图像的通道数相等。")]),t._v("\n比如输入图像维度为256，进行特征提取的卷积核也默认是256维。\n若设定输出64个特征，那么就一共有64个256维的卷积核用来提取特征，即提取特征的输出通道数为64，输出64个feature map。")]),t._v(" "),a("img",{staticStyle:{zoom:"33%"},attrs:{src:"https://img-blog.csdnimg.cn/20210715191734669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2tyaXNfcGF1bA==,size_16,color_FFFFFF,t_70",alt:"im g"}}),t._v(" "),a("h2",{attrs:{id:"_3dcnn"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_3dcnn"}},[t._v("#")]),t._v(" 3DCNN")]),t._v(" "),a("h3",{attrs:{id:"硬线层"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#硬线层"}},[t._v("#")]),t._v(" 硬线层")]),t._v(" "),a("p",[t._v("每帧提取5个通道信息（gray、gradient_X、gradient_Y、optflow_X、optflow_Y）")]),t._v(" "),a("h2",{attrs:{id:"shortcut"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#shortcut"}},[t._v("#")]),t._v(" Shortcut")]),t._v(" "),a("p",[t._v("残差块，也叫skip connect，")]),t._v(" "),a("h2",{attrs:{id:"cnn常见问题"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#cnn常见问题"}},[t._v("#")]),t._v(" CNN常见问题")]),t._v(" "),a("h3",{attrs:{id:"_1-1卷积为什么能降维"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-1卷积为什么能降维"}},[t._v("#")]),t._v(" 1*1卷积为什么能降维？")]),t._v(" "),a("p",[t._v("背景：最早出现在 Network In Network的论文中 ，使用1*1卷积是想"),a("strong",[t._v("加深加宽网络结构")]),t._v(" 。")]),t._v(" "),a("p",[t._v("所谓1x1默认是w和h上的1x1，但对于高维度，其实应该是这样：就是H和W不变，而是channel这个维度上降维，如图对于channel与原三维矩阵相同的1x1卷积核，直接channel就给干到了1维，而原来是32维。")]),t._v(" "),a("p",[t._v("1*1卷积的主要作用有以下几点：")]),t._v(" "),a("p",[t._v("1、降维。比如，"),a("strong",[t._v("一张500 x 500且厚度depth为100 的图片在20个filter上做1x1的卷积，那么结果的大小为500x500x20。")])]),t._v(" "),a("p",[t._v("2、加入非线性。卷积层之后经过激励层，1*1的卷积在前一层的学习表示上添加了非线性激励，提升网络的表达能力；")]),t._v(" "),a("p",[t._v("3、增加模型深度。可以减少网络模型参数，增加网络层深度，一定程度上提升模型的表征能力。")]),t._v(" "),a("h3",{attrs:{id:"网络退化问题"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#网络退化问题"}},[t._v("#")]),t._v(" 网络退化问题？")]),t._v(" "),a("p",[t._v("​\t\t举个例子，假设已经有了一个最优化的网络结构，是18层。当我们设计网络结构的时候，我们并不知道具体多少层次的网络是最优化的网络结构，假设设计了34层网络结构。那么多出来的16层其实是冗余的，我们希望训练网络的过程中，模型能够自己训练这五层为恒等映射，也就是经过这层时的输入与输出完全一样。")]),t._v(" "),a("p",[t._v("​\t\t但是往往模型很难将这16层恒等映射的参数学习正确，那么就一定会不比最优化的18层网络结构性能好，这就是"),a("strong",[t._v("随着网络深度增加，模型会产生退化现象")]),t._v("。它不是由过拟合产生的，而是由"),a("strong",[t._v("冗余的网络层学习了不是恒等映射的参数")]),t._v("造成的。")]),t._v(" "),a("h3",{attrs:{id:"为什么残差连接能解决网络退化问题"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#为什么残差连接能解决网络退化问题"}},[t._v("#")]),t._v(" 为什么残差连接能解决网络退化问题？")]),t._v(" "),a("p",[t._v("我们发现，要想让该冗余层能够恒等映射，我们只需要学习F(x)=0。"),a("strong",[t._v("学习F(x)=0比学习h(x)=x要简单，因为一般每层网络中的参数初始化偏向于0")]),t._v("，这样在相比于更新该网络层的参数来学习h(x)=x，该冗余层学习F(x)=0的更新参数能够更快收敛")]),t._v(" "),a("p",[t._v("并且ReLU能够将负数激活为0，过滤了负数的线性变化，也能够更快的使得F(x)=0。这样当网络自己决定哪些网络层为冗余层时，使用ResNet的网络很大程度上解决了学习恒等映射的问题，用学习残差F(x)=0更新该冗余层的参数来代替学习h(x)=x更新冗余层的参数。")]),t._v(" "),a("h3",{attrs:{id:"高斯金字塔"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#高斯金字塔"}},[t._v("#")]),t._v(" 高斯金字塔")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/05/25/jnNoLmtCz3S9WdO.jpg",alt:"img"}})]),t._v(" "),a("p",[t._v("高斯金字塔，本质就是在原图片的基础上，进行高斯模糊（一个滤波器，filter，其实就是一个卷积核），然后进行2*2的下采样。")]),t._v(" "),a("p",[t._v("得到了同一张图片不同尺度的子图。")]),t._v(" "),a("h3",{attrs:{id:"拉普拉斯金字塔"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#拉普拉斯金字塔"}},[t._v("#")]),t._v(" 拉普拉斯金字塔")]),t._v(" "),a("p",[t._v("拉普拉斯金字塔可以认为就是一个残差金字塔。")]),t._v(" "),a("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://s2.loli.net/2022/05/26/Kmf25XAeBlVhbtW.png",alt:"img"}}),t._v(" "),a("p",[t._v("我们知道一张图片进行下采样后，在进行上采样，图片是没办法恢复的一模一样的，也就是说下采样是一个不可逆的过程。")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://pic2.zhimg.com/80/v2-cd004410f46aa5657a946568ff403251_720w.jpg",alt:"im g"}})]),t._v(" "),a("p",[t._v("可以看出，原始图片下采样后得到的小尺寸图片虽然保留了视觉效果，但是将该小尺寸图像再次上采样也不能完整的恢复出原始图像。为了能够从下采样图像Down(Gi)中还原原始图像Gi，我们需要记"),a("strong",[t._v("录再次上采样得到Up(Down(Gi))与原始图片Gi之间的差异")]),t._v("，这就是拉普拉斯金字塔的核心思想")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-1641deeb3eec372b6ff3fc436c8651b6_720w.jpg",alt:"img"}})]),t._v(" "),a("p",[t._v("下面的就是"),a("strong",[t._v("拉普拉斯金字塔")])]),t._v(" "),a("p",[a("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-d88c440419db98a262482d31b4a19e22_720w.jpg",alt:"img"}})]),t._v(" "),a("h2",{attrs:{id:"_1特征提取"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1特征提取"}},[t._v("#")]),t._v(" 1特征提取")]),t._v(" "),a("h3",{attrs:{id:"_1-1形状特征"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-1形状特征"}},[t._v("#")]),t._v(" 1.1形状特征")]),t._v(" "),a("h4",{attrs:{id:"_1-1-1-hog"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-1-1-hog"}},[t._v("#")]),t._v(" 1.1.1 HOG")]),t._v(" "),a("p",[t._v("HOG主要是用于提取图片的一个形状特征，经常用HOG+SVM的方式来进行行人检测。")]),t._v(" "),a("h4",{attrs:{id:"_1-1-2-sift"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-1-2-sift"}},[t._v("#")]),t._v(" 1.1.2 SIFT")]),t._v(" "),a("h4",{attrs:{id:"_1-1-3-harris"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-1-3-harris"}},[t._v("#")]),t._v(" 1.1.3 Harris")]),t._v(" "),a("h3",{attrs:{id:"_1-2纹理特征"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-2纹理特征"}},[t._v("#")]),t._v(" 1.2纹理特征")]),t._v(" "),a("h4",{attrs:{id:"_1-2-1-lbp"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-2-1-lbp"}},[t._v("#")]),t._v(" 1.2.1 LBP")]),t._v(" "),a("h3",{attrs:{id:"_1-3-颜色特征"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-3-颜色特征"}},[t._v("#")]),t._v(" 1.3 颜色特征")]),t._v(" "),a("h3",{attrs:{id:"_1-4-空间关系特征"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-4-空间关系特征"}},[t._v("#")]),t._v(" 1.4 空间关系特征")]),t._v(" "),a("h2",{attrs:{id:"dw卷积"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dw卷积"}},[t._v("#")]),t._v(" DW卷积")]),t._v(" "),a("p",[t._v("也叫做深度可分离卷积，Separable Convolution的卷积运算方式。它将传统卷积分解为"),a("strong",[t._v("Depthwise Convolution")]),t._v("与"),a("strong",[t._v("Pointwise Convolution")]),t._v("两部分，有效的减小了参数数量。")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://yinguobing.com/separable-convolution/#fn2",target:"_blank",rel:"noopener noreferrer"}},[t._v("卷积神经网络中的Separable Convolution (yinguobing.com)"),a("OutboundLink")],1)]),t._v(" "),a("p",[t._v("相同的输入，同样是得到4张Feature map，Separable Convolution的参数个数是常规卷积的约1/3。因此，在参数量相同的前提下，采用Separable Convolution的神经网络层数可以做的更深。")]),t._v(" "),a("h2",{attrs:{id:"卷积基本训练代码"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#卷积基本训练代码"}},[t._v("#")]),t._v(" 卷积基本训练代码")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#!/usr/bin/env python")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# -*- coding: UTF-8 -*-")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token triple-quoted-string string"}},[t._v("'''\n@Project ：slimmable_networks \n@File    ：train_BN.py\n@Author  ：Jacky\n@Date    ：2023-06-01 20:09 \n'''")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nn "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" nn\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("optim "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" optim\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torchvision"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("datasets "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" datasets\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torchvision"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("transforms "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" transforms\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" time\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" ssl\nssl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("_create_default_https_context "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" ssl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("_create_unverified_context\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义卷积神经网络")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Net")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Module"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("__init__")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("super")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("Net"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("__init__"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("conv1 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Conv2d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("in_channels"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" out_channels"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("32")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" kernel_size"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" padding"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("bn1 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("BatchNorm2d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("32")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu1 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ReLU"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("inplace"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("conv2 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Conv2d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("in_channels"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("32")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" out_channels"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" kernel_size"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" padding"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("bn2 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("BatchNorm2d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu2 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ReLU"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("inplace"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("pool "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("MaxPool2d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("kernel_size"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" stride"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("fc1 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Linear"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("16")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("16")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("128")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu3 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ReLU"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("inplace"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("fc2 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Linear"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("128")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("forward")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("conv1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("bn1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("conv2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("bn2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("pool"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("view"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("16")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("16")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("fc1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("relu3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("fc2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" x\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 加载CIFAR-10数据集")]),t._v("\ntransform "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" transforms"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Compose"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("transforms"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ToTensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" transforms"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Normalize"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ntrainset "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" datasets"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("CIFAR10"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("root"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'./data/cifar10'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" train"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" download"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" transform"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("transform"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ntrainloader "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("utils"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("DataLoader"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("trainset"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" batch_size"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("64")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" shuffle"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义模型、损失函数和优化器")]),t._v("\nnet "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Net"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\ncriterion "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("CrossEntropyLoss"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\noptimizer "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" optim"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("SGD"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("net"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("parameters"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" lr"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.01")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" momentum"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.9")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 训练模型")]),t._v("\nstart_time "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" time"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("time"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),t._v(" epoch "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("in")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("range")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("10")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    running_loss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.0")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),t._v(" i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" data "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("in")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("enumerate")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("trainloader"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        inputs"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" labels "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" data\n        optimizer"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("zero_grad"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        outputs "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" net"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("inputs"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        loss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" criterion"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("outputs"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" labels"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        loss"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("backward"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        optimizer"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("step"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        running_loss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+=")]),t._v(" loss"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("item"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("if")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("%")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("==")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("99")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'[%d, %5d] loss: %.3f'")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("%")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("epoch "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" running_loss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("100")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n            running_loss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.0")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'Finished Training. Time taken:'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" time"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("time"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v(" start_time"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'seconds'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n")])])])])}),[],!1,null,null,null);s.default=r.exports}}]);