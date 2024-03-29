# ResNet

[ResNet论文逐段精读【论文精读】 - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv14347634?from=note)



原文链接：https://blog.csdn.net/weixin_43135178/article/details/115447031

在VGG中，卷积网络达到了19层，在GoogLeNet中，网络史无前例的达到了22层。那么，网络的精度会随着网络的层数增多而增多吗？在深度学习中，网络层数增多一般会伴着下面几个问题

> 1. 计算资源的消耗
> 2. 模型容易过拟合
> 3. 梯度消失/梯度爆炸问题的产生

问题1可以通过GPU集群来解决，对于一个企业资源并不是很大的问题；
问题2的过拟合通过采集海量数据，并配合Dropout正则化等方法也可以有效避免；
问题3通过Batch Normalization、ReLu等也可以避免。

貌似我们只要无脑的增加网络的层数，我们就能从此获益，但实验数据给了我们当头一棒。

**作者发现，随着网络层数的增加，网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，训练集loss反而会增大**。注意这并不是过拟合，因为在过拟合中训练loss是一直减小的。

当网络退化时，浅层网络能够达到比深层网络更好的训练效果，这时如果我们把低层的特征传到高层，那么效果应该至少不比浅层的网络效果差，或者说如果一个VGG-100网络在第98层使用的是和VGG-16第14层一模一样的特征，那么VGG-100的效果应该会和VGG-16的效果相同。所以，我们可以在VGG-100的98层和14层之间添加一条**直接映射（Identity Mapping）**来达到此效果。

从信息论的角度讲，由于DPI（数据处理不等式）的存在，在前向传输的过程中，**随着层数的加深，Feature Map包含的图像信息会逐层减少，而ResNet的直接映射的加入，保证了 L+1 层的网络一定比 L 层包含更多的图像信息**。基于这种使用直接映射来连接网络不同层直接的思想，残差网络应运而生。






网络深度过深，会出现梯度消失、梯度爆炸的问题



batch_normalization





映射层identity mapping （当层数增加的时候，如果浅层网络已经训练的很好了，那么深层网络一般不会差，原因就是在浅层网络上所叠加的层可以训练成映射层，即：输入是x，输出也x。把权重训练成按个样子）



`SGD`不能做到这样的事情！！！





ResNet ：Deep Residual Learning Framework；（residual剩余）



核心：residual connection；

用这个方法，可以让模型在加深的情况下，精度会更高，会由于浅层版本，而不是反而减弱

学习的是残差H（x）-x

<img src="https://s2.loli.net/2022/04/26/A2kpJmqvCa1Djit.png" alt="image-20220426110224975" style="zoom:50%;" />

可以看到X是这一层残差块的输入，也称作F(x)为残差，x为输入值，F（X）是经过第一层线性变化并激活后的输出，该图表示在残差网络中，第二层进行线性变化之后激活之前，F(x)加入了这一层输入值X，然后再进行激活后输出。在第二层输出值激活前加入X，这条路径称作shortcut连接。

<img src="https://s2.loli.net/2022/05/20/HQPYrevEISCMqlU.png" alt="image-20220520094129023" style="zoom:50%;" />





​		ResNet做的最重要的事情或者说是他的特别之处：他学习的不是目标H(x)，而是残差H(x)-x；每一层的H(x)不同

​		这样的一个好处就是：<u>只是加了一个东西进来，没有任何可以学的参数，不会增加任何的模型复杂度，也不会使计算变得更加复杂，而且这个网络跟之前一样，也是可以训练的，没有任何改变</u>

​		ResNet在浅层网络比较优秀的情况下，进一步的在此基础上进行优化！对于残差部分进行训练，在加上浅层网络原来的结果，让其结果的精度更高。



​		一个ResNet的话，它的好处就是在原有的基础上加上了浅层网络的梯度，深层的网络梯度很小没有关系，浅层网络可以进行训练，变成了加法，一个小的数加上一个大的数，相对来说梯度还是会比较大的。也就是说，不管后面新加的层数有多少，前面浅层网络的梯度始终是有用的，这就是从误差反向传播的角度来解释为什么训练的比较快

残差网络有什么好处呢？

::: tip 好处

显而易见：**因为增加了 x 项，那么该网络求 x 的偏导的时候，多了一项常数 1（对x的求导为1），所以反向传播过程，梯度连乘，也不会造成梯度消失。**

:::

残差是指预测值和观测值之间的差距。

![img](https://s2.loli.net/2022/05/20/WgBtIGkysVOYRCa.png)



最重要的就是：**残差连接**

> 输入输出形状不一样的时候怎样做残差连接
>
> - 填零
> - 投影
> - 所有的连接都做投影：就算输入输出的形状是一样的，一样可以在连接的时候做个1*1的卷积，但是输入和输出通道数是一样的，做一次投影

残差连接如何处理输入和输出的形状是不同的情况：

> - 第一个方案是在输入和输出上分别添加一些额外的0，使得这两个形状能够对应起来然后可以相加
> - 第二个方案是之前提到过的全连接怎么做投影，做到卷积上，**是通过一个叫做1x1的卷积层，这个卷积层的特点是在空间维度上不做任何东西，主要是在通道维度上做改变**。所以只要选取一个1x1的卷积使得输出通道是输入通道的两倍，这样就能将残差连接的输入和输出进行对比了。在ResNet中，如果把输出通道数翻了两倍，那么输入的高和宽通常都会被减半，所以在做1x1的卷积的时候，同样也会使步幅为2，这样的话使得高宽和通道上都能够匹配上。
>

![image-20220520104625571](https://s2.loli.net/2022/05/20/nLerOgaMZ7EVBkd.png)

<img src="https://s2.loli.net/2022/05/20/qDXw7ovlOQM2S8p.png" alt="imag -20220520104615109" style="zoom:50%;" />

![image-20220520104638122](https://s2.loli.net/2022/05/20/U5z8VJE2HNy7vLW.png)



## 代码解读

现有H(x) = F(x)+x, 只要F(x)=0,那么H(x)=x,H(x)就是恒等映射，也就是有了“什么都不做”的能力。ResNet基于这一思想提出了一种残差网络的结构，其中输入x可以传递到输出，传递的过程被称为ShortCut。
同时，下图里有两个权重层，即F(x)部分。假如“什么都不学习”是最优的，或者说H(x)=x是最优的，那么理论上来说，F(x)学习到的目标值为0即可；如果H(x)=x不是最优，那么基于神经网络强大的学习能力，F(x)可以尽可能去拟合我们期望的值。


BasicBlock
ResNet中使用的一种网络结构，在resnet18和resnet34中使用了BasicBlock：
输入输出通道数均为64，残差基础块中两个3×3卷积层参数量是：




BasicBlock类中计算了残差，该类继承了nn.Module。




```python
class BasicBlock(nn.Module):
    expansion = 1
    
def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out
```


bottleNeck
ResNet-34核心部分均使用3×3卷积层，总层数相对没那么多，对于更深的网络，作者们提出了另一种残差基础块。(在resnet50、resnet101、resnet152使用了Bottlenect构造网络.)

Bottleneck Block中使用了1×1卷积层。如输入通道数为256，1×1卷积层会将通道数先降为64，经过3×3卷积层后，再将通道数升为256。1×1卷积层的优势是在更深的网络中，用较小的参数量处理通道数很大的输入。

在Bottleneck Block中，输入输出通道数均为256，残差基础块中的参数量是：

与BasicBlock比较，使用1×1卷积层，参数量减少了。当然，使用这样的设计，也是因为更深的网络对显存和算力都有更高的要求，在算力有限的情况下，深层网络中的残差基础块应该减少算力消耗。


代码：

class Bottleneck(nn.Module):
    expansion = 4

```python
def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(inplanes, planes)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out
```





## 文章经典句子

一篇文章要成为经典，不见得一定要提出原创性的东西，很可能就是把之前的一些东西很巧妙的放在一起，能解决一个现在大家比较关心难的问题