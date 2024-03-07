(window.webpackJsonp=window.webpackJsonp||[]).push([[30],{451:function(t,_,v){"use strict";v.r(_);var s=v(65),e=Object(s.a)({},(function(){var t=this,_=t.$createElement,v=t._self._c||_;return v("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[v("h1",{attrs:{id:"激活函数"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#激活函数"}},[t._v("#")]),t._v(" 激活函数")]),t._v(" "),v("p",[t._v("作用：激活函数是用来加入非"),v("a",{attrs:{href:"https://so.csdn.net/so/search?q=%E7%BA%BF%E6%80%A7&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[t._v("线性"),v("OutboundLink")],1),t._v("因素的，解决线性模型所不能解决的问题")]),t._v(" "),v("blockquote",[v("p",[t._v("原因：由于机器学习做的事情其实都是线性运算（卷积、感知机、神经网络）")]),t._v(" "),v("p",[t._v("不管模型有多少层，有多深，线性变换乘以线性变换，仍然是线性变换。")])]),t._v(" "),v("p",[t._v("激活函数一般满足：")]),t._v(" "),v("blockquote",[v("ol",[v("li",[t._v("函数本身是简单的")]),t._v(" "),v("li",[t._v("函数的导函数也是简单的")]),t._v(" "),v("li",[t._v("非线性： 激活函数为非线性激活函数的时候，基本上两层的神经网络就可以模拟大多数函数。但是如果没有激活函数的时候，对多层的神经网络来说只是做到了一个基本向量空间的线性变换，这与单层的神经网络是等价的。")]),t._v(" "),v("li",[t._v("可微性： 在进行梯度优化和计算的时候，必须满足函数可微性的这一个条件以方便进行求导运算。")]),t._v(" "),v("li",[t._v("单调性： 当激活函数是单调函数的时候，单层的神经网络能够保持是凸函数。")]),t._v(" "),v("li",[t._v("f ( x ) ≈ x： 激活函数满足这个值主要为的是设置参数的初始化，以提高神经网络的训练效果。")]),t._v(" "),v("li",[t._v("输出的范围： 激活函数输出范围值是一个重要的参数，"),v("strong",[t._v("当输出的范围是有限的时候，基于梯度的优化方法会更加稳定")]),t._v("，因为特征量表示受到有限权值的影响会更加显著。当输出范围是一个无限值的时候，模型训练更加有效果。")])])]),t._v(" "),v("p",[v("img",{attrs:{src:"https://img-blog.csdnimg.cn/20190719100438779.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5ODMxMTYz,size_16,color_FFFFFF,t_70",alt:"img"}})]),t._v(" "),v("ul",[v("li",[v("strong",[t._v("饱和")]),t._v("激活函数： sigmoid、 tanh")]),t._v(" "),v("li",[v("strong",[t._v("非饱和")]),t._v("激活函数: ReLU 、Leaky Relu  、ELU【指数线性单元】、PReLU【"),v("strong",[t._v("参数化的")]),t._v("ReLU 】、RReLU【随机ReLU】")])]),t._v(" "),v("p",[t._v("相对于饱和激活函数，使用“"),v("strong",[t._v("非饱和激活函数”的优势")]),t._v("在于两点：\n1.首先，“非饱和激活函数”能解决深度神经网络【层数非常多！！】的“"),v("strong",[t._v("梯度消失”问题")]),t._v("，浅层网络【三五层那种】才用"),v("a",{attrs:{href:"https://so.csdn.net/so/search?q=sigmoid&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[t._v("sigmoid"),v("OutboundLink")],1),t._v(" 作为激活函数。\n2.其次，它能"),v("strong",[t._v("加快收敛速度")]),t._v("。")]),t._v(" "),v("h2",{attrs:{id:"sigmoid"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#sigmoid"}},[t._v("#")]),t._v(" sigmoid")]),t._v(" "),v("p",[t._v("饱和时梯度值非常小。由于BP算法反向传播的时候后层的梯度是以乘性方式传递到前层，因此当层数比较多的时候，传到前层的梯度就会非常小，网络权值得不到有效的更新，即梯度耗散。如果该层的权值初始化使得f(x) 处于饱和状态时，网络基本上权值无法更新。")]),t._v(" "),v("p",[t._v("目前已被淘汰，只适用于浅层网络。")]),t._v(" "),v("p",[t._v("tanh特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果显示出来，但有是，在特征相差比较复杂或是相差不是特别大时，需要更细微的分类判断的时候，sigmoid效果就好了。\n还有一个东西要注意，sigmoid 和 tanh作为激活函数的话，一定要注意一定要对 input 进行归一话，否则激活后的值都会进入"),v("strong",[t._v("平坦区")]),t._v("，使隐层的输出全部趋同，但是 ReLU 并不需要输入归一化来防止它们达到饱和。")]),t._v(" "),v("h2",{attrs:{id:"tanh"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#tanh"}},[t._v("#")]),t._v(" tanh")]),t._v(" "),v("p",[t._v("tanh与sigmoid本质上是一样的，也存在饱和的问题。")]),t._v(" "),v("h2",{attrs:{id:"softmax"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#softmax"}},[t._v("#")]),t._v(" softmax")]),t._v(" "),v("p",[t._v("一般也不叫他激活函数，可以叫做输出函数。")]),t._v(" "),v("p",[t._v("多用于多分类问题，在最后一层使用，用于计算输出"),v("strong",[t._v("每个结果的概率")]),t._v("。")]),t._v(" "),v("p",[v("img",{attrs:{src:"https://img-blog.csdn.net/20161219175058506",alt:"img"}})]),t._v(" "),v("h2",{attrs:{id:"relu"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#relu"}},[t._v("#")]),t._v(" ReLu")]),t._v(" "),v("div",{staticClass:"custom-block tip"},[v("p",{staticClass:"custom-block-title"},[t._v("优点")]),t._v(" "),v("p",[t._v("第一，采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多。")]),t._v(" "),v("p",[t._v("第二，对于深层网络，sigmoid函数反向传播时，很容易就会出现梯度消失的情况（在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失，，从而无法完成深层网络的训练。")]),t._v(" "),v("p",[t._v("第三，ReLu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。")])]),t._v(" "),v("p",[t._v("构建稀疏矩阵，也就是稀疏性，这个特性可以去除数据中的冗余，最大可能保留数据的特征，也就是大多数为0的稀疏矩阵来表示。其实这个特性主要是对于Relu，它就是取的max(0,x)，因为神经网络是不断反复计算，实际上变成了它在尝试不断试探如何用一个大多数为0的矩阵来尝试表达数据特征，结果因为稀疏特性的存在，反而这种方法变得运算得又快效果又好了。所以我们可以看到目前大部分的卷积神经网络中，基本上都是采用了ReLU 函数。")]),t._v(" "),v("p",[v("strong",[t._v("深度学习目前最常用的激活函数")])]),t._v(" "),v("p",[t._v("与Sigmoid/tanh函数相比，ReLu激活函数的优点是：")]),t._v(" "),v("ul",[v("li",[t._v("使用梯度下降（GD）法时，收敛速度更快")]),t._v(" "),v("li",[t._v("相比Relu只需要一个门限值，即可以得到激活值，计算速度更快")])]),t._v(" "),v("p",[t._v("缺点是：Relu的输入值为负的时候，输出始终为0，其一阶导数也始终为0，这样会导致神经元不能更新参数，也就是神经元不学习了，这种现象叫做“Dead Neuron”。")]),t._v(" "),v("p",[t._v("如果后层的某一个梯度特别大，导致W更新以后变得特别大，导致该层的输入<0，输出为0，这时该层就会‘die’，没有更新。当学习率比较大时可能会有40%的神经元都会在训练开始就‘die’，因此需要对学习率进行一个好的设置。")]),t._v(" "),v("p",[t._v("为了解决Relu函数这个缺点，在Relu函数的负半区间引入一个泄露（Leaky）值，所以称为Leaky Relu函数。")])])}),[],!1,null,null,null);_.default=e.exports}}]);