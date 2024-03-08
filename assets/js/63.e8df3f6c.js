(window.webpackJsonp=window.webpackJsonp||[]).push([[63],{486:function(t,a,s){"use strict";s.r(a);var r=s(65),n=Object(r.a)({},(function(){var t=this,a=t.$createElement,s=t._self._c||a;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"归一化方法"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#归一化方法"}},[t._v("#")]),t._v(" 归一化方法")]),t._v(" "),s("h2",{attrs:{id:"数据白化"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#数据白化"}},[t._v("#")]),t._v(" 数据白化")]),t._v(" "),s("p",[s("strong",[t._v("Min-Max 归一化（Min-Max Normalization）")])]),t._v(" "),s("p",[t._v("也称为"),s("strong",[t._v("离差标准化")]),t._v("，是对原始数据的线性变换，"),s("strong",[t._v("使结果值映射到[0 - 1]之间")]),t._v("。转换函数如下："),s("img",{staticStyle:{zoom:"67%"},attrs:{src:"https://s2.loli.net/2023/08/12/msriacHTyvwQ7ZX.png",alt:"image-20230812160644832"}})]),t._v(" "),s("p",[t._v("其中max为样本数据的最大值，min为样本数据的最小值。这种归一化方法比较适用在数值比较集中的情况。但是，如果max和min不稳定，很容易使得归一化结果不稳定，使得后续使用效果也不稳定，实际使用中可以用经验常量值来替代max和min。而且当有新数据加入时，可能导致max和min的变化，需要重新定义。")]),t._v(" "),s("p",[s("strong",[t._v("Z-Score 标准化（Z-Score Normalization）")])]),t._v(" "),s("p",[t._v("这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。经过处理的数据符合"),s("strong",[t._v("标准正态分布")]),t._v("，即均值为0，标准差为1，转化函数为："),s("img",{attrs:{src:"https://s2.loli.net/2023/08/12/Mhfg7QoWYICE4my.png",alt:"image-20230812160709973"}})]),t._v(" "),s("p",[t._v("其中μ是样本数据的均值（mean），σ是样本数据的标准差（std）。此外，标准化后的数据"),s("strong",[t._v("保持异常值中的有用信息")]),t._v("，使得算法对异常值不太敏感，这一点归一化就无法保证。")]),t._v(" "),s("h2",{attrs:{id:"bn"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#bn"}},[t._v("#")]),t._v(" BN")]),t._v(" "),s("h2",{attrs:{id:"ln"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ln"}},[t._v("#")]),t._v(" LN")]),t._v(" "),s("h2",{attrs:{id:"instance-norm"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#instance-norm"}},[t._v("#")]),t._v(" instance Norm")]),t._v(" "),s("h2",{attrs:{id:"gn"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#gn"}},[t._v("#")]),t._v(" GN")])])}),[],!1,null,null,null);a.default=n.exports}}]);