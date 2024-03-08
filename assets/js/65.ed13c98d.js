(window.webpackJsonp=window.webpackJsonp||[]).push([[65],{484:function(t,_,v){"use strict";v.r(_);var a=v(65),r=Object(a.a)({},(function(){var t=this,_=t.$createElement,v=t._self._c||_;return v("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[v("h1",{attrs:{id:"eulerian-heartrate-detection"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#eulerian-heartrate-detection"}},[t._v("#")]),t._v(" eulerian heartrate detection")]),t._v(" "),v("h2",{attrs:{id:"提取心率的原理"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#提取心率的原理"}},[t._v("#")]),t._v(" 提取心率的原理：")]),t._v(" "),v("p",[t._v("人的心率是和血液容积变化是一致的，同步的。")]),t._v(" "),v("p",[t._v("而光照经过皮肤表面，会被血液和皮肤吸收，导致反射的光束的光强发生变化。血液容积的变化与光强度的变化是呈正比的。")]),t._v(" "),v("p",[t._v("所以可以通过面部PPGI信号的提取来检测出人的心率变化。")]),t._v(" "),v("p",[t._v("但是由于PPGI信号比较微弱，所以我们需要进行颜色增强，将其特征变得更加明显；")]),t._v(" "),v("p",[t._v("而且PPGI中由AC信号、DC信号、静态基线信号构成，"),v("strong",[t._v("AC信号")]),t._v("才是我们需要去表征心率的信号。")]),t._v(" "),v("p",[t._v("PPGI的信噪比SNR比较低，所以需要进行"),v("strong",[t._v("去噪")]),t._v("。")]),t._v(" "),v("h2",{attrs:{id:"基于图像的心率信号提取"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#基于图像的心率信号提取"}},[t._v("#")]),t._v(" 基于图像的心率信号提取")]),t._v(" "),v("p",[t._v("对于图像的信号处理主要有以下两种方式：盲源分离、欧拉增强。")]),t._v(" "),v("h3",{attrs:{id:"_1盲源分离"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#_1盲源分离"}},[t._v("#")]),t._v(" ①盲源分离")]),t._v(" "),v("h3",{attrs:{id:"_2欧拉增强evm"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#_2欧拉增强evm"}},[t._v("#")]),t._v(" ②"),v("strong",[t._v("欧拉增强")]),t._v("EVM")]),t._v(" "),v("p",[t._v("原理：利用一阶泰勒展开来逼近信号（Ps：本文多加了 "),v("strong",[t._v("ROI区域提取、和颜色空间转换")]),t._v("）")]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/DIWtrL2NwxKUvVm.png",alt:"img"}})]),t._v(" "),v("p",[t._v("对视频进行合适的时空处理如"),v("strong",[t._v("时域的滤波")]),t._v("和"),v("strong",[t._v("空间的分解")]),t._v("，然后将不同频率的信号分别放大。")]),t._v(" "),v("ul",[v("li",[t._v("空间分解")])]),t._v(" "),v("blockquote",[v("p",[t._v("由于分辨率较高的图像不够平滑，无法利用一阶泰勒展开来近似，所以要进行空间分解（图像金字塔）")]),t._v(" "),v("p",[t._v("分辨率越低的视频序列，其包含的噪声越少（可以理解为高斯模糊解决椒盐噪声的感觉），信噪比SNR较高")])]),t._v(" "),v("ul",[v("li",[t._v("时域滤波")])]),t._v(" "),v("blockquote",[v("p",[t._v("得到空间分辨率（图像金字塔）后，对同一空间分辨率下的图像序列进行时域滤波，提取"),v("strong",[t._v("感兴趣的信号和频率")]),t._v("（利用"),v("strong",[t._v("带通滤波器")]),t._v("）")]),t._v(" "),v("p",[t._v("正常人的心率在60bpm-100bpm之间，可以选择1-1.8hz的带通滤波器提取。可以采用小波")])]),t._v(" "),v("ul",[v("li",[t._v("放大滤波")])]),t._v(" "),v("blockquote",[v("p",[t._v("将时域滤波提取出来的感兴趣频带 乘以 "),v("strong",[t._v("合适的放大因数 α")]),t._v(" 来放大信号的变化")])]),t._v(" "),v("ul",[v("li",[t._v("合成图像")])]),t._v(" "),v("blockquote",[v("p",[t._v("将不同分辨率下放大后的信号，并与原图像叠加得到最后的输出结果。")])]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/uX3T7Fl2MpcKC8m.png",alt:"image-20220526202737536"}})]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/lCiM4QPsjOza2Wm.png",alt:"image-20220526203358525"}})]),t._v(" "),v("h2",{attrs:{id:"基于haar级联人脸检测所有帧的roi区域"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#基于haar级联人脸检测所有帧的roi区域"}},[t._v("#")]),t._v(" 基于haar级联人脸检测所有帧的ROI区域")]),t._v(" "),v("p",[t._v("haar特征主要分为三类：边缘特征、线性特征、中心特征。")]),t._v(" "),v("p",[t._v("人脸器官可以用Haar-like基本特征来描述")]),t._v(" "),v("h2",{attrs:{id:"颜色空间转换"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#颜色空间转换"}},[t._v("#")]),t._v(" 颜色空间转换")]),t._v(" "),v("p",[t._v("ROI区域的图像 转换到 更符合人眼视觉的 HSV空间进行处理。")]),t._v(" "),v("p",[t._v("即欧拉增强可以选择YIQ空间的颜色增强和"),v("strong",[t._v("HSV空间")]),t._v("的颜色增强；")]),t._v(" "),v("p",[t._v("心率信号提取的时 候 ， 只 需 要 对人脸区 域的 信号进行 分析 ， 去除 复 杂背 景对 于心率 提取的 影响 。")]),t._v(" "),v("p",[t._v("２ ） 由于肤色的亮度变化反映了心率信号的变化，因此将特征区域图像转换到更符合人眼视觉的HSV 空间上进行处理。")]),t._v(" "),v("p",[t._v("３ ） 利用高斯金字塔对图像进行多分辨率分解")]),t._v(" "),v("p",[t._v("４ ） 利用理想带通滤波器对空间分解后的图像进行滤波处理， 过滤出与心率相关的颜色变化信号")]),t._v(" "),v("p",[t._v("５ ） 放大颜色变化并合成图像")]),t._v(" "),v("h2",{attrs:{id:"心率信号的提取和计算"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#心率信号的提取和计算"}},[t._v("#")]),t._v(" 心率信号的提取和计算")]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/zwqMYH34EbXQ7FG.png",alt:"image-20220526204538179"}})]),t._v(" "),v("p",[t._v("由于血液对于绿光的吸收能力最强，所以绿色通道最能反映心率变化。")]),t._v(" "),v("p",[t._v("步骤：")]),t._v(" "),v("p",[t._v("①")]),t._v(" "),v("p",[t._v("每一帧画面的信号值，是所有像素点之和的均值。")]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/2jtfsHdDRzqQoY8.png",alt:"image-20220526211325232"}})]),t._v(" "),v("p",[t._v("每个通道都会有他的信号值。")]),t._v(" "),v("p",[t._v("对每个R、G、B通道进行归一化操作")]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/XdtnPhg6MyWD1ir.png",alt:"image-20220526211522902"}})]),t._v(" "),v("p",[v("img",{attrs:{src:"https://s2.loli.net/2022/05/26/CF6AYT1UrQc2aLq.png",alt:"image-20220526211622658"}})])])}),[],!1,null,null,null);_.default=r.exports}}]);