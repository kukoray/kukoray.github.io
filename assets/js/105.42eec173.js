(window.webpackJsonp=window.webpackJsonp||[]).push([[105],{527:function(t,s,a){"use strict";a.r(s);var n=a(65),r=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"pytorch常用手册"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#pytorch常用手册"}},[t._v("#")]),t._v(" pytorch常用手册")]),t._v(" "),a("h2",{attrs:{id:"随机数种子设置"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#随机数种子设置"}},[t._v("#")]),t._v(" 随机数种子设置")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#---------------------------------------------------#")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#   设置种子")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#---------------------------------------------------#")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("seed_everything")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("11")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    random"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#设置random库")]),t._v("\n    np"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("random"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 设置numpy库")]),t._v("\n    torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("manual_seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 设置torch库——CPU")]),t._v("\n    torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("cuda"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("manual_seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 设置torch库——GPU")]),t._v("\n    torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("cuda"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("manual_seed_all"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 设置torch库——GPU")]),t._v("\n    torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("backends"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("cudnn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("deterministic "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),t._v("\n    torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("backends"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("cudnn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("benchmark "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("False")]),t._v("\n")])])]),a("h2",{attrs:{id:"数据读取"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#数据读取"}},[t._v("#")]),t._v(" 数据读取")]),t._v(" "),a("img",{staticStyle:{zoom:"67%"},attrs:{src:"https://s2.loli.net/2023/07/10/ksBKjP9DbH7oA6Q.png",alt:"image-20230710234942705"}}),t._v(" "),a("h3",{attrs:{id:"dataset"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dataset"}},[t._v("#")]),t._v(" Dataset")]),t._v(" "),a("h3",{attrs:{id:"dataloader"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#dataloader"}},[t._v("#")]),t._v(" Dataloader")]),t._v(" "),a("p",[t._v("判断模型梯度")]),t._v(" "),a("div",{staticClass:"language-python extra-class"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("nn "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" nn\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("import")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("optim "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("as")]),t._v(" optim\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义一个简单的模型")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("SimpleModel")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Module"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("__init__")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("super")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("SimpleModel"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("__init__"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linear "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Linear"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linear1 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("Linear"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("forward")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义模型的前向传播逻辑")]),t._v("\n        x "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" self"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("linear"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("x"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" x\nmodel "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" SimpleModel"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 定义损失函数和优化器")]),t._v("\ncriterion "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nn"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("MSELoss"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\noptimizer "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" optim"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("SGD"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("parameters"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" lr"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 输入数据")]),t._v("\ninputs "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 前向传播")]),t._v("\noutputs "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("inputs"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 计算损失")]),t._v("\nloss "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" criterion"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("outputs"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" torch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tensor"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 反向传播和参数更新")]),t._v("\nloss"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("backward"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\noptimizer"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("step"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 输出参数的属性")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),t._v(" param "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("in")]),t._v(" model"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("parameters"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[t._v("f'Parameter: ")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("param"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'")])]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[t._v("f' - Data: ")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("param"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("data"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'")])]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[t._v("f' - Gradient: ")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("param"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("grad"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'")])]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("print")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[t._v("f' - Requires Gradient: ")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("param"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("requires_grad"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("'")])]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n\n")])])]),a("p",[t._v("在PyTorch中，"),a("code",[t._v("model.state_dict()")]),t._v("方法返回的是模型中所有可学习参数（需要梯度的参数）的当前状态，即参数的数值，而不包括参数的梯度信息。这是为了方便保存和加载模型的参数状态而设计的。模型的梯度信息在反向传播过程中被计算，但在"),a("code",[t._v("state_dict()")]),t._v("中是不包含的。")]),t._v(" "),a("h2",{attrs:{id:"bn层"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#bn层"}},[t._v("#")]),t._v(" BN层")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://pica.zhimg.com/70/v2-870657a8bb025c4d0f2c04d453aa73e1_1440w.image?source=172ae18b&biz_tag=Post",alt:"关于pytorch中BN层（具体实现）的一些小细节"}})]),t._v(" "),a("p",[t._v("BN层的输出Y与输入X之间的关系是：Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta，此不赘言。其中gamma、beta为可学习参数（在pytorch中分别改叫weight和bias），训练时通过反向传播更新；")]),t._v(" "),a("p",[t._v("而running_mean、running_var则是在前向时先由X计算出mean和var，再由mean和var以动量momentum来更新running_mean和running_var。")]),t._v(" "),a("p",[t._v("在训练阶段，running_mean和running_var在每次前向时更新一次；")]),t._v(" "),a("p",[t._v("在测试阶段，则通过net.eval()固定该BN层的running_mean和running_var，"),a("strong",[t._v("此时这两个值即为训练阶段最后一次前向时确定的值")]),t._v("，并在整个测试阶段保持不变")]),t._v(" "),a("p",[t._v("rx+b")]),t._v(" "),a("p",[t._v("Clean: r1x+b1")]),t._v(" "),a("p",[t._v("Noise:r2x+b2")]),t._v(" "),a("p",[t._v("(r1+r2)x+b1+b2")]),t._v(" "),a("p",[t._v("从PyTorch 0.4.1开始, BN层中新增加了一个参数 track_running_stats,")]),t._v(" "),a("p",[t._v("BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)")]),t._v(" "),a("p",[t._v("这个参数的作用如下:")]),t._v(" "),a("p",[t._v("训练时用来统计训练时的forward过的min-batch数目,每经过一个min-batch, track_running_stats+=1\n如果没有指定momentum, 则使用1/num_batches_tracked 作为因数来计算均值和方差(running mean and variance).")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2024/03/08/Teg5Dryo1Id92A7.png",alt:"image-20230920112439997"}})]),t._v(" "),a("p",[a("code",[t._v("track_running_stats==False")])]),t._v(" "),a("p",[t._v("可以使用这个来控制，running_mean和running_var是否统计")])])}),[],!1,null,null,null);s.default=r.exports}}]);