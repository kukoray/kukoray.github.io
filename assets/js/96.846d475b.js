(window.webpackJsonp=window.webpackJsonp||[]).push([[96],{516:function(t,s,a){"use strict";a.r(s);var n=a(65),e=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"学习路线"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#学习路线"}},[t._v("#")]),t._v(" 学习路线")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/09/19/Yr8MQpmINZjzxvn.jpg",alt:"img"}})]),t._v(" "),a("h2",{attrs:{id:"一、内存的分区模型"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#一、内存的分区模型"}},[t._v("#")]),t._v(" 一、内存的分区模型")]),t._v(" "),a("p",[t._v("c++程序的分区")]),t._v(" "),a("ul",[a("li",[t._v("代码区")]),t._v(" "),a("li",[t._v("全局区")]),t._v(" "),a("li",[t._v("栈区：编译器自动分配释放，存放函数的参数值，局部变量等（通常我们dfs时说的爆栈就是这个东西）")]),t._v(" "),a("li",[t._v("堆区：由程序员分配和释放")])]),t._v(" "),a("p",[t._v("代码区：存放cpu执行的机器指令，代码区是共享的（对于频繁被执行的程序，内存中只需要存一份即可），也是只读（避免因为一些原因被修改了）")]),t._v(" "),a("p",[t._v("全局区：包括全局变量和静态变量（static）、常量（字符串常量、全局常量），该区域的数据在程序结束后由操作系统释放。"),a("strong",[t._v("局部变量、局部常量不在这个里面")])]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/09/22/dIAxSYrOHt4VvNQ.png",alt:"image-20220922164152922"}})]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/09/22/Zdj1Kf87rNMoqJp.png",alt:"image-20220922164236349"}})]),t._v(" "),a("p",[t._v("栈区：由编译器，存放局部变量、形参；")]),t._v(" "),a("p",[t._v("函数不要返回局部变量的地址，编译器第一次对于这个地址进行了一次保留，第二次的时候会被编译器释放，就访问不到那个局部变量的数据了。")]),t._v(" "),a("p",[t._v("堆区：由程序员分配释放")]),t._v(" "),a("p",[t._v("在c++中主要利用new")]),t._v(" "),a("p",[t._v("int * p = new int(10);   //这里指针p也是局部变量，但是指针保存的数据在堆区（申请一个int的空间，初始化为10，返回int *型的指针指向这块空间）")]),t._v(" "),a("p",[t._v("释放堆区，使用delete操作符")]),t._v(" "),a("p",[t._v("delete p;")]),t._v(" "),a("p",[t._v("int * arr = new int [10]; // 这里是分配了10个int的空间，返回了这个数组的首地址")]),t._v(" "),a("p",[t._v("delete [] arr;")]),t._v(" "),a("h2",{attrs:{id:"二、c-中的引用"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#二、c-中的引用"}},[t._v("#")]),t._v(" 二、c++中的引用")]),t._v(" "),a("p",[t._v("引用必须要初始化，必须要告诉他是谁的别名")]),t._v(" "),a("p",[t._v("一旦初始化之后，就不可以更改；")]),t._v(" "),a("p",[t._v("在函数中 引用也是可以实现用形参来改变实参的作用的")]),t._v(" "),a("p",[t._v("int a = 10;")]),t._v(" "),a("p",[t._v("int & ref = a;")]),t._v(" "),a("p",[t._v("和堆区的原理一样，局部变量的引用不要作为函数的返回值，因为会被栈区释放，第一次能保留，第二次的 时候就被编译器释放了")]),t._v(" "),a("p",[t._v("另外，一个函数的返回值如果是引用类型，那么这个函数名也可以作为坐值对他进行赋值；")]),t._v(" "),a("p",[t._v("例如  test()=1000;  //   int & test(){}")]),t._v(" "),a("p",[t._v("引用的本质：是一个"),a("strong",[t._v("指针常量")]),t._v("；   int & ref = a;相当于 int * const ref = &a ;")]),t._v(" "),a("p",[t._v("常量引用：const int & val ;防止函数中对于引用变量的改变，导致实参发生改变")]),t._v(" "),a("p",[t._v("const int  & 和 int &是两种不同的数据类型")]),t._v(" "),a("h2",{attrs:{id:"三、类和对象"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#三、类和对象"}},[t._v("#")]),t._v(" 三、类和对象")]),t._v(" "),a("p",[t._v("面向对象四大特性：封装、继承、多态、抽象")]),t._v(" "),a("div",{staticClass:"language-cpp extra-class"},[a("pre",{pre:!0,attrs:{class:"language-cpp"}},[a("code",[a("span",{pre:!0,attrs:{class:"token macro property"}},[a("span",{pre:!0,attrs:{class:"token directive-hash"}},[t._v("#")]),a("span",{pre:!0,attrs:{class:"token directive keyword"}},[t._v("include")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("<iostream>")])]),t._v("\n \n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("using")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("namespace")]),t._v(" std"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n \n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Box")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("public")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 长度")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" breadth"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 宽度")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" height"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 高度")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 成员函数声明")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("get")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("set")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" len"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" bre"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" hei "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 成员函数定义")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Box")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("get")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" breadth "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" height"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n \n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Box")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("set")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" len"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" bre"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" hei"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" len"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    breadth "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" bre"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    height "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" hei"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("int")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("main")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n   Box Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("        "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 声明 Box1，类型为 Box")]),t._v("\n   Box Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("        "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 声明 Box2，类型为 Box")]),t._v("\n   Box Box3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("        "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 声明 Box3，类型为 Box")]),t._v("\n   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("     "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 用于存储体积")]),t._v("\n \n   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// box 1 详述")]),t._v("\n   Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("height "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("5.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" \n   Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("6.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" \n   Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("breadth "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("7.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n \n   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// box 2 详述")]),t._v("\n   Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("height "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("10.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("12.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("breadth "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("13.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n \n   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// box 1 的体积")]),t._v("\n   volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("height "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" Box1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("breadth"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Box1 的体积："')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v("endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n \n   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// box 2 的体积")]),t._v("\n   volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("height "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" Box2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("breadth"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Box2 的体积："')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v("endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n \n \n   "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// box 3 详述")]),t._v("\n   Box3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("set")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("16.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("12.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" \n   volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Box3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("get")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" \n   cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Box3 的体积："')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" volume "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v("endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n")])])]),a("p",[t._v("保护权限和私有权限都是类内可以直接访问，类外无法访问")]),t._v(" "),a("p",[t._v("面试题：struct和class的区别")]),t._v(" "),a("p",[t._v("区别：")]),t._v(" "),a("ol",[a("li",[t._v("struct默认权限为公共")]),t._v(" "),a("li",[t._v("class默认访问权限为私有")])]),t._v(" "),a("h3",{attrs:{id:"构造函数和析构函数"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#构造函数和析构函数"}},[t._v("#")]),t._v(" 构造函数和析构函数")]),t._v(" "),a("p",[t._v("我们不提供时，编译器会提供空的实现")]),t._v(" "),a("div",{staticClass:"language-cpp extra-class"},[a("pre",{pre:!0,attrs:{class:"language-cpp"}},[a("code",[a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("//")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// Created by Jacky on 2022-09-23.")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("//")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token macro property"}},[a("span",{pre:!0,attrs:{class:"token directive-hash"}},[t._v("#")]),a("span",{pre:!0,attrs:{class:"token directive keyword"}},[t._v("include")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v("<iostream>")])]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("using")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("namespace")]),t._v(" std"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("public")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 这是构造函数")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("virtual")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("const")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("setLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("private")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 成员函数定义，包括构造函数")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Object is being created"')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("length")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("const")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("void")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("setLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    Line"),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),t._v("length "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 程序的主函数")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("int")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("main")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    Line line"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 设置长度")]),t._v("\n    line"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("setLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("6.0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"Length of line : "')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" line"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getLength")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n")])])]),a("p",[t._v("析构函数不可以有参数")]),t._v(" "),a("p",[t._v("对象在销毁前，会调用析构函数，只会调用一次")]),t._v(" "),a("h3",{attrs:{id:"深拷贝与浅拷贝"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#深拷贝与浅拷贝"}},[t._v("#")]),t._v(" 深拷贝与浅拷贝")]),t._v(" "),a("p",[t._v("拷贝构造函数")]),t._v(" "),a("div",{staticClass:"language-cpp extra-class"},[a("pre",{pre:!0,attrs:{class:"language-cpp"}},[a("code",[a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token double-colon punctuation"}},[t._v("::")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("Line")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("const")]),t._v(" Line "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("&")]),t._v("obj"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    cout "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"调用拷贝构造函数并为指针 ptr 分配内存"')]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<<")]),t._v(" endl"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    ptr "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("new")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("int")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("ptr "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("obj"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("ptr"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 拷贝值")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n")])])]),a("p",[t._v("深拷贝与浅拷贝的区别就在于深拷贝会在堆内存中另外申请空间来储存数据，从而也就解决了指针悬挂的问题。")]),t._v(" "),a("p",[t._v("简而言之，当数据成员中有指针时，必须要用深拷贝。")]),t._v(" "),a("p",[t._v("如果类没有显式实现拷贝构造函数，那么系统会调用默认的拷贝函数：浅拷贝")]),t._v(" "),a("h3",{attrs:{id:"抽象类-接口类"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#抽象类-接口类"}},[t._v("#")]),t._v(" 抽象类，接口类")]),t._v(" "),a("p",[t._v("接口描述了类的行为和功能，而不需要完成类的特定实现。")]),t._v(" "),a("p",[t._v("C++ 接口是使用"),a("strong",[t._v("抽象类")]),t._v("来实现的，抽象类与数据抽象互不混淆，数据抽象是一个把实现细节与相关的数据分离开的概念。")]),t._v(" "),a("p",[a("u",[t._v("如果类中至少有一个函数被声明为纯虚函数，则这个类就是抽象类")]),t._v('。纯虚函数是通过在声明中使用 "= 0" 来指定的，如下所示：')]),t._v(" "),a("div",{staticClass:"language-cpp extra-class"},[a("pre",{pre:!0,attrs:{class:"language-cpp"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("class")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token class-name"}},[t._v("Box")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("public")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 纯虚函数")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("virtual")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("getVolume")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("private")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(":")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("      "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 长度")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" breadth"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("     "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 宽度")]),t._v("\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("double")]),t._v(" height"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("      "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 高度")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n")])])]),a("p",[t._v("抽象类不能被用于实例化对象，它只能作为"),a("strong",[t._v("接口")]),t._v("使用。如果试图实例化一个抽象类的对象，会导致编译错误。")]),t._v(" "),a("h2",{attrs:{id:"四、文件和流"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#四、文件和流"}},[t._v("#")]),t._v(" 四、文件和流")])])}),[],!1,null,null,null);s.default=e.exports}}]);