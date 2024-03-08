(window.webpackJsonp=window.webpackJsonp||[]).push([[86],{507:function(t,s,a){"use strict";a.r(s);var n=a(65),e=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"数码管下板"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#数码管下板"}},[t._v("#")]),t._v(" 数码管下板")]),t._v(" "),a("h2",{attrs:{id:"数码管显示降频"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#数码管显示降频"}},[t._v("#")]),t._v(" 数码管显示降频")]),t._v(" "),a("p",[t._v("将频率降到1000hz左右，这是显示效果最好的。")]),t._v(" "),a("blockquote",[a("p",[t._v("为了减少实际使用的FPGA芯片的IO端口，可采用分时复用的扫描显示方案进行数码管驱动。分时复用的扫描显示利用了人眼的视觉暂留特性，如果公共端控制信号的刷新速度足够快，人眼就分辨不出LED的闪烁，认为数码管时同时点亮的。控制信号的最佳刷新频率为1000Hz左右")])]),t._v(" "),a("div",{staticClass:"language-verilog extra-class"},[a("pre",{pre:!0,attrs:{class:"language-verilog"}},[a("code",[a("span",{pre:!0,attrs:{class:"token constant"}},[t._v("`timescale")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("ns "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("ps\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("module")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("top_led_dynamic")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("output")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("reg")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("7")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("    seg"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\t"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 数码管的公共段选信号")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("output")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("reg")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("    an"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\t\t"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 作为4个数码管的位选信号")]),t._v("\n    \n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("input")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("wire")]),t._v("            clk"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("input")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("wire")]),t._v("            rst"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("input")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("wire")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("   in3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" in2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" in1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" in0\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    \t"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// EGo1数码管是共阴极的，需要连接高电平，对应位置被点亮")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_0")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hc0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_1")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hf9")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_2")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'ha4")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_3")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hb0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_4")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h99")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_5")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h92")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_6")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h82")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_7")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hf8")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_8")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h80")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_9")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h90")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _a "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h88")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _b "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h83")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _c "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hc6")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _d "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'ha1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _e "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h86")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _f "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'h8e")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   _err "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("8'hcf")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\t   \n\t   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("parameter")]),t._v("   N "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("18")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("//此参数就是调节数码管下板频率的。")]),t._v("\n    \n       \n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("reg")]),t._v("     "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("N"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("  regN"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" \n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("reg")]),t._v("     "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("       hex_in"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    \n    "),a("span",{pre:!0,attrs:{class:"token important"}},[t._v("always @")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("posedge")]),t._v(" clk "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("or")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("posedge")]),t._v(" rst"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("   "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("if")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("rst "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("==")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1'b1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n            regN    "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("else")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n            regN    "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  regN "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// regN实现对100MHz的系统时钟的2^16分频")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("// 让每一个数码管都机会亮起来")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token important"}},[t._v("always @")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("case")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("regN"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("N"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v(" N"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2'b00")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n                an  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'b0001")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n                hex_in  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  in0"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2'b01")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n                an  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'b0010")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n                hex_in  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  in1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2'b10")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n                an  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'b0100")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n                hex_in  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  in2"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2'b11")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n                an  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'b1000")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n                hex_in  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  in3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("default")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n                an  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'b1111")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n                hex_in  "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  in3"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("endcase")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n    \n    "),a("span",{pre:!0,attrs:{class:"token important"}},[t._v("always @")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("case")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("hex_in"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_2")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_3")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h4")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_4")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_5")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h6")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_6")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h7")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_7")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h8")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_8")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'h9")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("_9")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'ha")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _a"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'hb")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _b"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'hc")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _c"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'hd")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _d"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'he")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _e"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("4'hf")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("   seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _f"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("default")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("seg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<=")]),t._v("  _err"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("endcase")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n            \n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("endmodule")]),t._v("\n\n")])])]),a("h2",{attrs:{id:"clk降频"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#clk降频"}},[t._v("#")]),t._v(" clk降频")]),t._v(" "),a("p",[t._v("为什么需要降频？")]),t._v(" "),a("blockquote",[a("p",[t._v("例子：当使用数码管做一个60秒计时器时；")]),t._v(" "),a("p",[t._v("如果使用EGO1本身的clk信号（P17） 其自身的频率为100MHz；")]),t._v(" "),a("p",[t._v("最后数码管的显示效果就是全都是8；")]),t._v(" "),a("p",[t._v("原因就是clk频率太快，导致每一个晶管在肉眼看来都是点亮状态。")]),t._v(" "),a("p",[t._v("所以我们需要对clk进行降频。")])]),t._v(" "),a("p",[t._v("如何降频？")]),t._v(" "),a("div",{staticClass:"language-verilog extra-class"},[a("pre",{pre:!0,attrs:{class:"language-verilog"}},[a("code",[t._v("    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("reg")]),t._v(" clk_10 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1'b0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("integer")]),t._v(" count "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n    "),a("span",{pre:!0,attrs:{class:"token important"}},[t._v("always@")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("clk"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("if")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("count "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("==")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("10000000")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n            count"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n            clk "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("~")]),t._v("clk"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("else")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("begin")]),t._v("\n            count "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" count"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("end")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("//此处的clk_10就是降频之后的10hz频率")]),t._v("\n")])])]),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/06/11/Gy4BCj6RYF9iN7U.png",alt:"image-20220424090205597"}})]),t._v(" "),a("p",[t._v("对于段选信号来说：")]),t._v(" "),a("p",[t._v("​\t\t\tseg [7:0] 其高位到低位依次对应的是：dp，g，f，e，d，c，b，a。")]),t._v(" "),a("p",[t._v("​\t\t\t对应的引脚是D5、B2、B3、A1、B1、A3、A4、B4")]),t._v(" "),a("p",[t._v("段选信号左边四个数码管 共用 同样的8个引脚；右边的四个 共用 8个引脚。")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/06/11/FcxEe53d4yuYlQL.png",alt:"image-20220424090345330"}})]),t._v(" "),a("p",[t._v("an就是位选信号：说白就是有8个数码管，an如果是[7:0]的话，哪一位是1就让哪一位亮；例如10000000，就是G2这个位亮，至于具体显示什么数字，那就是由段选信号seg来决定的。")]),t._v(" "),a("p",[t._v("当然an不一定需要7位，因为最后还是跟你接口所连接的情况来看的（即 约束文件constraint）")]),t._v(" "),a("p",[t._v("例如an可能就是4位（[3:0]），然后高位到低位，依次接的是 G2、C2、C1、H1；")]),t._v(" "),a("p",[t._v("那本质上给4‘b0001，那也是可以让H1这个位亮的。")]),t._v(" "),a("p",[a("strong",[t._v("数码管亮的本质就是：每一次都只让一个位亮，但是通过快速的动态刷新，让其感觉是所有位都是一起亮的。")])])])}),[],!1,null,null,null);s.default=e.exports}}]);