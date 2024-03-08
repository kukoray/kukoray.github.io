(window.webpackJsonp=window.webpackJsonp||[]).push([[8],{431:function(s,a,t){"use strict";t.r(a);var e=t(65),n=Object(e.a)({},(function(){var s=this,a=s.$createElement,t=s._self._c||a;return t("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[t("h1",{attrs:{id:"shell编程"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#shell编程"}},[s._v("#")]),s._v(" shell编程")]),s._v(" "),t("h2",{attrs:{id:"_1-sh"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-sh"}},[s._v("#")]),s._v(" 1 sh")]),s._v(" "),t("p",[s._v("这里指的shell是指shell脚本编程，不是指shell 本身")]),s._v(" "),t("p",[s._v("Linux 的 Shell 种类众多，常见的有：")]),s._v(" "),t("ul",[t("li",[s._v("Bourne Shell（/usr/bin/sh或/bin/sh）")]),s._v(" "),t("li",[s._v("Bourne Again Shell（/bin/bash）")]),s._v(" "),t("li",[s._v("C Shell（/usr/bin/csh）")]),s._v(" "),t("li",[s._v("K Shell（/usr/bin/ksh）")]),s._v(" "),t("li",[s._v("Shell for Root（/sbin/sh）")])]),s._v(" "),t("p",[t("strong",[s._v("我们一般常用 "),t("code",[s._v("sh")]),s._v(" 或者 "),t("code",[s._v("bash")])])]),s._v(" "),t("p",[t("strong",[s._v("#!/bin/sh")]),s._v("，它同样也可以改为 "),t("strong",[s._v("#!/bin/bash")]),s._v("。")]),s._v(" "),t("p",[t("strong",[s._v("#!")]),s._v(" 告诉系统其后路径所指定的程序即是解释此脚本文件的 Shell 程序。")]),s._v(" "),t("h3",{attrs:{id:"_1-1-运行-shell"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-1-运行-shell"}},[s._v("#")]),s._v(" 1.1 运行 Shell")]),s._v(" "),t("p",[t("strong",[s._v("1、作为可执行程序")])]),s._v(" "),t("p",[s._v("将上面的代码保存为 test.sh，并 cd 到相应目录：")]),s._v(" "),t("div",{staticClass:"language- extra-class"},[t("pre",{pre:!0,attrs:{class:"language-text"}},[t("code",[s._v("chmod +x ./test.sh  #使脚本具有执行权限\n./test.sh  #执行脚本\n")])])]),t("p",[s._v("注意，一定要写成 "),t("strong",[s._v("./test.sh")]),s._v("，而不是 "),t("strong",[s._v("test.sh")]),s._v("，运行其它二进制的程序也一样，直接写 test.sh，linux 系统会去 PATH 里寻找有没有叫 test.sh 的，而只有 /bin, /sbin, /usr/bin，/usr/sbin 等在 PATH 里，你的当前目录通常不在 PATH 里，所以写成 test.sh 是会找不到命令的，要用 ./test.sh 告诉系统说，就在当前目录找。")]),s._v(" "),t("p",[t("strong",[s._v("2、作为解释器参数")])]),s._v(" "),t("p",[s._v("这种运行方式是，直接运行解释器，其参数就是 shell 脚本的文件名，如：")]),s._v(" "),t("div",{staticClass:"language- extra-class"},[t("pre",{pre:!0,attrs:{class:"language-text"}},[t("code",[s._v("/bin/sh test.sh\n/bin/php test.php\n")])])]),t("h3",{attrs:{id:"_1-2-编写shell"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-2-编写shell"}},[s._v("#")]),s._v(" 1.2 编写shell")]),s._v(" "),t("p",[s._v("==变量名和等号之间不能有空格==")]),s._v(" "),t("p",[s._v("==单引号包裹的变量无法解析，双引号的可以实现解析==")]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token shebang important"}},[s._v("#!/bin/bash")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("name")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"Shell"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("url")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"http://c.biancheng.net/shell/"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("str1")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$name")]),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$url")]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#中间不能有空格")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("str2")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$name")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$url")]),s._v('"')]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#如果被双引号包围，那么中间可以有空格")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("str3")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$name")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('": "')]),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$url")]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#中间可以出现别的字符串")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("str4")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$name")]),s._v(": "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$url")]),s._v('"')]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#这样写也可以")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("str5")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("${name}")]),s._v("Script: "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("${url}")]),s._v('index.html"')]),s._v("  "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#这个时候需要给变量名加上大括号")]),s._v("\n")])])]),t("table",[t("thead",[t("tr",[t("th",[s._v("command")]),s._v(" "),t("th",[s._v("解释")]),s._v(" "),t("th",[s._v("返回")])])]),s._v(" "),t("tbody",[t("tr",[t("td",[s._v("-d file")]),s._v(" "),t("td",[s._v("检测文件是否是目录，如果是，则返回 true。")]),s._v(" "),t("td",[s._v("[ -d $file ] 返回 false。")])]),s._v(" "),t("tr",[t("td",[s._v("-f file")]),s._v(" "),t("td",[s._v("检测文件是否是普通文件（既不是目录，也不是设备文件），如果是，则返回 true。")]),s._v(" "),t("td",[s._v("[ -f $file ] 返回 true。")])]),s._v(" "),t("tr",[t("td",[s._v("-e file")]),s._v(" "),t("td",[s._v("检测文件（包括目录）是否存在，如果是，则返回 true。")]),s._v(" "),t("td",[s._v("[ -e $file ] 返回 true。")])])])]),s._v(" "),t("p",[t("strong",[s._v("判断文件是否存在")])]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("if")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("[")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("!")]),s._v(" -e trainingandtestdata.zip "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("]")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("then")]),s._v("\n    "),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("wget")]),s._v(" --no-check-certificate http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("fi")]),s._v("\n")])])]),t("p",[s._v("当文件夹中有同名的文件夹和文件时，可以使用"),t("code",[s._v("-d")]),s._v(" 或者 "),t("code",[s._v("-f")]),s._v("  来区分")]),s._v(" "),t("p",[t("strong",[s._v("解压zip文件")])]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token function"}},[s._v("unzip")]),s._v(" trainingandtestdata.zip\n")])])]),t("p",[t("strong",[s._v("变量的使用")])]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("NAME")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"sent140"')]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("cd")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("..")]),s._v("/utils\n\npython3 stats.py --name "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$NAME")]),s._v("\n\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("cd")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("..")]),s._v("/"),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$NAME")]),s._v("\n\n")])])]),t("p",[t("strong",[s._v("转移字符$")])]),s._v(" "),t("table",[t("thead",[t("tr",[t("th",[s._v("command")]),s._v(" "),t("th",[s._v("解释")])])]),s._v(" "),t("tbody",[t("tr",[t("td",[s._v("$#")]),s._v(" "),t("td",[s._v("传给脚本的参数个数")])]),s._v(" "),t("tr",[t("td",[s._v("$0")]),s._v(" "),t("td",[s._v("脚本本身的名字")])]),s._v(" "),t("tr",[t("td",[s._v("$1")]),s._v(" "),t("td",[s._v("传递给该shell脚本的第一个参数")])]),s._v(" "),t("tr",[t("td",[s._v("$2")]),s._v(" "),t("td",[s._v("传递给该shell脚本的第二个参数")])]),s._v(" "),t("tr",[t("td",[s._v("$@")]),s._v(" "),t("td",[s._v("传给脚本的所有参数的列表")])]),s._v(" "),t("tr",[t("td",[s._v("$$")]),s._v(" "),t("td",[s._v("脚本运行的当前进程ID号")])]),s._v(" "),t("tr",[t("td",[s._v("$?")]),s._v(" "),t("td",[s._v("显示最后命令的退出状态，0表示没有错误，其他表示有错误")])])])]),s._v(" "),t("p",[s._v("建立脚本peng.sh如下：")]),s._v(" "),t("div",{staticClass:"language-bash extra-class"},[t("pre",{pre:!0,attrs:{class:"language-bash"}},[t("code",[t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#/bin/bash")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("total")]),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v("$"),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("[")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$1")]),s._v(" * "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$2")]),s._v(" + "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$3")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("]")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$1")]),s._v(" * "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$2")]),s._v(" + "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$3")]),s._v(" = "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$total")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("123")]),s._v("\n")])])]),t("p",[s._v("运行如下：")]),s._v(" "),t("div",{staticClass:"language-bash extra-class"},[t("pre",{pre:!0,attrs:{class:"language-bash"}},[t("code",[s._v("./peng.sh "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("4")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("5")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("6")]),s._v("\n")])])]),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token shebang important"}},[s._v("#!/bin/sh")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"参数个数:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$#")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"脚本名字:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$0")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"参数1:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$1")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"参数2:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$2")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"所有参数列表:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$@")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"pid:'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$$")]),s._v('"')]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("if")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("[")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$1")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("100")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("]")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("then")]),s._v("\n     "),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"命令退出状态：'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$?")]),s._v('"')]),s._v(" \n     "),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("exit")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("0")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#参数正确，退出状态为0")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("else")]),s._v("\n     "),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("echo")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"命令退出状态：'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$?")]),s._v('"')]),s._v("\n     "),t("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("exit")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token number"}},[s._v("1")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#参数错误，退出状态1")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("fi")]),s._v("\n")])])]),t("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://img-blog.csdnimg.cn/37e37dd76e0f42eb8e7c4c0bb3aab3cd.png",alt:"img"}}),s._v(" "),t("h2",{attrs:{id:"pushd-和-popd"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#pushd-和-popd"}},[s._v("#")]),s._v(" pushd 和 popd")]),s._v(" "),t("p",[t("code",[s._v("cd -")]),s._v(" 可以回到之前切换过来的目录")]),s._v(" "),t("p",[s._v("pushd  xx/xx/xx")]),s._v(" "),t("p",[s._v("切换到该路径，并将该路径压到栈顶")]),s._v(" "),t("p",[t("img",{attrs:{src:"https://s2.loli.net/2022/12/06/e8thfQpuMF7PETH.png",alt:"image-20221206132733541"}})]),s._v(" "),t("p",[t("code",[s._v("popd")])]),s._v(" "),t("p",[s._v("删除栈顶的路径，并切换到此时的栈顶路径")]),s._v(" "),t("p",[t("img",{attrs:{src:"https://s2.loli.net/2022/12/06/KwO2B3iAEPqlzuX.png",alt:"image-20221206132928346"}})]),s._v(" "),t("p",[t("code",[s._v("pushd +3")]),s._v(" 切换到栈目录中下标为3的路径，并将该路径置到栈顶，后面的路径也跟随放到栈顶")]),s._v(" "),t("p",[t("img",{attrs:{src:"https://s2.loli.net/2022/12/06/X7OdwETfbACBWQF.png",alt:"image-20221206133129690"}})]),s._v(" "),t("p",[t("strong",[s._v("通常用法")])]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# Check that GloVe embeddings are available; else, download them")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("pushd")]),s._v(" models/sent140\n "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("if")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("[")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token operator"}},[s._v("!")]),s._v(" -f glove.6B.300d.txt "),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("]")]),t("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(";")]),s._v(" "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("then")]),s._v("\n  ./get_embs.sh\n "),t("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("fi")]),s._v("\n"),t("span",{pre:!0,attrs:{class:"token function"}},[s._v("popd")]),s._v("\n")])])]),t("h2",{attrs:{id:"接受用户变量"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#接受用户变量"}},[s._v("#")]),s._v(" 接受用户变量")]),s._v(" "),t("p",[s._v("read -p '请输入需要创建的文件路径：'   filePath")]),s._v(" "),t("div",{staticClass:"language-shell extra-class"},[t("pre",{pre:!0,attrs:{class:"language-shell"}},[t("code",[t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$@")]),s._v(" 和 "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$*")]),s._v(" 都表示传递给函数或脚本的所有参数\n当 "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$*")]),s._v(" 和 "),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$@")]),s._v(" 不被双引号"),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('" "')]),s._v("包围时，它们之间没有任何区别，都是将接收到的每个参数看做一份数据，彼此之间以空格来分隔。\n但是当它们被双引号"),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('" "')]),s._v("包含时，就会有区别了：\n"),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$*")]),s._v('"')]),s._v("会将所有的参数从整体上看做一份数据，而不是把每个参数都看做一份数据。\n"),t("span",{pre:!0,attrs:{class:"token string"}},[s._v('"'),t("span",{pre:!0,attrs:{class:"token variable"}},[s._v("$@")]),s._v('"')]),s._v("仍然将每个参数都看作一份数据，彼此之间是独立的。\n")])])])])}),[],!1,null,null,null);a.default=n.exports}}]);