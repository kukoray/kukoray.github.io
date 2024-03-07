(window.webpackJsonp=window.webpackJsonp||[]).push([[9],{428:function(t,a,s){"use strict";s.r(a);var v=s(65),r=Object(v.a)({},(function(){var t=this,a=t.$createElement,s=t._self._c||a;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"linux常用命令"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#linux常用命令"}},[t._v("#")]),t._v(" linux常用命令")]),t._v(" "),s("h2",{attrs:{id:"lsof"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#lsof"}},[t._v("#")]),t._v(" lsof")]),t._v(" "),s("p",[t._v("查看服务器 8000 端口的占用情况：")]),t._v(" "),s("div",{staticClass:"language-bash extra-class"},[s("pre",{pre:!0,attrs:{class:"language-bash"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("lsof")]),t._v(" -i:8000\n")])])]),s("h2",{attrs:{id:"rm"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#rm"}},[t._v("#")]),t._v(" rm")]),t._v(" "),s("ul",[s("li",[t._v("-r 将目录及以下之档案亦逐一删除。（递归删除）")])]),t._v(" "),s("h2",{attrs:{id:"mkdir"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#mkdir"}},[t._v("#")]),t._v(" mkdir")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("mkdir")]),t._v(" -p runoob2/test\n-p 递归创建，如果没有runoob2文件夹，就会创建；\n")])])]),s("h2",{attrs:{id:"echo"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#echo"}},[t._v("#")]),t._v(" echo")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token shebang important"}},[t._v("#!/bin/sh")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v("echo")]),t._v(" -e "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v('"OK! '),s("span",{pre:!0,attrs:{class:"token entity",title:"\\c"}},[t._v("\\c")]),t._v('"')]),t._v(" "),s("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# -e 开启转义 \\c 不换行")]),t._v("\n"),s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v("echo")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token string"}},[t._v('"It is a test"')]),t._v("\n")])])]),s("p",[t._v("显示执行结果")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v("echo")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token variable"}},[s("span",{pre:!0,attrs:{class:"token variable"}},[t._v("`")]),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("date")]),s("span",{pre:!0,attrs:{class:"token variable"}},[t._v("`")])]),t._v("\n")])])]),s("h2",{attrs:{id:"conda"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#conda"}},[t._v("#")]),t._v(" conda")]),t._v(" "),s("ul",[s("li",[t._v("更新软件：conda update 软件名")]),t._v(" "),s("li",[t._v("卸载软件：conda remove 软件名")]),t._v(" "),s("li",[t._v("删除环境：conda remove -n 环境名")]),t._v(" "),s("li",[t._v("克隆环境：conda create –n 新环境名 –clone 旧环境名")])]),t._v(" "),s("p",[t._v("（克隆就是备用的，如果发现新的好用的环境）")]),t._v(" "),s("ul",[s("li",[t._v("查找软件：conda search 软件名")])]),t._v(" "),s("p",[t._v("（查找用得最多）\n注意：以上的操作要在小环境里")]),t._v(" "),s("p",[t._v("一次性安装多个库")]),t._v(" "),s("div",{staticClass:"language-text extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("conda install -y trim-galore  hisat2  subread  multiqc  samtools  salmon  fastp\n")])])]),s("p",[t._v("conda config --show")]),t._v(" "),s("h2",{attrs:{id:"df"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#df"}},[t._v("#")]),t._v(" df")]),t._v(" "),s("h2",{attrs:{id:"du"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#du"}},[t._v("#")]),t._v(" du")]),t._v(" "),s("p",[t._v("查看文件夹占用磁盘大小，常用方法")]),t._v(" "),s("p",[t._v("du  -sh")]),t._v(" "),s("h2",{attrs:{id:"ps"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ps"}},[t._v("#")]),t._v(" ps")]),t._v(" "),s("p",[t._v("ps aux | grep ctjia | grep dolphindb")]),t._v(" "),s("h2",{attrs:{id:"grep"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#grep"}},[t._v("#")]),t._v(" grep")]),t._v(" "),s("p",[t._v('grep "xsaasa"  dolphindb.log | wc -l')]),t._v(" "),s("p",[t._v("统计dolphindb.log 文件中xsaasa的出现行数")]),t._v(" "),s("p",[t._v('grep -c  "xsaasa"  dolphindb.log')]),t._v(" "),s("h2",{attrs:{id:"xargs"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#xargs"}},[t._v("#")]),t._v(" xargs")]),t._v(" "),s("h2",{attrs:{id:"ulimit"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ulimit"}},[t._v("#")]),t._v(" ulimit")]),t._v(" "),s("p",[t._v("ulimit -a")]),t._v(" "),s("p",[t._v("ulimit -c")]),t._v(" "),s("h2",{attrs:{id:"scp"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#scp"}},[t._v("#")]),t._v(" scp")]),t._v(" "),s("h2",{attrs:{id:"sftp"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sftp"}},[t._v("#")]),t._v(" sftp")]),t._v(" "),s("p",[t._v("sftp   root@120.48.33.220")]),t._v(" "),s("p",[t._v("sftp使用的是ssh相同的22端口，可以直接下载服务器上的文件到本地")]),t._v(" "),s("h2",{attrs:{id:"netstat"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#netstat"}},[t._v("#")]),t._v(" netstat")]),t._v(" "),s("p",[t._v("netstat -ntlp")]),t._v(" "),s("div",{staticClass:"language-bash extra-class"},[s("pre",{pre:!0,attrs:{class:"language-bash"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("netstat")]),t._v(" -tunlp "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("|")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("grep")]),t._v(" 端口号\n")])])]),s("ul",[s("li",[t._v("-t (tcp) 仅显示tcp相关选项")]),t._v(" "),s("li",[t._v("-u (udp)仅显示udp相关选项")]),t._v(" "),s("li",[t._v("-n 拒绝显示别名，能显示数字的全部转化为数字")]),t._v(" "),s("li",[t._v("-l 仅列出在Listen(监听)的服务状态")]),t._v(" "),s("li",[t._v("-p 显示建立相关链接的程序名")])]),t._v(" "),s("h2",{attrs:{id:"lsof-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#lsof-2"}},[t._v("#")]),t._v(" lsof")]),t._v(" "),s("p",[t._v("lsof -i:8903")]),t._v(" "),s("p",[t._v("lsof -p PID")]),t._v(" "),s("h2",{attrs:{id:"tcpdump"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tcpdump"}},[t._v("#")]),t._v(" tcpdump")]),t._v(" "),s("h2",{attrs:{id:"tail"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tail"}},[t._v("#")]),t._v(" tail")]),t._v(" "),s("p",[t._v("tail -f  XXX.log")]),t._v(" "),s("h2",{attrs:{id:"vim"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#vim"}},[t._v("#")]),t._v(" vim")]),t._v(" "),s("p",[t._v("shift +G 到末尾")]),t._v(" "),s("p",[t._v("shift +ZZ 保存退出")]),t._v(" "),s("h2",{attrs:{id:"nohup"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#nohup"}},[t._v("#")]),t._v(" nohup")]),t._v(" "),s("p",[t._v("nohup     &")]),t._v(" "),s("p",[t._v("nohup    2>&1 &")]),t._v(" "),s("p",[t._v("基本含义")]),t._v(" "),s("ul",[s("li",[t._v("/dev/null 表示空设备文件")]),t._v(" "),s("li",[t._v("0 表示stdin标准输入")]),t._v(" "),s("li",[t._v("1 表示stdout标准输出")]),t._v(" "),s("li",[t._v("2 表示stderr标准错误")])]),t._v(" "),s("p",[t._v("每个程序在运行后，都会至少打开三个文件描述符，分别是0：标准输入；1：标准输出；2：标准错误。")]),t._v(" "),s("p",[t._v("2>&1表明将文件描述2（标准错误输出）的内容重定向到文件描述符1（标准输出），为什么1前面需要&？当没有&时，1会被认为是一个普通的文件，有&表示重定向的目标不是一个文件，而是一个文件描述符。在前面我们知道，test.sh >log.txt又将文件描述符1的内容重定向到了文件log.txt，那么最终标准错误也会重定向到log.txt。")]),t._v(" "),s("p",[t._v("& 放在命令到结尾，表示后台运行，防止终端一直被某个进程占用，这样终端可以执行别到任务，配合 >file 2>&1可以将log保存到某个文件中，但如果终端关闭，则进程也停止运行。如 command > file.log 2>&1 &")]),t._v(" "),s("p",[t._v("通过查看/proc/进程id/fd下的内容，可了解进程打开的文件描述符信息。")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("cd /proc/5270/fd   #进程5270所有打开的文件描述符信息都在此\nls -l              #列出目录下的内容\n 0 -> /dev/pts/7\n 1 -> /dev/pts/7\n 2 -> /dev/pts/7\n 255 -> /home/hyb/workspaces/shell/test.sh\n")])])]),s("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://s2.loli.net/2023/03/05/j9XS8gGzFyI36hr.png",alt:"image-20230305123839368"}}),t._v(" "),s("h2",{attrs:{id:"telnet"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#telnet"}},[t._v("#")]),t._v(" telnet")]),t._v(" "),s("p",[t._v("linux上会通过各种工具（perf，dstat，iostate，pidstat）等找到系统的性能瓶颈；")]),t._v(" "),s("h2",{attrs:{id:"perf"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#perf"}},[t._v("#")]),t._v(" perf")]),t._v(" "),s("h2",{attrs:{id:"pidof"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#pidof"}},[t._v("#")]),t._v(" pidof")]),t._v(" "),s("p",[t._v("pidof  java")]),t._v(" "),s("h2",{attrs:{id:"uname"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#uname"}},[t._v("#")]),t._v(" uname")]),t._v(" "),s("p",[t._v("uname -a")]),t._v(" "),s("h2",{attrs:{id:"mkdir-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#mkdir-2"}},[t._v("#")]),t._v(" mkdir")]),t._v(" "),s("p",[t._v("mkdir -p  递归创建")]),t._v(" "),s("h2",{attrs:{id:"awk"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#awk"}},[t._v("#")]),t._v(" awk")]),t._v(" "),s("h2",{attrs:{id:"sed"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sed"}},[t._v("#")]),t._v(" sed")]),t._v(" "),s("h2",{attrs:{id:"iperf"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#iperf"}},[t._v("#")]),t._v(" iperf")]),t._v(" "),s("p",[t._v("测试网络通信带宽")]),t._v(" "),s("p",[t._v("在A机器上")]),t._v(" "),s("p",[t._v("iperf -s")]),t._v(" "),s("p",[t._v("在B机器上")]),t._v(" "),s("p",[t._v("iperf -c HOST_A")]),t._v(" "),s("h2",{attrs:{id:"tar"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tar"}},[t._v("#")]),t._v(" tar")]),t._v(" "),s("p",[t._v("tar zxvf")]),t._v(" "),s("h2",{attrs:{id:"pstack"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#pstack"}},[t._v("#")]),t._v(" pstack")]),t._v(" "),s("h2",{attrs:{id:"gdb"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#gdb"}},[t._v("#")]),t._v(" gdb")]),t._v(" "),s("p",[t._v("调试core文件")]),t._v(" "),s("p",[t._v("bt")]),t._v(" "),s("h2",{attrs:{id:"kill"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#kill"}},[t._v("#")]),t._v(" kill")]),t._v(" "),s("p",[t._v("kill 命令的功能是给进程发送信号")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v("信号id")]),t._v(" "),s("th",[t._v("信号名")]),t._v(" "),s("th",[t._v("信号描述")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[t._v("2")]),t._v(" "),s("td",[t._v("SIGINT")]),t._v(" "),s("td",[t._v("中断(INTerrupt)。这与从终端发送的 Ctrl-c 执行相同的功能。它通常会终止一个程序。")])]),t._v(" "),s("tr",[s("td",[t._v("9")]),t._v(" "),s("td",[t._v("SIGKILL")]),t._v(" "),s("td",[t._v("杀死(KILL)。这个信号很特别。尽管程序可能会选择以不同的方式处理发送给它们的信号，包括一起忽略它们，但 KILL 信号实际上从未发送到目标程序。相反，内核会立即终止进程。 当一个进程以这种方式被终止时，它没有机会在它自己之后“清理”或保存它的工作。出于这个原因，SIGKILL 信号应仅用作其他终止信号失败时的最后手段。")])]),t._v(" "),s("tr",[s("td",[t._v("15")]),t._v(" "),s("td",[t._v("SIGTERM")]),t._v(" "),s("td",[t._v("终止(TERMinate)。这是 kill 命令发送的"),s("strong",[t._v("默认信号")]),t._v("。如果程序仍然“存活”到足以接收信号，它将终止。")])])])]),t._v(" "),s("p",[t._v("kill -9 强制停止")]),t._v(" "),s("p",[t._v("kill -15 优雅停止，一般会等接口执行完毕之后才结束，也是kill的默认信号")]),t._v(" "),s("div",{staticClass:"language-bash extra-class"},[s("pre",{pre:!0,attrs:{class:"language-bash"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("kill")]),t._v(" -9 PID\n"),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("netstat")]),t._v(" -ntlp   //查看当前所有tcp端口\n"),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("netstat")]),t._v(" -ntulp "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("|")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("grep")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("80")]),t._v("   //查看所有80端口使用情况\n"),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("netstat")]),t._v(" -ntulp "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("|")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("grep")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("3306")]),t._v("   //查看所有3306端口使用情况\n")])])]),s("h2",{attrs:{id:"core"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#core"}},[t._v("#")]),t._v(" core")]),t._v(" "),s("p",[s("strong",[t._v("1.Core文件简介")])]),t._v(" "),s("p",[t._v("Core文件其实就是内存的映像，当程序崩溃时，存储内存的相应信息，主用用于对程序进行调试。当程序崩溃时便会产生core文件，其实准确的应该说是core dump 文件,默认生成位置与可执行程序位于同一目录下，文件名为core.***,其中***是某一数字。")]),t._v(" "),s("p",[s("strong",[t._v("2.开启或关闭Core文件的生成")])]),t._v(" "),s("p",[t._v("关闭或阻止core文件生成：")]),t._v(" "),s("p",[t._v("$ulimit -c 0")]),t._v(" "),s("p",[t._v("打开core文件生成：")]),t._v(" "),s("p",[t._v("$ulimit -c unlimited   临时生效，重启后失效（当然一般也可以了）")]),t._v(" "),s("p",[t._v("检查core文件的选项是否打开：")]),t._v(" "),s("p",[t._v("$ulimit -a")]),t._v(" "),s("p",[t._v("以上配置只对当前会话起作用，下次重新登陆后，还是得重新配置。要想配置永久生效，得在/etc/profile或者/etc/security/limits.conf文件中进行配置。")]),t._v(" "),s("p",[t._v("首先以root权限登陆，然后打开/etc/security/limits.conf文件，进行配置：")]),t._v(" "),s("p",[t._v("#vim /etc/security/limits.conf")]),t._v(" "),s("p",[s("domain",[s("type",[s("item",[s("value")],1)],1)],1)],1),t._v(" "),s("p",[t._v("​    *        soft      core     unlimited")]),t._v(" "),s("p",[t._v("（3）在/etc/security/limits.conf最后增加如下两行记录：")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("@root soft core unlimited\n@root hard core unlimited\n")])])]),s("p",[t._v("（1）在/etc/rc.local 中增加一行 ulimit -c unlimited")]),t._v(" "),s("p",[t._v("（2）在/etc/profile 中增加一行 ulimit -c unlimited")]),t._v(" "),s("p",[t._v("这两个都是开机自启的一种方式")]),t._v(" "),s("p",[t._v("查看core生成的位置")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("cat /proc/sys/kernel/core_pattern\n")])])]),s("h2",{attrs:{id:"ssh"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ssh"}},[t._v("#")]),t._v(" ssh")]),t._v(" "),s("p",[t._v("一、使用账号密码")]),t._v(" "),s("p",[t._v("端口如果改了的话需要指定端口号")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("ssh")]),t._v(" jacky@xx.xx.xx.xx -p "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("2333")]),t._v("\n然后输入密码即可\n")])])]),s("p",[t._v("二、使用公钥")]),t._v(" "),s("p",[t._v("将客户端的id_rsa.pub放入服务器的authorized_keys中")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("ssh")]),t._v(" jacky@xx.xx.xx.xx -p "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("2333")]),t._v("\n无需输入密码\n")])])]),s("p",[t._v("三、使用私钥（服务器的私钥）")]),t._v(" "),s("p",[t._v("步骤：")]),t._v(" "),s("p",[t._v("将服务器的公钥id_rsa.pub放入authorized_keys中")]),t._v(" "),s("p",[t._v("将服务器的私钥文件下载到客户端，并修改权限为600")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("chmod")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("600")]),t._v(" ~/id_rsa  \n")])])]),s("p",[t._v("用客户端下载的私钥id_rsa来实现免密登录")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[t._v("ssh")]),t._v(" -i ~/id_rsa -p "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("22")]),t._v(" jacky@xx.xx.xx.xx \n")])])]),s("h2",{attrs:{id:"etc-profile"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#etc-profile"}},[t._v("#")]),t._v(" /etc/profile")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.352.b08-2.el7_9.x86_64\nexport PATH=$JAVA_HOME/bin:$PATH\nexport CLASSPATH=:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar\n")])])]),s("h2",{attrs:{id:"etc-hosts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#etc-hosts"}},[t._v("#")]),t._v(" /etc/hosts")]),t._v(" "),s("p",[t._v("配置主机名")]),t._v(" "),s("h2",{attrs:{id:"git"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#git"}},[t._v("#")]),t._v(" git")]),t._v(" "),s("p",[t._v("git rebase")]),t._v(" "),s("h2",{attrs:{id:"vim-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#vim-2"}},[t._v("#")]),t._v(" vim")]),t._v(" "),s("p",[t._v("i  a  o    : 都是insert")]),t._v(" "),s("p",[t._v("G  ：可以直接到文件的末尾")]),t._v(" "),s("p",[t._v("gg : 回到行首")]),t._v(" "),s("p",[t._v("/word   搜索单词word  （n 搜索下一个匹配词   N搜索上一个匹配词）")]),t._v(" "),s("p",[t._v("x  : del     X: backspace")]),t._v(" "),s("p",[t._v("dd  ： 剪切 这一行（4dd  剪切四行）     p 粘贴到这一行(在当前游标下一行开始粘贴 ， P则是在当前游标这一行粘贴)")]),t._v(" "),s("p",[t._v("yy  : 复制游标所在这一行  （4yy）")]),t._v(" "),s("p",[t._v("u  撤回   （ctrl + r   ： redo）")]),t._v(" "),s("p",[t._v("shift + ↑  向上翻页")]),t._v(" "),s("p",[t._v("ZZ  ：保存并退出")]),t._v(" "),s("p",[t._v("ZQ ：不保存，强制退出")]),t._v(" "),s("p",[t._v("搜索替换：")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v(":n1,n2s/word1/word2/g")]),t._v(" "),s("th",[t._v("n1 与 n2 为数字。在第 n1 与 n2 行之间寻找 word1 这个字符串，并将该字符串取代为 word2 ！举例来说，在 100 到 200 行之间搜寻 vbird 并取代为 VBIRD 则： 『:100,200s/vbird/VBIRD/g』。(常用)")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[s("strong",[t._v(":1,$s/word1/word2/g")]),t._v(" "),s("strong",[t._v("或 :%s/word1/word2/g")])]),t._v(" "),s("td",[s("strong",[t._v("从第一行到最后一行寻找 word1 字符串，并将该字符串取代为 word2 ！(常用)")])])]),t._v(" "),s("tr",[s("td",[s("strong",[t._v(":1,$s/word1/word2/gc 或 :%s/word1/word2/gc")])]),t._v(" "),s("td",[s("strong",[t._v("从第一行到最后一行寻找 word1 字符串，并将该字符串取代为 word2 ！且在取代前显示提示字符给用户确认 (confirm) 是否需要取代！(常用)")])])])])]),t._v(" "),s("p",[t._v("/etc/sudoers")]),t._v(" "),s("p",[t._v("/etc/passwd")]),t._v(" "),s("p",[t._v("%wheel 组")]),t._v(" "),s("p",[t._v("lsof  -p  PID")]),t._v(" "),s("p",[t._v("pstack  PID")]),t._v(" "),s("p",[t._v("gdb  attach   PID")]),t._v(" "),s("p",[t._v("thread apply  all bt")]),t._v(" "),s("p",[t._v("info threads")]),t._v(" "),s("p",[t._v("bt")]),t._v(" "),s("p",[t._v("grep top")]),t._v(" "),s("p",[t._v("perf top -p "),s("PID")],1),t._v(" "),s("p",[t._v("iostat -x -m 1")]),t._v(" "),s("p",[t._v("top -p  PID  -H")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("1. bt：查看函数调用栈的所有信息，当程序执行异常时，可通过此命令查看程序的调用过程；\n\n2. info threads：显示当前进程中的线程；\n\n3. thread id：切换到具体的线程id，一般切换到具体的线程后再执行bt等操作。\n")])])]),s("p",[t._v("dstat")]),t._v(" "),s("p",[t._v("修改系统时间")]),t._v(" "),s("p",[t._v("sudo timedatectl set-timezone 'Asia/Shanghai'")]),t._v(" "),s("h2",{attrs:{id:"md5sum"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#md5sum"}},[t._v("#")]),t._v(" md5sum")]),t._v(" "),s("p",[t._v("MD5 全称是报文摘要算法（Message-Digest Algorithm 5）")]),t._v(" "),s("p",[t._v("主要作用是：通过文件内容来生成一串对应的二进制字符串，然后可以通过这个字符串来校验文件前后是否被修改")]),t._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost ctjia"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("mkdir")]),t._v(" mdmd\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost ctjia"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v("cd")]),t._v(" mdmd/\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("cp")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("..")]),t._v("/data.csv ./\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("ls")]),t._v("\ndata.csv\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("cp")]),t._v(" data.csv data1.csv\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ md5sum * "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" d.md5\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("cat")]),t._v(" d.md5\n5d01ddf06dfab4af245001d907af65c8  data1.csv\n5d01ddf06dfab4af245001d907af65c8  data.csv\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ md5sum -c d.md5\ndata1.csv: OK\ndata.csv: OK\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("vim")]),t._v(" data1.csv\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$ md5sum -c d.md5\ndata1.csv: FAILED\ndata.csv: OK\nmd5sum: WARNING: "),s("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v(" computed checksum did NOT match\n"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("ctjia@localhost mdmd"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("$\n")])])]),s("h2",{attrs:{id:"tar-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tar-2"}},[t._v("#")]),t._v(" tar")]),t._v(" "),s("p",[t._v("打包命令")]),t._v(" "),s("p",[t._v("-x 解压  -c压缩")]),t._v(" "),s("p",[t._v("-v显示解打包过程")]),t._v(" "),s("p",[t._v("-f指定解压的tar包的包名")]),t._v(" "),s("p",[t._v("-C 指定解压目录")]),t._v(" "),s("p",[t._v("常用：")]),t._v(" "),s("ul",[s("li",[t._v("解压"),s("code",[t._v("tar -xvf")])]),t._v(" "),s("li",[t._v("打包"),s("code",[t._v("tar -cvf")])])]),t._v(" "),s("p",[t._v("如果包是tar.gz")]),t._v(" "),s("p",[t._v("那么加上-z参数")]),t._v(" "),s("ul",[s("li",[t._v("解压"),s("code",[t._v("tar -zxvf")]),t._v("     tar -zxvf tmp.tar.gz")]),t._v(" "),s("li",[t._v("打包"),s("code",[t._v("tar -zcvf")]),t._v("     tar -zcvf tmp.tar.gz /tmp/")])]),t._v(" "),s("h2",{attrs:{id:"sed-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sed-2"}},[t._v("#")]),t._v(" sed")]),t._v(" "),s("div",{staticClass:"language- extra-class"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("sed -i '/GOPATH/d' ~/.bashrc\n")])])]),s("p",[t._v("删除带  GOPATH的行")])])}),[],!1,null,null,null);a.default=r.exports}}]);