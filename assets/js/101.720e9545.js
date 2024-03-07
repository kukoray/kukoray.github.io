(window.webpackJsonp=window.webpackJsonp||[]).push([[101],{521:function(t,s,a){"use strict";a.r(s);var n=a(65),e=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"git常用命令"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#git常用命令"}},[t._v("#")]),t._v(" git常用命令")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://s2.loli.net/2022/10/18/lbJgpCOr2kAGnjt.png",alt:"image-20221018115203991"}})]),t._v(" "),a("p",[t._v("git工作区中文件的状态")]),t._v(" "),a("blockquote",[a("ul",[a("li",[t._v("untracked 未跟踪（未被纳入版本控制）")]),t._v(" "),a("li",[t._v("tracked 已跟踪（被纳入版本控制）\n"),a("ul",[a("li",[t._v("unmodified  未修改状态")]),t._v(" "),a("li",[t._v("modified  已修改状态")]),t._v(" "),a("li",[t._v("staged   已暂存状态")])])])])]),t._v(" "),a("h2",{attrs:{id:"全局设置"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#全局设置"}},[t._v("#")]),t._v(" 全局设置")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" config --global user.naem "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"kukoray"')]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" config --global user.email "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"kukoray@163.com"')]),t._v("\n\n查看配置项信息\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" config --list\n\n另外git的配置其实是一个文件，找到该文件也可以修改配置\n")])])]),a("h2",{attrs:{id:"常用命令"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#常用命令"}},[t._v("#")]),t._v(" 常用命令")]),t._v(" "),a("p",[t._v("本地仓库")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" status 查看文件状态\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" 将文件的修改加入暂存区\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("】git "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" test.txt\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),t._v("】git "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v(".")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" reset 将暂存区的文件取消暂存或者是切换到指定版本\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("】git reset test.txt\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),t._v("】git reset --hard 090950e68cc099c5c021194d14v18d7516sad456\n\tps:这里面的版本信息，可以通过git log 来查看所提交的 所有版本信息\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" commit 将暂存区的文件修改提交到版本库\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("】git commit -m "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"这里输入提交信息"')]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" log 查看日志\n")])])]),a("p",[t._v("远程仓库")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" remote 查看远程仓库（如果是clone下来的项目，会自动配置好了remote信息，init的项目需要手动配置）\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("】git remote\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),t._v("】git remote -v\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" remote "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("shortName"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("url"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" 添加远程仓库\n\t"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("】git remote "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" origin https://gitee.com/kukoray/test.git\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" clone "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("url"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v("从远程仓库克隆\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" pull "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("remote-name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("remote-branch-name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("从远程仓库拉取\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" push "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("remote-name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("remote-branch-name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("推送到远程仓库\n")])])]),a("h2",{attrs:{id:"分支操作"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#分支操作"}},[t._v("#")]),t._v(" 分支操作")]),t._v(" "),a("p",[t._v("查看分支")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch       列出所有本地分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch -r    列出所有远程分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch -a    列出所有本地分支和远程分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch -b "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" 创建分支并跳转\n\n// 删除本地分支,ps:如果你还在一个分支上，那么 Git 是不允许你删除这个分支的。所以，请记得退出分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch -d localBranchName\n\n// 删除远程分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" push origin --delete remoteBranchName\n")])])]),a("p",[t._v("创建分支")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("\n\nex:\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" branch hfutplus_dev     在本地创建一个名为hfutplus_dev的分支\n\n")])])]),a("p",[t._v("切换分支")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" checkout master \n")])])]),a("p",[t._v("推送本地分支至远程仓库")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" push origin hfutplus_dev     将本地名为hfutplus_dev的分支推送至远程仓库\n")])])]),a("p",[t._v("分支合并")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" merge "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v("\n\n例如：如果当前在master分支\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" merge dev      作用是：将dev分支merge到当前所在的分支，dev分支不会发生改变\n\n\n当分支发生冲突时，需要解决完冲突，在提交\n如果解决完冲突，提交时commit还是报错\n可以使用 "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" commit "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"XXXX"')]),t._v(" -i 来忽略报错\n")])])]),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("add")]),t._v(" test.txt    是将test.txt文件加入到缓存区\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" commit -m "),a("span",{pre:!0,attrs:{class:"token string"}},[t._v('"推送缓存区所有文件至当前本地仓库的分支"')]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" push origin master    将本地的分支推送到远程仓库\n")])])]),a("p",[t._v("标签操作：类似于一种快照技术，对于当前版本的情况进行一个记录，标签内的内容是无法修改的")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" tag   列出已有标签\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" tag "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" 创建标签\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" push "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("shortName"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" 将标签推送至远程仓库.   shortName是远程仓库的别名 一般都是origin\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" checkout -b "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("branch"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("name"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" 检出标签，创建一个新的分支来检出这个标签\n")])])]),a("h2",{attrs:{id:"gitignore"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#gitignore"}},[t._v("#")]),t._v(" gitignore")]),t._v(" "),a("div",{staticClass:"language-shell extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("# 当远程仓库有一些其他的提交历史，需要拉取的时候做")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("git")]),t._v(" pull origin master --allow-unrelated-histories\n")])])])])}),[],!1,null,null,null);s.default=e.exports}}]);