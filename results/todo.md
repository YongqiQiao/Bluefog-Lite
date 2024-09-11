<!-- 按通信量作scaling，记录通信量
40采样的问题，试一下1和5，比较效果（注意学习率）
试试前向法
check：每个agent自己的model，看看model是否相近，看看各自的acc和loss
cifar -->

拓扑影响：4，8，16   静态：ring 指数图多测试几个拓扑
多卡：单卡8节点，4卡2节点等测试区别
通信时间和计算时间：比较传整个model和梯度vector，记录每个agent的时间
记录单个epoch时间，观察波动
更大的数据集和模型：试试fine-tune
调研：联邦学习里微调大模型：motivation
