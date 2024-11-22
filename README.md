# topic：基于深度学习因果推断的资产定价分析

概述：

#### 1.基于错误定价的股市资产推断原理

这部分参考这个文献：Bartram S M, Grinblatt M. Agnostic fundamental analysis works[J]. Journal of Financial Economics, 2018, 128(1): 125-147. 

根据股票的基本面指标构建错误定价变量->列出错误定价的公式(见上文献，pdf的公式是错的）->讲一下选择它的原因（文献）

**表1 变量说明** pdf

**表2 全部变量的描述性统计** 

#### 2.基于Adaboost算法的错误定价变量构建

原理：利用Adaboost算法，根据同一日期的所有股票的基本面因子，分组预测每个股票的错误定价变量

预测后根据日期分组，对于每个日期进行选股投资：股票按照mispricing的值从大到小排序，比较两种投资策略：Q5:买入前20%的股票，x月卖出；Q1:买入后20%股票，x月后卖出，对比收益率

结果：Q5的平均收益率基本高于Q1，且随x的推移，差距愈发明显，证明了错误定价变量在资产定价上的作用以及延迟生效的特性。

折线图：

x=1:

![image-20241122114809748](C:\Users\taosuozi\AppData\Roaming\Typora\typora-user-images\image-20241122114809748.png)

x=6:

![image-20241122114859541](C:\Users\taosuozi\AppData\Roaming\Typora\typora-user-images\image-20241122114859541.png)

x=12:

![image-20241122114833846](C:\Users\taosuozi\AppData\Roaming\Typora\typora-user-images\image-20241122114833846.png)

**表3 错误定价M构建的投资组合表现**

<img src="C:\Users\taosuozi\AppData\Roaming\Typora\typora-user-images\image-20241122114107078.png" alt="image-20241122114107078" style="zoom:33%;" />

#### 3.错误定价变量（Mispricing）的因果推断分析

主要参考pdf **1、因果图基础模型** **2、变分自编码器（VAE）深度学习模型** 和 **4.3 错误定价与市值变量的因果推断 ITE（Individual Treatment Effect）**

**表4 错误定价变量M与市值变量的个体处理效应ITE**

M:

ITE： 735671877861023
T-statistic: 33.9656
P-value: 0.0000

市值变量（作为对照组）：

ITE: -0.0005595158544303851
T-statistic: -27.8666
P-value: 0.0000

错误定价变量的 ITE 为xxx ，显著性较高，表明错误定价对资产收益具有一定的正向因果效应。

市值变量的 ITE 为xxx，表明市值对资产收益的因果效应较小。