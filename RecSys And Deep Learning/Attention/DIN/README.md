## 公众号
关注公众号:**推荐算法工程师**,输入"进群",加入交流群,和小伙伴们一起讨论机器学习,深度学习,推荐算法.

<img src="https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/deep_learning/dnn/tensorflow_mnist/wechat.jpg" width = "200" height = "200" />

## 前言

Deep Interest Network(DIN) 是盖坤大神领导的阿里妈妈的精准定向检索及基础算法团队提出的，发表在18年的ACM SIGKDD会议上。这篇文章讨论的是电子商业中的CTR预估问题.重点是对用户的历史行为数据进行分析和挖掘.

论文指出,很多CTR预估的模型,比如Deep Cross Network,Wide&Deep Learning等,采用的都是一种相似的模型架构,首先使用Embedding&MLP 从高维稀疏输入中提取低维稠密vectors,然后使用Sum/Average Pooling等技巧获得定长vectors.然后使用MLP进行预测.但是这样做存在一些问题.

首先是将原始的高维稀疏向量压缩为低维稠密向量后,定长向量sum pooling会导致信息损失;另外,低维向量的表达能力有限,而用户的兴趣多种多样(不是几十种这种量级的多种多样),为了提高定长vector的表达能力需要进行维度扩展,然而这容易带来维度灾难等问题,网站物品越来越多的时候总不能一直提高定长vector的维度吧.

然后作者注意到,用户是否会点击推荐给他的物品,仅取决于历史行为的一部分,并称之为local activation.因此,计算推荐物品的点击率时,只有部分物品代表的兴趣分布真正在起作用.DIN正是通过使用attention机制,对不同推荐的物品,获得用户不同的特征表示,从而进行更加精确的CTR预估.

论文链接：https://arxiv.org/pdf/1706.06978.pdf

## 1.模型分析
### 1.1 Baseline
首先说一下，论文中使用Goods表示用户历史中的广告/商品，使用Ad代表候选广告/商品。
![Screenshot_2019-10-03_20-13-49.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_20-13-49.png)

首先看下对照组设置的模型,其中user profile表示的意思如下：
![Screenshot_2019-10-03_20-16-37.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_20-16-37.png)

显而易见，首先对各类数据进行稀疏编码，然后使用MLP获得稠密向量，把各类特征串联起来，后面再来个MLP。
这里User Behavior使用multi-hot编码，因为用户不太可能只对其中一个广告感兴趣。要注意的是User Profile和Goods id以及Context Features的稠密向量并不一定是同一纬度。用户的特征向量怎么来的呢？将某个用户历史中点击过的物品的稠密向量进行Sum Pooling，就是直接求和，通过它们来代表当前用户。
### 1.2 DIN
![Screenshot_2019-10-03_20-14-07.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_20-14-07.png)

上图就是DIN模型，就是比Baseline多了一个Attention机制，下面我们重点讲讲这个attention的细节。

显而易见，所谓attention，就是给用户历史记录中不同广告赋予不同的权重，DIN中这个权重是一个和候选Ad相关的函数，也就是上图右上角的Activation Unit：将Inputs from user和Inputs from Candidate Ad这两部分特征做交互，每个Goods都需要和Candidate Ad通过这个Activation Unit来获得该Goods的权重。

然后和baseline一样，使用Sum Pooling获得定长vector作为用户的representation，其中权重函数a就是上图中的Activation Unit:
![Screenshot_2019-10-03_20-58-52.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_20-58-52.png)

但是同样是定长的user vector，候选广告不同时，DIN中user vector是不同的，而baseline中是相同的，显然DIN获得的用户representation更具有灵活性，也更加“精确”。

DIN中的Attention部分， 简单地说，就是和Goods和Candidate Ad有关的权重函数，赋予不同Goods不同权重，从而获得更好的用户representation。


## 2. 训练技巧
### 2.1 Dice激活函数
PRelu是一种常用的激活函数：
![Screenshot_2019-10-03_21-14-48.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_21-14-48.png)

但作者认为，不应该所有的突变点都选为0，而是应该依赖数据分布。因此，对PRelu进行改进，提出Dice激活函数，其中s是激活函数输入中的其中一维，即对每维进行去均值求方差操作：
![Screenshot_2019-10-03_21-16-34.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_21-16-34.png)

[1]中认为，去均值化操作使得突变点为s的均值，实现了data dependent的想法；而使用sigmoid可以得到0-1间的概率值，从而权衡s和αs。

### 2.2 Mini-batch 正则化


我们知道，电商中的商品数据符合长尾分布，只有少量商品多次出现，大量商品只出现几次或不出现。因此User Behaviors很稀疏是肯定的，由于输入是高维系数特征，而模型又很复杂，参数太多，相当于复杂模型而数据不足，这种情况很容易过拟合。

而对大体量的拥有上亿参数和非常稀疏的训练网络来说，直接应用L1,L2这种传统正则化方法是不适用的。比如L2正则化，在一个mini-batch中，只有非零特征对应的参数才会被更新，然而所有的参数都需要计算L2正则。

因此作者提出进行Mini-batch Aware正则化，只计算每个mini-batch中非零特征的相关参数：
![Screenshot_2019-10-03_22-03-17.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_22-03-17.png)
上述公式可以转化为：
![Screenshot_2019-10-03_21-51-34.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_21-51-34.png)

此时出现次数越多的非零特征对应的乘法权重越大。而上述公式可近似为：
![Screenshot_2019-10-03_21-53-30.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_21-53-30.png)
此时amj取值0或1，这种进一步的近似类似从sum pooling到max pooling的转化，论文并没有给出原因和证明，我想是为了简化训练过程吧。


## 3.实战
### 3.1 数据集
论文中使用了淘宝和亚马逊的数据集，数据集都太大了。可以下载一个亚马逊的小数据集进行实验：
```python
echo "begin download data"
mkdir raw_data && cd raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
echo "download data successful"
```
试验中使用了这个小数据集的迷你版本：reviews_Electronics_5.json和meta_Electronics.json，来自参考代码。其中reviews_Electronics_5.json：
![Screenshot_2019-10-03_22-29-40.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_22-29-40.png)
meta_Electronics.json：
![Screenshot_2019-10-03_22-29-53.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-03_22-29-53.png)
由于两个数据可能来自不同的部门，因此为了获得完整的reviewerID,asin,和categories等条目，需要取两者的公共部分。详见代码。
### 3.2 代码分析

代码使用的是jupyter notebook风格，各变量的形状我都标注在后面了。attention和Dice部分对照论文一看就懂。大概讲一下整个架构，也就是Model函数，分为四部分：首先是candidate ad的embedding:
```python
# get item embedding vectors for input 
i_c = tf.gather(cate_list, self.item)  # obtain category of every item
item_emb = tf.concat(values=[
    tf.nn.embedding_lookup(item_emb_w, self.item), # [B, H/2]
    tf.nn.embedding_lookup(cate_emb_w, i_c)        # [B, H/2]
], axis=1) # [B, H]
```
然后是user behaviors的embedding:
```python
# get hist embedding vectors for input
h_c = tf.gather(cate_list, self.hist)
hist_emb = tf.concat(values=[
    tf.nn.embedding_lookup(item_emb_w, self.hist), # [B, T, H/2]
    tf.nn.embedding_lookup(cate_emb_w, h_c)        # [B, T, H/2]
], axis=2) # [B, T, H]
```
上述实现我觉得有些问题。self.hist中除用户历史商品外其他都是0，也就是说几乎所有的用户都拥有第0个商品的embedding，由于在之前的DataInput函数预处理时恰好将包含0商品的记录全删掉了，此时0商品的embedding相当于对所有用户都添加了一个噪声，再加上attention机制，影响应该不大，但仍然有些问题。

由behaviors中各goods的embedding获得user embedding：
```python
# get user embedding vectors based on user hist vectors and item(to predict) vectors
att_hist_emb = attention(item_emb, hist_emb, self.sl)       # [B, 1, H]
att_hist_emb = tf.layers.batch_normalization(inputs=att_hist_emb)  # [B, 1, H]
att_hist_emb = tf.reshape(att_hist_emb, [-1, hidden_units]) # [B, H]
att_hist_emb = tf.layers.dense(att_hist_emb, hidden_units)  # [B, H]
user_emb = att_hist_emb
```
然后是后面的MLP预测部分：
```python
base_i = tf.concat([user_emb, item_emb], axis=-1) # [B, 2*H]
base_i = tf.layers.batch_normalization(base_i, name='base_i', reuse=tf.AUTO_REUSE) # [B, 2*H]
d_layer_1_i = tf.layers.dense(base_i, 80, activation=tf.nn.sigmoid, name='f1', reuse=tf.AUTO_REUSE) # [B, 80]
d_layer_1_i = dice(d_layer_1_i, name='dice1')
d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2', reuse=tf.AUTO_REUSE) # [B, 40]
d_layer_2_i = dice(d_layer_2_i, name='dice2')
d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3', reuse=tf.AUTO_REUSE) # [B, 1]
# 特征平铺
d_layer_3_i = tf.reshape(d_layer_3_i, [-1]) # [B,]
i_b = tf.gather(item_emb_b, self.item) # obtain bias of every item, [B,]
self.pre = i_b+d_layer_3_i # [B]
```
论文开源代码：https://github.com/zhougr1993/DeepInterestNetwork

参考代码：https://github.com/Crawler-y/Deep-Interest-Network

完整代码和数据：https://github.com/wyl6/Recommender-Systems-Samples/tree/master/RecSys%20And%20Deep%20Learning/Attention/DIN

## 参考
[1] https://juejin.im/post/5b5591156fb9a04fe91a7a52

[2] https://www.jianshu.com/p/a356a135a0d2