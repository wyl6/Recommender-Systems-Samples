# 基于深度学习的推荐(六)：CTR预估经典模型NFM

## 前言
早期做特征工程的时候,采用人工或决策树等来选择特征,然而这些方法无法学习到训练集中没有出现的特征组合.而近几年出现的基于embedding的方法,可以学习到训练集中没有出现的组合,作者将embedding方法归为两类,一类是FM这种线性模型,之前介绍过的FNN就是利用FM作为初始化的embedding;另一类是基于神经网络的非线性模型,NFM(Neural Factorization Machine)则是将两种embedding结合起来.NFM是发表在SIGIR 2017上的文章,出现在深度学习与推荐系统结合的初期,模型相对较为简单,可以拿来练习tensorflow.论文地址:https://arxiv.org/pdf/1708.05027.pdf

## NFM模型
首先来回顾下FM模型:
![Screenshot_2019-10-25_16-33-17.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-25_16-33-17.png)

设embedding向量维度为k,其中的二阶交叉项可以进行优化:
![20190618160910.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/20190618160910.png)

交叉项得到的是一个值,如果去掉最外面那层求和,得到一个k维的向量.这个k维的向量就是所谓的"Bi-Interaction Layer"的结果f(x):

![Screenshot_2019-10-25_16-29-59.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-25_16-29-59.png)
最终的预估公式就是:
![Screenshot_2019-10-25_16-37-23.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-25_16-37-23.png)

此时再看模型一目了然:
![Screenshot_2019-10-25_16-38-25.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-10-25_16-38-25.png)

## 代码实战
这部分代码改自之前AFM的代码,有兴趣可以自己改改试一试,挺简单的.其中interaction layer的实现提供了优化前和优化后两种写法,可以运行下比较比较时间,差距蛮大.简单看看几个关键点的实现,首先是embedding layer:
```python
with tf.name_scope('Embedding_Layer'):
    self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feature_index)  # [None, field_size, embedding_size]
    feat_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1])  # [None, field_size, 1]
    self.embeddings = tf.multiply(self.embeddings, feat_value)  # [None, field_size, embedding_size]
```
然后是预测公式的线性部分:
```python
with tf.name_scope('linear_part'):
    self.linear_part = tf.nn.embedding_lookup(self.weights['linear_w'],self.feature_index)  # [None, field_size, 1]
    self.linear_part = tf.reduce_sum(tf.multiply(self.linear_part, feat_value), axis=2)  # [None, field_size]
    self.linear_part = tf.nn.dropout(self.linear_part, self.dropout_keep_fm[0])  # [None, field_size]
    self.linear_out = tf.reduce_sum(self.linear_part, axis=1, keep_dims=True) # [None, 1]
    self.w0 = tf.multiply(self.biases['w0'], tf.ones_like(self.linear_out)) # [None, 1]
```
B-Interaction Layer这一层的实现比较简单:
```python
with tf.variable_scope('interaction_layer'):
    self.sum_square_emb = tf.square(tf.reduce_sum(self.embeddings, axis=1)) # [None, embedding_size]
    self.square_sum_emb = tf.reduce_sum(tf.square(self.embeddings), axis=1) # [None, embedding_size]
    self.fully_out = 0.5 * tf.subtract(self.sum_square_emb, self.square_sum_emb) # [None, embedding_size]
```
然后就是后面的全连接层和最后的预测结果:
```python
with tf.name_scope('fully_layer'):
    for i in range(len(self.deep_layers)):
        self.fully_out = tf.add(tf.matmul(self.fully_out, self.weights[i]), self.biases[i])
        self.fully_out = self.deep_layers_activation(self.fully_out)
        if(self.batch_norm):
            self.fully_out = self.batch_norm_layer(self.fully_out, self.train_phase)
        self.fully_out = tf.nn.dropout(self.fully_out, keep_prob=self.dropout_fm[i])

with tf.name_scope('out'):
    self.out = tf.add_n([self.w0, self.linear_out, self.fully_out]) # # yAFM = w0 + wx + f(x)
```

参考代码:

https://github.com/wyl6/Recommender-Systems-Samples/tree/master/RecSys%20And%20Deep%20Learning/Attention/AFM

https://github.com/faychu/nfm/blob/master/NeuralFM.py

完整数据和代码:

https://github.com/wyl6/Recommender-Systems-Samples/tree/master/RecSys%20And%20Deep%20Learning/DNN/nfm

## 参考
[1] https://arxiv.org/pdf/1708.05027.pdf

[2] https://github.com/faychu/nfm/blob/master/NeuralFM.py