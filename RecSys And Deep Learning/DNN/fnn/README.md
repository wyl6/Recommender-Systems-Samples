## 前言
> 论文地址:https://arxiv.org/pdf/1601.02376.pdf

> 论文开源代码(基于Theano实现):https://github.com/wnzhang/deep-ctr

> 参考代码(无FM初始化):https://github.com/Sherryuu/CTR-of-deep-learning

> 重构代码:https://github.com/wyl6/Recommender-Systems-Samples/tree/master/RecSys%20And%20Deep%20Learning/DNN/fnn



## FNN = FM+MLP
### FM在到底初始化什么
FNN首先使用FM初始化输入embedding层,然后使用MLP来进行CTR预估,具体怎么做的呢?看论文中的一张图:

![Screenshot_2019-08-27_09-59-36.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_09-59-36.png)

单看图来理解的有一定的迷惑性,加上z的输出结公式就更有迷惑性了:

![Screenshot_2019-08-27_10-01-44.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_10-01-44.png)

其中wi为第i个field经FM初始化得到的一次项系数,vi就是隐向量,K为隐向量vi的维度.如果初始化的是z,那Dense Real Layer显示的结果显示每个field只有1个wi和vi,这不对啊,之前看FM的时候每个field的每个特征都对应一个wi和vi,这是怎么回事呢?实际上,FM初始化的是系数向量x到dense layer之间的权重矩阵W:

![Screenshot_2019-08-27_10-37-02.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_10-37-02.png)

我给大家画张图,假设样本有3个field,3个field维度分别为N1,N2,N3,那我们经过FM初始化可以获得N=N1+N2+N3个隐向量和一次项系数w,用它们组成权重矩阵W0:

![Screenshot_2019-08-27_11-07-28.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_11-07-28.png)

但是作者并没有直接将x和权重矩阵相乘来计算z,这样计算出的结果是K+1维,相当于把样本的所有非零特征对应的K+1维向量加起来.降维太过了,数据压缩太厉害总会损失一部分信息,因此作者将每个field分别相乘得到K+1维结果,最后把所有field的结果串联起来:

![Screenshot_2019-08-27_13-17-22.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_13-17-22.png)

这样初始化时,由于样本每个field只有一个非零值,第i个field得到的z值就是非零特征对应的w和v:

![Screenshot_2019-08-27_10-01-44.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_10-01-44.png)

### FNN的流程
了解FM初始化的是权重矩阵W0后,FNN流程就清楚了,从后往前看,一步到位:

![Screenshot_2019-08-27_10-09-21.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_10-09-21.png)



## 代码实战
### 数据格式
数据共有22个field,各field中属性取值的可枚举个数为:
```python
FIELD_SIZES = [1037, 151, 59, 1603, 4, 333, 77890, 1857, 9, 8, 4, 7, 22, 3, 92, 56, 4, 920, 38176, 240, 2697, 4]
```
样本x则被划分为:
![Screenshot_2019-08-27_11-32-37.png](https://raw.githubusercontent.com/wyl6/wyl6.github.io/master/imgs_for_blogs/Screenshot_2019-08-27_11-32-37.png)
由于是模拟CTR预估,所以标签y是二分类的,实验中y∈{0,1}.

### 参数保存与加载
FNN源代码中没有FM初始化这部分,只有MLP，博主自己加上了.

参数保存,常见的就是使用`tf.train.Saver`.保存所有模型的参数和值,然后加载部分参数或全部参数;或者保存指定参数和参数值,然后加载想要的参数和参数的值.为了和源代码借口保持一致,我们并没有使用`tf.train.Saver`,而是直接获取参数值,构造一个字典,保存到本地:
```python
def dump(self, model_path):
#        weight = [self.vars['w'], self.vars['v'], self.vars['b']]
#        saver = tf.train.Saver(weight)
#        saver.save(self.sess, model_path)
#        print(self.sess.run(self.vars['w']))
#        print(self.sess.run('w:0'))
#        print(self.vars['w'])
#        for i,j in self.vars.items():
#            print(i, j)
#            print(self.sess.run(j))
        var_map = {}
        for name, var in self.vars.items():
            print('----------------',name, var)
            var_map[name] = self.sess.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print('model dumped at', model_path)
        load_var_map = pkl.load(open(model_path, 'rb'))
        print('load_var_map[w]', load_var_map['w'])
```
pkl.dump可以保存多种类型的数据,用pkl.load加载,下面是加载的部分:
```python
            feature_size = sum(field_sizes)
            init_vars.append(('w', [feature_size, 1], 'fm', dtype))
            init_vars.append(('v', [feature_size, embed_size], 'fm', dtype))
            init_vars.append(('b', [1, ], 'fm', dtype))
            
            self.vars = utils.init_var_map(init_vars, init_path)
            init_w0 = tf.concat([self.vars['w'],self.vars['v']], 1)
            lower, upper = 0, field_sizes[0]
            for i in range(num_inputs):
                if(i != 0):
                    lower, upper = upper, upper+field_sizes[i]
                self.vars['embed_%d' % i] = init_w0[lower:upper]
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
```
其中的`init_var_map`函数如下：
```python
def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            value = tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype)
            var_map[var_name] = tf.Variable(value, name=var_name, dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method, name=var_name, dtype=dtype)
        elif init_method == 'fm':
            var_map[var_name] = tf.Variable(load_var_map[var_name], name=var_name, dtype=dtype)
        else:
            print('BadParam: init method', init_method)
    return var_map
```

### 模型如何使用
调试的过程如下,首先设置`algo='fm'`,获得一次项系数w和隐向量v,保存参数;然后`algo='fnn'`,进行CTR预测.
```python
# algo = 'fm'
algo = 'fnn'
if algo in {'fnn','anfm','amlp','ccpm','pnn1','pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)
    
if algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 128,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 129,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0,
        'init_path':pkl_path,
    }
    print(fnn_params)
    model = FNN(**fnn_params)
```
### 运行结果
FNN使用‘Xavier’初始化时:
```python
        for i in range(num_inputs):
            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
```

运行10次效果为：
```python
[0]     training...
[0]     evaluating...
[0]     loss (with l2 norm):0.358097    train-auc: 0.610657     eval-auc: 0.661392
[1]     training...
[1]     evaluating...
[1]     loss (with l2 norm):0.350506    train-auc: 0.624879     eval-auc: 0.679986
[2]     training...
[2]     evaluating...
[2]     loss (with l2 norm):0.348581    train-auc: 0.631834     eval-auc: 0.688470
[3]     training...
[3]     evaluating...
[3]     loss (with l2 norm):0.347268    train-auc: 0.637031     eval-auc: 0.694607
[4]     training...
[4]     evaluating...
[4]     loss (with l2 norm):0.346279    train-auc: 0.641287     eval-auc: 0.699670
[5]     training...
[5]     evaluating...
[5]     loss (with l2 norm):0.345490    train-auc: 0.644798     eval-auc: 0.703892
[6]     training...
[6]     evaluating...
[6]     loss (with l2 norm):0.344828    train-auc: 0.647727     eval-auc: 0.707407
[7]     training...
[7]     evaluating...
[7]     loss (with l2 norm):0.344262    train-auc: 0.650155     eval-auc: 0.710297
[8]     training...
[8]     evaluating...
[8]     loss (with l2 norm):0.343769    train-auc: 0.652261     eval-auc: 0.712707
[9]     training...
[9]     evaluating...
[9]     loss (with l2 norm):0.343332    train-auc: 0.654116     eval-auc: 0.714787
```
FM迭代50次后初始化:FNN运行结果为：
```python
[0]     training...
[0]     evaluating...
[0]     loss (with l2 norm):0.361066    train-auc: 0.607293     eval-auc: 0.642668
[1]     training...
[1]     evaluating...
[1]     loss (with l2 norm):0.353281    train-auc: 0.634517     eval-auc: 0.679833
[2]     training...
[2]     evaluating...
[2]     loss (with l2 norm):0.350498    train-auc: 0.640884     eval-auc: 0.688085
[3]     training...
[3]     evaluating...
[3]     loss (with l2 norm):0.347988    train-auc: 0.648423     eval-auc: 0.696806
[4]     training...
[4]     evaluating...
[4]     loss (with l2 norm):0.345739    train-auc: 0.657166     eval-auc: 0.706803
[5]     training...
[5]     evaluating...
[5]     loss (with l2 norm):0.343678    train-auc: 0.665929     eval-auc: 0.716429
[6]     training...
[6]     evaluating...
[6]     loss (with l2 norm):0.341738    train-auc: 0.674693     eval-auc: 0.725318
[7]     training...
[7]     evaluating...
[7]     loss (with l2 norm):0.339869    train-auc: 0.682893     eval-auc: 0.733139
[8]     training...
[8]     evaluating...
[8]     loss (with l2 norm):0.338055    train-auc: 0.690134     eval-auc: 0.739590
[9]     training...
[9]     evaluating...
[9]     loss (with l2 norm):0.336269    train-auc: 0.696557     eval-auc: 0.744801
```
auc值变大，明显得到改善。

## 参考
https://arxiv.org/pdf/1601.02376.pdf
