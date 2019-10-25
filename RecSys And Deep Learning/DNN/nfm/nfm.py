# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from DataParse import DataParse
from tqdm import tqdm
np.random.seed(2018)

class AFM(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 feature_size,
                 field_size,
                 embedding_size=8,
                 attention_size=10,
                 dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 1],
                 deep_layers_activation=tf.nn.relu,
                 epochs=1,
                 batch_size=128,
                 learning_rate=0.001,
                 optimizer_type='adam',
                 batch_norm=False,
                 batch_norm_decay=0.995,
                 verbose=True,
                 random_seed=2018,
                 loss_type='logloss',
                 metric_type='auc',
                 l2_reg=0.0):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.metric_type = metric_type

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.weights, self.biases = self.init_weights(len(self.deep_layers))

            with tf.name_scope('Embedding_Layer'):
                self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feature_index)  # [None, field_size, embedding_size]
                feat_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1])  # [None, field_size, 1]
                self.embeddings = tf.multiply(self.embeddings, feat_value)  # [None, field_size, embedding_size]

            with tf.name_scope('linear_part'):
                self.linear_part = tf.nn.embedding_lookup(self.weights['linear_w'],self.feature_index)  # [None, field_size, 1]
                self.linear_part = tf.reduce_sum(tf.multiply(self.linear_part, feat_value), axis=2)  # [None, field_size]
                self.linear_part = tf.nn.dropout(self.linear_part, self.dropout_keep_fm[0])  # [None, field_size]
                self.linear_out = tf.reduce_sum(self.linear_part, axis=1, keepdims=True) # [None, 1]
                self.w0 = tf.multiply(self.biases['w0'], tf.ones_like(self.linear_out)) # [None, 1]

#            with tf.name_scope('Pair-wise_Interaction_Layer'):
#                pair_wise_product_list = []
#                for i in range(self.field_size):
#                    for j in range(i + 1, self.field_size):
#                        pair_wise_product_list.append(tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :])) # [None, embedding_size]
#                self.pair_wise_product = tf.stack(pair_wise_product_list) # [embedding_size*(embedding_size + 1)/2, None, embedding_size]
#                self.pair_wise_product = tf.transpose(self.pair_wise_product, perm=[1, 0, 2], name='pair_wise_product') # [None, field_size*(field_size - 1)/2, embedding_size]
#                self.pair_wise_product = tf.reduce_sum(self.pair_wise_product, axis=1); # [None, embedding_size]
#                self.fully_out = tf.nn.dropout(self.pair_wise_product, self.dropout_keep_fm[1]) # [None, embedding_size]
                
            with tf.variable_scope('interaction_layer'):
                self.sum_square_emb = tf.square(tf.reduce_sum(self.embeddings, axis=1)) # [None, embedding_size]
                self.square_sum_emb = tf.reduce_sum(tf.square(self.embeddings), axis=1) # [None, embedding_size]
                self.fully_out = 0.5 * tf.subtract(self.sum_square_emb, self.square_sum_emb) # [None, embedding_size]
                
            with tf.name_scope('fully_layer'):
                for i in range(len(self.deep_layers)):
                    self.fully_out = tf.add(tf.matmul(self.fully_out, self.weights[i]), self.biases[i])
                    self.fully_out = self.deep_layers_activation(self.fully_out)
                    if(self.batch_norm):
                        self.fully_out = self.batch_norm_layer(self.fully_out, self.train_phase)
                    self.fully_out = tf.nn.dropout(self.fully_out, keep_prob=self.dropout_fm[i])
            
            with tf.name_scope('out'):
                self.out = tf.add_n([self.w0, self.linear_out, self.fully_out]) # # yAFM = w0 + wx + f(x)

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.95).minimize(self.loss)
            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # Initialization
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def init_weights(self, num_layers):
        self.weights = {}
        self.biases = {}
        # embedding weights
        self.weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        # linear weights and biases
        self.biases['w0'] = tf.Variable(tf.constant(0.1), name='w0')
        self.weights['linear_w'] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='linear_w')
        # fully connected weights
        x_size = self.embedding_size
        for i in range(num_layers):
            self.weights[i] = tf.Variable(tf.random_uniform([x_size, self.deep_layers[i]],0.0, 1.0), name='w_%d'%i)
            self.biases[i] = tf.Variable(0.1, name='b_%d'%i)
            x_size = self.deep_layers[i]
        return self.weights, self.biases
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        batch_out = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return batch_out

    def fit(self, feat_index, feat_val, label, valid_feat_index=None, valid_feat_val=None, valid_label=None):
        '''
        :param feat_index: [[idx1_1, idx1_2,...], [idx2_1, idx2_2,...],...]
                            idxi_j is the feature index of feature field j of sample i in the training set
        :param feat_val: [[value1_1, value1_2,...], [value2_1, value2_2,...]...]
                        valuei_j is the feature value of feature field j of sample i in the training set
        :param label: [[label1], [label2], [label3], [label4],...]
        :return: None
        '''
        has_valid = valid_feat_index is not None
        total_time = 0
        for epoch in range(self.epochs):
            start_time = time.time()
            for i in tqdm(range(0, len(feat_index), self.batch_size), ncols=100):
                feat_index_batch = feat_index[i: i + self.batch_size]
                feat_val_batch = feat_val[i: i + self.batch_size]
                batch_y = label[i: i + self.batch_size]

                feed_dict = {
                    self.feature_index: feat_index_batch,
                    self.feature_value: feat_val_batch,
                    self.label: batch_y,
                    self.dropout_keep_fm: self.dropout_fm,
                    self.train_phase: True
                }
                cost, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            end_time = time.time()
            
            if self.verbose:
                if has_valid:
                    train_metric = self.evaluate(feat_index, feat_val, label)
                    valid_metric = self.evaluate(valid_feat_index, valid_feat_val, valid_label)
                    print('[%s] train-%s=%.4f, valid-%s=%.4f [%.1f s]' % (
                        epoch + 1, self.metric_type, train_metric, self.metric_type, valid_metric,
                        end_time - start_time))
                else:
                    train_metric = self.evaluate(feat_index, feat_val, label)
                    print('[%s] train-%s=%.4f [%.1f s]' % (
                        epoch + 1, self.metric_type, train_metric, end_time - start_time))
            total_time = total_time + end_time - start_time

        print('cost total time=%.1f s' % total_time)

    def predict(self, feat_index, feat_val):
        feed_dict = {
            self.feature_index: feat_index,
            self.feature_value: feat_val,
            self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
            self.train_phase: False
        }
        y_pred = self.sess.run(self.out, feed_dict=feed_dict)
        return y_pred

    def evaluate(self, feat_index, feat_val, label):
        y_pred = self.predict(feat_index, feat_val)
        print(type(y_pred), type(label))
        if self.metric_type == 'auc':
            return roc_auc_score(label, y_pred)
        elif self.metric_type == 'logloss':
            return log_loss(label, y_pred)
        elif self.metric_type == 'acc':
            return accuracy_score(label, (y_pred > 0.5).astype('int32'))

if __name__ == '__main__':
    print('read dataset...')
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    y_train = pd.read_csv('./data/y_train.csv')
    y_val = pd.read_csv('./data/y_test.csv')

    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                        'native_country']

    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    train_feature_index, train_feature_val = dataParse.parse(train)
    test_feature_index, test_feature_val = dataParse.parse(test)

    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    model = AFM(feature_size=dataParse.feature_size,
                field_size=dataParse.field_size,
                metric_type='acc')

    model.fit(train_feature_index, train_feature_val, y_train)
    test_metric = model.evaluate(test_feature_index, test_feature_val, y_val)
    print('test-auc=%.4f' % test_metric)
