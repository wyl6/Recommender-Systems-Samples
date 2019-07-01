# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils

from models.model_base import BaseModel
from util.data_utils import BatchGenerator
from util.metrics import mae, rmse

class SVD(BaseModel):
    def __init__(self, config, sess):
        super(SVD, self).__init__(config)
        self.sess = sess
    
    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            users = tf.placeholder(tf.int32, shape=[None,], name='users')
            items = tf.placeholder(tf.int32, shape=[None,], name='items')
            ratings = tf.placeholder(tf.float32, shape=[None,], name='ratings')
        return users, items, ratings
    
    def create_constants(self, mu):
        with tf.variable_scope('constant'):
            mu = tf.constant(mu, shape=[], dtype=tf.float32)
        return mu
    
    def create_user_terms(self, users):
        num_users = self.num_users
        num_factors = self.num_factors
        
        with tf.variable_scope('user'):
            user_embeddings = tf.get_variable(
                    name='embedding',
                    shape=[num_users, num_factors],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_p_u))
            user_bias = tf.get_variable(
                    name='bias',
                    shape=[num_users, ],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_u))
            p_u = tf.nn.embedding_lookup(
                    user_embeddings,
                    users,
                    name='p_u')
            b_u = tf.nn.embedding_lookup(
                    user_bias,
                    users,
                    name='b_u')
        return p_u, b_u
    
    def create_item_terms(self, items):
        num_items = self.num_items
        num_factors = self.num_factors
        
        with tf.variable_scope('item'):
            item_embeddings = tf.get_variable(
                    name='embedding',
                    shape=[num_items, num_factors],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_q_i))
            item_bias = tf.get_variable(
                    name='bias',
                    shape=[num_items, ],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_i))
            q_i = tf.nn.embedding_lookup(
                    item_embeddings,
                    items,
                    name='q_i')
            b_i = tf.nn.embedding_lookup(
                    item_bias,
                    items,
                    name='b_i')
        return q_i, b_i
    
    def create_prediction(self, mu, b_u, b_i, p_u, q_i):
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(tf.multiply(p_u, q_i), axis=1)
            pred = tf.add_n([b_u, b_i, pred])
            pred = tf.add(pred, mu, name='pred')
        return pred
    
    def create_loss(self, pred, ratings):
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(ratings, pred), name='loss')
        return loss
    
    def create_optimizer(self, loss):
        with tf.variable_scope('optimizer'):
            objective = tf.add(loss,
                              tf.add_n(tf.get_collection(
                                      tf.GraphKeys.REGULARIZATION_LOSSES)),name='objective')
            try:
                optimizer = tf.contrib.keras.optimizers.Nadam().minimize(objective, name='optimizer')
            except:
                optimizer = tf.train.AdamOptimizer().minimize(objective, name='optimizer')
        return optimizer
        
    def build_graph(self, mu):
        mu = self.create_constants(mu)
        self.users, self.items, self.ratings = self.create_placeholders()
        
        p_u, b_u = self.create_user_terms(self.users)
        q_i, b_i = self.create_item_terms(self.items)
        
        self.pred = self.create_prediction(mu, b_u, b_i, p_u, q_i)
        loss = self.create_loss(self.ratings, self.pred)
        self.optimizer = self.create_optimizer(loss)
        self.built = True
        
    def run_train(self, x, y, epoches, batch_size, val_data):
        train_gen = BatchGenerator(x, y, batch_size=batch_size)
        steps_per_epoch = np.ceil(train_gen.length / batch_size).astype(int)
        
        self.sess.run(tf.global_variables_initializer())
        
        for i in range (1, epoches+1):
            print('Epoch {} / {}'.format(i, epoches))
            pbar = utils.Progbar(steps_per_epoch)
            
            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]
                
                self.sess.run(self.optimizer,
                              feed_dict={
                                      self.users: users,
                                      self.items: items,
                                      self.ratings: ratings})
                pred = self.predict(batch[0])
                
                update_values = [
                        ('rmse', rmse(ratings, pred)),
                        ('mae', mae(ratings, pred))]
                
                
            if(val_data is not None and step == steps_per_epoch):
                valid_x, valid_y = val_data
                valid_pred = self.predict(valid_x)
                update_values += [
                        ('val_rmse', rmse(valid_y, valid_pred)),
                        ('val_mae', mae(valid_y, valid_pred))]
                pbar.update(step, value=update_values, force=(step==steps_per_epoch))
    
    def train(self, x, y, epoches=100, batch_size=1024, val_data=None):
        if (x.shape[0] != y.shape[0] or x.shape[1] != 2):
            raise ValueError('The shape 0f x should be (samples, 2) and '
            'the shape of y should be (samples, 1).')
        if(not self.built):
            self.build_graph(np.mean(y))
    
        self.run_train(x, y, epoches, batch_size, val_data)
    
    def predict(self, x):
        if(not self.built):
            raise RuntimeError('The model must be trained'
                               'before prediciton')
        if(x.shape[1] != 2):
            raise ValueError('The shape of x should be (samples, 2)')
        
        pred = self.sess.run(self.pred, feed_dict={self.users:x[:,0], self.items:x[:,1]})
        
        pred = pred.clip(min=self.min_value, max=self.max_value)
        
        return pred
    
    