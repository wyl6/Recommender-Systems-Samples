from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle as pkl

import numpy as np
import tensorflow as tf

import utils

dtype = utils.DTYPE

class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self, fetches, X=None, y=None, mode='train'):
            feed_dict = {}
            if type(self.X) is list:
                for i in range(len(X)):
                    feed_dict[self.X[i]] = X[i]
            else:
                feed_dict[self.X] = X
            if y is not None:
                feed_dict[self.y] = y
            if self.layer_keeps is not None:
                if mode == 'train':
                    feed_dict[self.layer_keeps] = self.keep_prob_train
                elif mode == 'test':
                    feed_dict[self.layer_keeps] = self.keep_prob_test
            return self.sess.run(fetches, feed_dict)


class PNN1(Model):
    def __init__(self, 
                 field_size=None,
                 embed_size=10, 
                 layer_sizes=None,
                 layer_acts = None,
                 drop_out = None,
                 embed_l2 = None,
                 layer_l2 = None,
                 init_path = None,
                 opt_algo = 'gd',
                 learning_rate = 1e-3,
                 random_seed = None):
        
        
        Model.__init__(self)

        init_vars = []
        num_inputs = len(field_size)
        for i in range(num_inputs):
            init_vars.append(('embed_%d'%i, [field_size[i], embed_size], 'xavier', dtype))
        num_pairs = int(num_inputs*(num_inputs-1)/2);
        node_in = num_inputs*embed_size+num_pairs
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d'%i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d'%i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        
            
        self.graph = tf.Graph()
        with self.graph.as_default():
            if(random_seed is not None):
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1-np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)] # [num_inputs, field_size[i], k]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1) # [num_inputs*k]
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])  # [batch, num_inputs, k]
            
            row = [] # num_pairs
            col = [] # num_pairs
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    row.append(i)
                    col.append(j)
            
            p = tf.transpose(
                    tf.gather(
                    tf.transpose(xw3d, [1,0,2]), # [num_inputs, batch, k]
                    row), # [num_pairs, batch, k]
                    [1,0,2]) # [batch, num_pairs, k]
            
            q = tf.transpose(
                    tf.gather(
                    tf.transpose(xw3d, [1,0,2]), # [num_inputs, batch, k]
                    col), # [num_pairs, batch, k]
                    [1,0,2]) # [batch, num_pairs, k]
            
            p = tf.reshape(p, [-1, num_pairs, embed_size]) # [batch, num_pairs, k]
            q = tf.reshape(q, [-1, num_pairs, embed_size]) # [batch, num_pairs, k]
            
            ip = tf.reshape(tf.reduce_sum(p*q, [-1]), [-1, num_pairs])
            l = tf.concat([xw, ip], 1) # [num_inputs*k + num_pairs]
            
            for i in range(len(layer_sizes)):
                w = self.vars['w%d'%i]
                b = self.vars['b%d'%i]
                l = utils.activate(tf.matmul(l, w)+b, layer_acts[i])
                l = tf.nn.dropout(l, self.layer_keeps[i])
                
            print('l', l)
            l = tf.squeeze(l)  
            self.y_prob = tf.sigmoid(l)
            print('l', l)
            self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            
            if(layer_l2 is not None):
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    w = self.vars['w%d'%i]
                    self.loss += layer_l2*tf.nn.l2_loss(w)
            
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
        
class PNN2(Model):
    def __init__(self, 
                 field_size=None,
                 embed_size=10, 
                 layer_sizes=None,
                 layer_acts = None,
                 drop_out = None,
                 embed_l2 = None,
                 layer_l2 = None,
                 init_path = None,
                 opt_algo = 'gd',
                 learning_rate = 1e-3,
                 random_seed = None,
                 layer_norm=True):
        
        
        Model.__init__(self)

        init_vars = []
        num_inputs = len(field_size)
        for i in range(num_inputs):
            init_vars.append(('embed_%d'%i, [field_size[i], embed_size], 'xavier', dtype))
        num_pairs = int(num_inputs*(num_inputs-1)/2);
        node_in = num_inputs*embed_size+num_pairs
        init_vars.append(('kernel', [embed_size, num_pairs, embed_size], 'xavier', dtype))
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d'%i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d'%i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        
            
        self.graph = tf.Graph()
        with self.graph.as_default():
            if(random_seed is not None):
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1-np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)] # [num_inputs, field_size[i], k]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1) # [num_inputs*k]
            xw3d = tf.reshape(xw, [-1, num_inputs, embed_size])  # [batch, num_inputs, k]
            
            row = [] # num_pairs
            col = [] # num_pairs
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    row.append(i)
                    col.append(j)
            
            p = tf.transpose(
                    tf.gather(
                    tf.transpose(xw3d, [1,0,2]), # [num_inputs, batch, k]
                    row), # [num_pairs, batch, k]
                    [1,0,2]) # [batch, num_pairs, k]
            
            q = tf.transpose(
                    tf.gather(
                    tf.transpose(xw3d, [1,0,2]), # [num_inputs, batch, k]
                    col), # [num_pairs, batch, k]
                    [1,0,2]) # [batch, num_pairs, k]
            
            p = tf.reshape(p, [-1, num_pairs, embed_size]) # [b, p, k]
            q = tf.reshape(q, [-1, num_pairs, embed_size]) # [b, p, k]
            
            # k全为1时，就是严格按照公式
            p = tf.expand_dims(p, 1) # [batch, 1, p, k]
            k = self.vars['kernel'] # [k, p, k]
            ip = tf.multiply(k, p) # [batch, k, p, k]
            ip = tf.reduce_sum(ip, axis=-1) # [batch, k, p]
            ip = tf.transpose(ip, [0, 2, 1]) # [batch, p, k]
            ip = tf.multiply(ip, q) # [batch, p, k]
            ip = tf.reduce_sum(ip, axis=-1) # [batch, p]
            
            l = tf.concat([xw, ip], 1) # [num_inputs*k + num_pairs]
            
            for i in range(len(layer_sizes)):
                w = self.vars['w%d'%i]
                b = self.vars['b%d'%i]
                l = utils.activate(tf.matmul(l, w)+b, layer_acts[i])
                l = tf.nn.dropout(l, self.layer_keeps[i])
                
            print('l', l)
            l = tf.squeeze(l)  
            self.y_prob = tf.sigmoid(l)
            print('l', l)
            self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            
            if(layer_l2 is not None):
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    w = self.vars['w%d'%i]
                    self.loss += layer_l2*tf.nn.l2_loss(w)
            
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
        