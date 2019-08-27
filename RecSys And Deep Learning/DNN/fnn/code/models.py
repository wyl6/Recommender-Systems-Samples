from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
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
        
    def restore(self, model_path):
        weight = [self.vars['w'], self.vars['v'], self.vars['b']]
        saver = tf.train.Saver(weight)
        saver.restore(self.sess, model_path)

class FM(Model):
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('v', [input_dim, factor_order], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None: 
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_vars, init_path)

            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X, self.vars['v']))
            p = 0.5 * tf.reshape(
                tf.reduce_sum(xv - tf.sparse_tensor_dense_matmul(X_square, tf.square(self.vars['v'])), 1),
                [-1, output_dim])
            xw = tf.sparse_tensor_dense_matmul(self.X, self.vars['w'])
            logits = tf.reshape(xw + self.vars['b'] + p, [-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)) + \
                        l2_w * tf.nn.l2_loss(xw) + \
                        l2_v * tf.nn.l2_loss(xv)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
            
class FNN(Model):
    def __init__(self, field_sizes=None, embed_size=10, layer_sizes=None, layer_acts=None, drop_out=None,
                 embed_l2=None, layer_l2=None, init_path=None, opt_algo='gd', learning_rate=1e-2, random_seed=None):
        Model.__init__(self)
        init_vars = []
        num_inputs = len(field_sizes)
#        for i in range(num_inputs):
#            init_vars.append(('embed_%d' % i, [field_sizes[i], embed_size], 'xavier', dtype))
        node_in = num_inputs * embed_size
        print('node_in', node_in)
        for i in range(len(layer_sizes)):
            init_vars.append(('w%d' % i, [node_in, layer_sizes[i]], 'xavier', dtype))
            init_vars.append(('b%d' % i, [layer_sizes[i]], 'zero', dtype))
            node_in = layer_sizes[i]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = [tf.sparse_placeholder(dtype) for i in range(num_inputs)]
            self.y = tf.placeholder(dtype)
            self.keep_prob_train = 1 - np.array(drop_out)
            self.keep_prob_test = np.ones_like(drop_out)
            self.layer_keeps = tf.placeholder(dtype)
            
            ##################################################
            # todo: restore w,v,b parameters from fm model
            feature_size = sum(field_sizes)
            init_vars.append(('w', [feature_size, 1], 'fm', dtype))
            init_vars.append(('v', [feature_size, embed_size], 'fm', dtype))
            init_vars.append(('b', [1, ], 'fm', dtype))
            
            self.vars = utils.init_var_map(init_vars, init_path)
            ##################################################
            # use fm paraeters to fit original interface
            init_w0 = tf.concat([self.vars['w'],self.vars['v']], 1)
            lower, upper = 0, field_sizes[0]
            for i in range(num_inputs):
                if(i != 0):
                    lower, upper = upper, upper+field_sizes[i]
                self.vars['embed_%d' % i] = init_w0[lower:upper]
            ##################################################
            print('init_vars, init_path', init_vars, init_path)
            w0 = [self.vars['embed_%d' % i] for i in range(num_inputs)]
            print('X[0].shape', self.X[0].shape)
            print('w0[0].shape', w0[0].shape)
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[i], w0[i]) for i in range(num_inputs)], 1)
            ##################################################
            l = xw

            for i in range(len(layer_sizes)):
                wi = self.vars['w%d' % i]
                bi = self.vars['b%d' % i]
                print('l.shape', 'wi.shape', 'bi.shape', l.shape, wi.shape, bi.shape)
                l = tf.nn.dropout(
                    utils.activate(
                        tf.matmul(l, wi) + bi,
                        layer_acts[i]),
                    self.layer_keeps[i])

            l = tf.squeeze(l)
            self.y_prob = tf.sigmoid(l)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=self.y))
            if layer_l2 is not None:
                self.loss += embed_l2 * tf.nn.l2_loss(xw)
                for i in range(len(layer_sizes)):
                    wi = self.vars['w%d' % i]
                    self.loss += layer_l2[i] * tf.nn.l2_loss(wi)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
