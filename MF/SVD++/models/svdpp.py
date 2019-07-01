# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils
    
from models.svd import SVD
from util.data_utils import BatchGenerator
from util.metrics import mae, rmse

def convert_to_sparse_format(x):
    'convert a list of lists into sparse format'
    sparse = {'indices':[], 'values':[]}
    
    for row, x_i in enumerate(x):
        for col, x_ij in enumerate(x_i):
            sparse['indices'].append((row, col))
            sparse['values'].append(x_ij)
            
    max_col = np.max([len(x_i) for x_i in x]).astype(np.int32)
    sparse['dense_shape'] = (len(x), max_col)
    
    return sparse

def get_implicit_feedback(x, num_users, num_items, dual):
    if(not dual):
        N = [[] for u in range(num_users)]
        for u, i in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
        return convert_to_sparse_format(N)
    else:
        N = [[] for u in range(num_users)]
        H = [[] for u in range(num_items)]
        for u, i in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
            H[i].append(u)
        return convert_to_sparse_format(N), convert_to_sparse_format(H)

class SVDPP(SVD):
    
    def __init__(self, config, sess, dual=False):
        super(SVDPP, self).__init__(config, sess)
        self.dual = dual
        
    def create_implicit_feedback(self, implicit_feedback, dual=False):
        with tf.variable_scope('implicit_feedback'):
            if(not dual):
                N = tf.SparseTensor(**implicit_feedback)
                return N
            else:
                N = tf.SparseTensor(**implicit_feedback[0])
                H = tf.SparseTensor(**implicit_feedback[1])
                return N, H

    def create_user_terms(self, users, N):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors
        
        p_u, b_u = super(SVDPP, self).create_user_terms(users)
        
        with tf.variable_scope('user'):
            implicit_feedback_embeddings = tf.get_variable(
                    name='implicit_feedback_embeddings',
                    shape=[num_items, num_factors],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_y_u))

        y_u = tf.gather(
                tf.nn.embedding_lookup_sparse(
                        implicit_feedback_embeddings,
                        N,
                        sp_weights=None,
                        combiner='sqrtn'),
                users,
                name='y_u')
#        print('create_user_terms--------------------------------------')
#        print('implicit_feedback_embeddings', implicit_feedback_embeddings.shape)
#        print('N', N.shape)
#        print('y_u.shape', y_u.shape)
        return p_u, b_u, y_u

    def create_item_terms(self, items, H=None):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors
        
        q_i, b_i = super(SVDPP, self).create_item_terms(items)
        
        if(H is None):
            return q_i, b_i
        else:
            with tf.variable_scope('item'):
                implicit_feedback_embeddings = tf.get_variable(
                        name='implicit_feedback_embeddings',
                        shape=[num_users, num_factors],
                        initializer=tf.zeros_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(self.reg_y_u))
    
            g_i = tf.gather(
                    tf.nn.embedding_lookup_sparse(
                            implicit_feedback_embeddings,
                            H,
                            sp_weights=None,
                            combiner='sqrtn'),
                    items,
                    name='g_i')
            return q_i, b_i, g_i

    def create_prediction(self, mu, b_u, b_i, p_u, q_i, y_u, g_i=None):
#        print('create_prediction--------------------------------------')
#        print('p_u', p_u.shape)
#        print('y_u', y_u.shape)
#        print('q_i', q_i.shape)
        with tf.variable_scope('prediction'):
            if g_i is None:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), q_i),
                    axis=1)
            else:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), tf.add(q_i, g_i)),
                    axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def build_graph(self, mu, implicit_feedback):
        mu = super(SVDPP, self).create_constants(mu)
        self.users, self.items, self.ratings = super(SVDPP, self).create_placeholders()
        
        if not self.dual:
            N = self.create_implicit_feedback(implicit_feedback)

            p_u, b_u, y_u = self.create_user_terms(self.users, N)
            q_i, b_i = self.create_item_terms(self.items)

            self.pred = self.create_prediction(mu, b_u, b_i, p_u, q_i, y_u)
        else:
            N, H = self.create_implicit_feedback(implicit_feedback, True)

            p_u, b_u, y_u = self.create_user_terms(self.users, N)
            q_i, b_i, g_i = self.create_item_terms(self.items, H)

            self.pred = self.create_prediction(mu, b_u, b_i, p_u, q_i, y_u, g_i)

        loss = super(SVDPP, self).create_loss(self.ratings, self.pred)

        self.optimizer = super(SVDPP, self).create_optimizer(loss)

        self.built = True

    def train(self, x, y, epoches=100, batch_size=1024, val_data=None):
        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self.built:
            implicit_feedback = get_implicit_feedback(
                x, self.num_users, self.num_items, self.dual)
            self.build_graph(np.mean(y), implicit_feedback)

        self.run_train(x, y, epoches, batch_size, val_data)
        
