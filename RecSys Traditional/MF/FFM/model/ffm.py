import tensorflow as tf
import numpy as np

class FFM(object):
    '''
    field_nums: F = 6
    feature_nums: N = 24
    embedding_size:  k= 3
    '''
    def __init__(self, hparams, df_i, df_v):
        self.hparams = hparams
        tf.set_random_seed(self.hparams.seed)
        self.line_result = self.line_section(df_i, df_v)
        self.fm_result = self.fm_section(df_i, df_v)
        print(self.line_result, self.fm_result)
        self.logits = self.line_result+self.fm_result
    
    def line_section(self, df_i, df_v):
        '''
        # df_i records the positon of i_th feature in 24
        # df_v records the value of i_th feature
        '''
        with tf.variable_scope('line'):
            weights = tf.get_variable('weights',
                                      shape=[self.hparams.feature_nums, 1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.glorot_uniform())
            batch_weights = tf.nn.embedding_lookup(weights, df_i) # (*, 6, 1)
            batch_weights = tf.squeeze(batch_weights, axis=2) # remove dimensions of size 1 ==> (*, 6)
            line_result = tf.multiply(df_v, batch_weights, name='line_w_x')
            biase = tf.get_variable('biase',
                                    shape=[1,1],
                                    dtype=tf.float32,
                                    initializer=tf.initializers.zeros())
            line_result = tf.add(tf.reduce_mean(line_result, axis=1, keepdims=True), biase) # (*, 1)
            
        return line_result
    
    def fm_section(self, df_i, df_v):
        with tf.variable_scope('fm'):
            embedding = tf.get_variable('embedding',
                                        shape=[self.hparams.field_nums,
                                               self.hparams.feature_nums,
                                               self.hparams.embedding_size],
                                               dtype=tf.float32,
                                               initializer=tf.initializers.random_normal())
        fm_result = None
        for i in range(self.hparams.field_nums):
            for j in range(i+1, self.hparams.field_nums):
                vi_fj = tf.nn.embedding_lookup(embedding[j], df_i[:, i]) # (*, k)
                vj_fi = tf.nn.embedding_lookup(embedding[i], df_i[:, j]) # (*, k)
                wij = tf.reduce_sum(tf.multiply(vi_fj, vj_fi), axis=1, keepdims=True) # (*, 1)
                
                x_i = tf.expand_dims(df_v[:, i], 1) # (*, 1)
                x_j = tf.expand_dims(df_v[:, j], 1) # (*, 1)
                xij = tf.multiply(x_i, x_j) # (*, 1)
                
                if(fm_result == None):
                    fm_result = tf.multiply(wij, xij) # (*, 1)
                else:
                    fm_result += tf.multiply(wij, xij)
        return fm_result
                