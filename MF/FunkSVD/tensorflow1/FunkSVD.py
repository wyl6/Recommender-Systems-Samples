import numpy as np
import tensorflow as tf
import time
import os
from util import rmse, status_printer

def inference_svd(batch_user, batch_item, num_user, num_item, dim=5):
    
    with tf.name_scope('Declaring_variables'):
        
        ## obtain global bias, user bias, item bias
        w_bias_user = tf.get_variable('num_bias_user', shape=[num_user])
        w_bias_item = tf.get_variable('num_bias_item', shape=[num_item])
        bias_global = tf.get_variable('bias_global', shape=[])
        bias_user = tf.nn.embedding_lookup(w_bias_user, 
                                           batch_user, 
                                           name='batch_bias_user')
        bias_item = tf.nn.embedding_lookup(w_bias_item, 
                                           batch_item, 
                                           name='batch_bias_imte')
        
        ## obtain user embedding, item embedding
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        w_user = tf.get_variable('num_embed_user', 
                                 shape=[num_user, dim], 
                                 initializer=initializer)
        w_item = tf.get_variable('num_embed_item', 
                                 shape=[num_item, dim], 
                                 initializer=initializer)
        embed_user = tf.nn.embedding_lookup(w_user, 
                                            batch_user, 
                                            name='batch_embed_user')
        embed_item = tf.nn.embedding_lookup(w_item, 
                                            batch_item, 
                                            name='batch_embed_item')
        
    with tf.name_scope('Prediction_regularizer'):
        
        ## obtain r_ij
        infer = tf.reduce_sum(tf.multiply(embed_user, embed_item), 1) ## p_i*q_j
        
        ## obtain bias
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name='svd_inference')
        
        ## obtain regularizer
        l2_user = tf.sqrt(tf.nn.l2_loss(embed_user))
        l2_item = tf.sqrt(tf.nn.l2_loss(embed_item))
        l2_sum = tf.add(l2_user, l2_item)
        bias_user_sq = tf.square(bias_user)
        bias_item_sq = tf.square(bias_item)
        bias_sum = tf.add(bias_user_sq, bias_item_sq)
        regularizer = tf.add(l2_sum, bias_sum, name='svd_regularizer')
        
        regularizer = tf.add(tf.nn.l2_loss(embed_user), 
                             tf.nn.l2_loss(embed_item), 
                             name="svd_regularizer")
        
    with tf.name_scope('Dict_values'):
        dict_of_values = {'infer':infer,
                          'regularizer':regularizer,
                          'w_user':w_user,
                          'w_item':w_item
                          }
    return dict_of_values


def loss_function(infer, regularizer, batch_rate, reg):
    '''
    calculate loss = loss_l2+loss_regularizer
    '''
    loss_l2 = tf.square(tf.subtract(infer, batch_rate))
    reg = tf.constant(reg, dtype=tf.float32, name='reg')
    loss_regularizer = tf.multiply(regularizer, reg)
    loss = tf.add(loss_l2, loss_regularizer)
    return loss



class SVD(object):
    '''
    
    '''
    def __init__(self, 
                 num_of_users,
                 num_of_items,
                 train_batch_generator,
                 valid_batch_generator,
                 test_batch_generator,
                 finder=None,
                 model='svd'):
        self.num_of_users = num_of_users
        self.num_of_items = num_of_items
        self.train_batch_generator = train_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.test_batch_generator = test_batch_generator
        self.finder = finder
        self.model = model
        self.general_duration = 0
        self.num_steps = 0
        self.dimension = None
        self.reg = None
        self.best_acc_test = float('inf')
    
    def set_graph(self, 
                  hp_dim, 
                  hp_reg, 
                  learning_rate, 
                  momentum_factor):
        '''
        
        '''
        self.dimension = hp_dim
        self.reg = hp_reg
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.graph = tf.Graph()
        
        with self.graph.as_default():

            self.batch_user = tf.placeholder(tf.int32, 
                                             shape=[None], 
                                             name='id_user')
            self.batch_item = tf.placeholder(tf.int32, 
                                             shape=[None], 
                                             name='id_item')
            self.batch_rate = tf.placeholder(tf.float32, 
                                             shape=[None], 
                                             name='true_rate')
            svd_model = inference_svd(self.batch_user,
                                       self.batch_item,
                                       num_user=self.num_of_users,
                                       num_item=self.num_of_items,
                                       dim=hp_dim)
            self.infer = svd_model['infer']
            self.regularizer = svd_model['regularizer']
            global_step = tf.train.get_or_create_global_step()
        
            with tf.name_scope('loss'):
                self.loss = loss_function(infer=self.infer,
                                          regularizer=self.regularizer,
                                          batch_rate=self.batch_rate,
                                          reg=hp_reg)
            
            with tf.name_scope('training'):
                global_step = tf.train.get_or_create_global_step()
                assert(global_step is not None)
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_factor)
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)
            
            with tf.name_scope('saver'):
                self.saver =  tf.train.Saver()
                save_dir = 'checkpoints/'
                if (not os.path.exists(save_dir)): os.makedirs(save_dir)
                self.save_path = os.path.join(save_dir, 'best_validation')
            
            # minibatch accuracy using rmse
            with tf.name_scope('accuracy'):
                difference = tf.pow(tf.subtract(self.infer, self.batch_rate), 2)
                self.acc_op = tf.sqrt(tf.reduce_mean(difference))
        
    def training(self, 
                 hp_dim, 
                 hp_reg, 
                 learning_rate, 
                 momentum_factor, 
                 num_steps, 
                 verbose=True):
        self.set_graph(hp_dim, 
                       hp_reg, 
                       learning_rate, 
                       momentum_factor)
        self.num_steps = num_steps
        marker = ' '
        
        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            if(verbose):
                print('{} {} {} {}'.format('step','batch_error','test_error','elapsed_time'))
            else:
                print('Training')
            initial_time = time.time()
            for step in range(num_steps):
                users, items, rates = self.train_batch_generator.get_batch()
                f_dict = {self.batch_user:users,
                          self.batch_item:items,
                          self.batch_rate:rates,}
                _, batch_refer, loss, train_error = sess.run([self.train_op,
                                                              self.infer,
                                                              self.loss,
                                                              self.acc_op],
                                                              feed_dict=f_dict)
                if(not verbose):
                    percentage = (step/num_steps)*100
                    if(percentage % 10 == 0):
                        print(int(percentage), '%', end='...')
                
                if(step%1000 == 0):
                    users, items, rates = self.valid_batch_generator.get_batch()
                    f_dict = {self.batch_user:users,
                          self.batch_item:items,
                          self.batch_rate:rates,}
                    batch_refer = sess.run(self.infer, feed_dict=f_dict)
                    test_error = rmse(batch_refer, rates)
                    if(test_error < self.best_acc_test):
                        self.best_test_error = test_error
                        marker = '*'
                        self.saver.save(sess=sess, save_path=self.save_path)
                        
                    if(verbose):
                        print('{:3d} {:f} {:f}{:s} {:f}'.format(step,
                              train_error,
                              test_error,
                              marker,
                              time.time()-initial_time))
            self.general_duration = time.time()-initial_time
    
    def print_status(self):
        status_printer(self.num_steps, self.general_duration)
        
    def prediction(self, 
                   list_users=None,
                   list_items=None,
                   show_valid=False):
        if(self.dimension is None and self.regularizer is None):
            print('You can not predict without training!')
        else:
            self.set_graph(self.dimension,
                           self.reg,
                           self.learning_rate,
                           self.momentum_factor)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                
                if(show_valid):
                    users, items, rates = self.test_batch_generator.get_batch()
                    f_dict = {self.batch_user:users, 
                              self.batch_item:items, 
                              self.batch_rate:rates}
                    valid_error = sess.run(self.acc_op, feed_dict=f_dict)
                    return valid_error
                else:
                    f_dict = {self.batch_user:list_users, 
                              self.batch_item:list_items}
                    refer = sess.run(self.infer, feed_dict=f_dict)
                    return refer
                    