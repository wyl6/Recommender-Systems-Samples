# -*- coding: utf-8 -*-

import numpy as np
import util
import FunkSVD

class SVDmodel(object):
    def __init__(self, df, users, items, ratings):
        self.df = df
        self.users = users
        self.items = items
        self.ratings = ratings
        self.size = len(df)
        self.num_of_users = max(self.df[self.users])+1
        self.num_of_items = max(self.df[self.items])+1
        self.train, self.valid, self.test = self.data_separation()
        
    def data_separation(self):
        rows = len(self.df)
        random_ids = np.random.permutation(rows)
        random_df = self.df.iloc[random_ids].reset_index(drop=True)
        split_index = int(rows*0.7)
        new_split = int(rows*0.9)
        
        df_train = random_df[0:split_index].reset_index(drop=True)
        df_valid = random_df[split_index:new_split].reset_index(drop=True)
        df_test = random_df[new_split:].reset_index(drop=True)
        
        return df_train, df_valid, df_test
    
    def training(self,
                 hp_dim=20,
                 hp_reg=0.1,
                 learning_rate=0.01,
                 momentum_factor=0.9,
                 batch_size=100,
                 num_steps=1000,
                 verbose=True):
        self.train_batches = util.BatchGenerator(self.train,
                                                 batch_size,
                                                 self.users,
                                                 self.items,
                                                 self.ratings)
        self.valid_batches = util.BatchGenerator(self.valid,
                                                 batch_size,
                                                 self.users,
                                                 self.items,
                                                 self.ratings)
        self.test_batches = util.BatchGenerator(self.test,
                                                 batch_size,
                                                 self.users,
                                                 self.items,
                                                 self.ratings)
        self.tf_counterpart = FunkSVD.SVD(self.num_of_users,
                                          self.num_of_items,
                                          self.train_batches,
                                          self.valid_batches,
                                          self.test_batches)
        self.tf_counterpart.training(hp_dim,
                                     hp_reg,
                                     learning_rate,
                                     momentum_factor,
                                     num_steps,
                                     verbose)
        self.duration = round(self.tf_counterpart.general_duration, 2)
        if(verbose):
            self.tf_counterpart.print_status()

    def valid_prediction(self):
        return self.tf_counterpart.prediction(show_valid=True)
    
    def prediction(self, list_of_users, list_of_items, show_valid=False):
        return self.tf_counterpart.prediction(list_of_users, 
                                              list_of_items, 
                                              show_valid=show_valid)
    