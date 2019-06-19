# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

def status_printer(num_steps, general_duration):
    sec = timedelta(seconds=int(general_duration))
    d_time = datetime(1,1,1)+sec # year, month, day, + second
    print(' ')
    print('The duration of the whole training with %s steps is %.2f seconds,'
          %(num_steps, general_duration))
    print('Which is equal to: %d:%d:%d:%d'
          %(d_time.day-1, d_time.hour, d_time.minute, d_time.second), end=' ')
    print("Days:Hours:Min:Sec")
    
def mse(y_true, y_pred):
    '''
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    '''
    assert len(y_true) == len(y_pred)
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def load_dataframe(path, sep="::"):
    '''
    Assuming that we are only working with the datasets from movielens.
    '''
    if(path[-3:] == 'dat'):
        col_names = ['userId', 'movieId', 'rating', 'st']
        raw_df = pd.read_csv(path, sep=sep, names=col_names, engine='python')
    elif(path[-3:] == 'csv'):
        raw_df = pd.read_csv(path)
    
    raw_df['userId'] = raw_df['userId']-1
    raw_df['movieId'] = raw_df['movieId']-1
    raw_df['user'] = raw_df['userId'].astype(np.int32)
    raw_df['item'] = raw_df['movieId'].astype(np.int32)
    raw_df['rating'] = raw_df['rating'].astype(np.float32)
    df = raw_df[['user','item','rating']]
    return df

def counter_intersection(df1, df2, df3):
    '''
    Given three dataframes, return the intersection 
    between (df1,df2),(df1,df3), (df2, df3)
    '''
    from hashlib import sha1
    raw1, raw2, raw3 = np.array(df1, df2, df3)
    array1, array2, array3 = raw1.copy(), raw2.copy(), raw3.copy()
    set1 = set([sha1(i).hexdigest() for i in array1])
    set2 = set([sha1(i).hexdigest() for i in array2])
    set3 = set([sha1(i).hexdigest() for i in array3])
    dic = {}
    dic['1-2'] = len(set1.intersection(set2))
    dic['1-3'] = len(set1.intersection(set3))
    dic['2-3'] = len(set2.intersection(set3))
    return dic

class BatchGenerator(object):
    
    def __init__(self, df, batch_size, users, items, ratings):
        self.batch_size = batch_size
        self.users = np.array(df[users])
        self.items = np.array(df[items])
        self.ratings = np.array(df[ratings])
        self.num_cols = len(df.columns)
        self.size = len(df)
        
    def get_batch(self):
        indices = np.random.randint(0, self.size, self.batch_size)
        users = self.users[indices]
        items = self.items[indices]
        ratings = self.ratings[indices]
        return users, items, ratings
    
class ItemFinder(object):
    '''
    Class that given one user, it returns 
    the array of all items rated by that user.
    '''
    def __init__(self, df, users, items, ratings, nsvd_size):
        self.users = users
        self.items = items
        self.df = df
        self.dic = {}
        self._set_item_dic(nsvd_size)
        
    def get_item(self, user):
        user_df = self.df[self.df[self.users] == user]
        user_items = np.array(user_df[self.items])
        return user_items
    
    def set_item_dic(self, size_command='mean'):
        if(not self.dic):
            all_users = self.df[self.users].unique()
            new_item = max(self.df[self.items].unique())+1
            sizes = {}
            print('\nWriting dic ...')
            for user in all_users:
                items_rated = self.get_item(user)
                self.dic[user] = items_rated
                sizes[user] = len(items_rated)
            if(size_command == 'max'):
                self.size = np.max(list(sizes.values()))
            elif(size_command == 'mean'):
                self.size = int(np.mean(list(sizes.values())))
            elif(size_command == 'min'):
                self.size = np.min(list(sizes.values()))
            print('Resizing...')
            for user in all_users:
                if(self.size <= sizes[user]):
                    self.dic[user] = self.dic[user][0:self.size]
                else:
                    diff = self.size-sizes[user]
                    tail = np.array([new_item for i in range(diff)])
                    result = np.concatenate((self.dic[user], tail), axis=0)
                    self.dic[user] = result
            print('Generating size factors...')
            if(size_command == 'max'):
                for user in all_users:
                    sizes[user] = 1/np.sqrt(sizes[user])
                self.size_factor = sizes
            else:
                self.size_factor = dict.fromkeys(sizes, 1/np.sqrt(self.size))
        else:
            pass
    
    
    
    def get_item_array(self, users):
        return np.array([self.dic[user] for user in users])
    
    def get_size_factor(self, users):
        return np.array(self.size_factor[user] for user in users)
    
    
    
    

if(__name__ == '__main__'):
    num_steps = 1
    general_duration = 10
    status_printer(num_steps, general_duration)