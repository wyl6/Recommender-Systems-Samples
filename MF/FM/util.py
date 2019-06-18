# -*- coding: utf-8 -*-
from itertools import count
import numpy as np
from scipy.sparse import csr
from collections import defaultdict
import pandas as pd

def vectorize_dic(dic, ix=None, p=None):
    if(ix == None):
        d = count(0)
        ix = defaultdict(lambda:next(d))
        
    n = len(list(dic.values())[0])
    g = len(list(dic.keys()))
    nz = n*g
    
    col_ix = np.empty(nz, dtype=int)
    
    i = 0
    for k, lis in dic.items():
        col_ix[i::g] = [ix[str(k)+str(el)] for el in lis]
        i += 1
    
    row_ix = np.repeat(np.arange(n), g)
    data = np.ones(nz);print('data.shape ', data.shape)

    if(p == None):
        p = len(ix)
        
    ixx = np.where(col_ix < p)
    
    return csr.csr_matrix(
        (data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


        
def load_dataset():
    cols = ['user','item','rating','timestamp']
    train = pd.read_csv('../../../Datasets/MovieLens/ml-100k/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('../../../Datasets/MovieLens/ml-100k/ua.test', delimiter='\t', names=cols)
    
    
    x_train, ix = vectorize_dic({'users':train.user.values,'items':train.item.values})
    x_test, ix = vectorize_dic({'users':test.user.values, 'items':test.item.values}, ix, x_train.shape[1])
    
    y_train = train.rating.values
    y_test = test.rating.values
    
    x_train = x_train.todense()
    x_test = x_test.todense()
    
    return x_train, x_test, y_train, y_test



def batcher(X, Y=None, batch_size=-1):
    n_samples = X.shape[0]
    
    if(batch_size == -1):
        batch_size = n_samples
    if(batch_size < 1):
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i+batch_size, n_samples)
        ret_x = X[i:upper_bound]
        ret_y = Y[i:upper_bound]
        yield(ret_x, ret_y)