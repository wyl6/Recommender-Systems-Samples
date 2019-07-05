# -*- coding: utf-8 -*-

from util.data_utils import get_zip_file
import pandas as pd

def load_data():
    
    URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    FILE_PATH = 'ml-1m/ratings.dat'
    
    file = get_zip_file(URL, FILE_PATH)
    df = pd.read_csv(file, sep="::", header=None, engine='python')
    
    return df.iloc[:,:2].values, df.iloc[:,2].values