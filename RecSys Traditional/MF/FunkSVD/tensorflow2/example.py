# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from util.metrics import mae
from util.metrics import rmse
from datasets import ml1m
from util.config import Config
from models.svd import SVD
from sklearn.model_selection import train_test_split

# Note that x is a 2D numpy array, 
# x[i, :] contains the user-item pair, and y[i] is the corresponding rating.
x, y = ml1m.load_data()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

config = Config()
config.num_users = np.max(x[:, 0]) + 1
config.num_items = np.max(x[:, 1]) + 1
config.min_value = np.min(y)
config.max_value = np.max(y)

with tf.Session() as sess:
    model = SVD(config, sess)
    model.train(x_train, y_train, val_data=(
        x_test, y_test), epoches=2, batch_size=1024)
        
    y_pred = model.predict(x_test)
    print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
        
    # Save model
    model = model.save_model('model/')
    
    # Load model
    # model = model.load_model('model/')