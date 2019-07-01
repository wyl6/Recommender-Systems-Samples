# -*- coding: utf-8 -*-

## configure parameters
from os import path
import numpy as np
import matplotlib.pyplot as plt
import sys
import util
import recommender

parent_path = path.relpath('../../Datasets')
sys.path.insert(0, parent_path)
path = parent_path+'/MovieLens/ml-1m/ratings.dat'
df = util.load_dataframe(path)
model = recommender.SVDmodel(df, 'user', 'item', 'rating')

regularizer_constant = 0.5
learning_rate = 0.001
momentum_factor = 0.9
batch_size = 100
num_steps = 9000


## model traning
test_range = range(1,2)
all_dimensions = list(test_range)
errors = []
times = []

for dimension in test_range:
    print('\ndimension = {}'.format(dimension))
    model.training(dimension,
                   regularizer_constant,
                   learning_rate,
                   momentum_factor,
                   batch_size,
                   num_steps,
                   verbose=False)
    users, items, rates = model.test_batches.get_batch()
    pre_rates = model.prediction(users, items, show_valid=False)
    error = util.rmse(y_true=items, y_pred=pre_rates)
    errors.append(error)
    times.append(model.duration)


### results visualization
#plt.plot(all_dimensions, errors)
#plt.xlabel('Vector size')
#plt.ylabel('Test error')
#plt.show()
#
#plt.plot(all_dimensions, times)
#plt.xlabel('Vector size')
#plt.ylabel('Train duration')
#plt.show()
#
#best_result = min(list(zip(errors, all_dimensions, times)))
#result_string = 'In an experimen with sizes from {0} to {1}, \
#the best size for vector representation is {2} with error {3}, \
#and using the size will take {4} seconds.'.format(all_dimensions[0],
#all_dimensions[-1],best_result[0],best_result[1],best_result[2])
#print(result_string)