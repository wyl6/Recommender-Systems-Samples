# In[56]:

from __future__ import print_function
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from util import load_dataset, batcher

# In[57]: load train and test data
        
x_train, x_test, y_train, y_test = load_dataset()


# In[61]:define variables


n, p = x_train.shape
# number of latent factors
k = 10
# design features of users
X = tf.placeholder('float', shape=[None, p])
# target vector
Y = tf.placeholder('float', shape=[None, 1])

# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# matrix factorization factors, randomly initialized
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

# estimation of y, initialized to 0
Y_hat = tf.Variable(tf.zeros([n, 1]))


# In[67]:define loss and optimizer


# calculate output with FM equation
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True))
pair_interactions = tf.multiply(0.5, 
                                tf.reduce_sum(
                                    tf.subtract(
                                        tf.pow(tf.matmul(X,tf.transpose(V)), 2), 
                                        tf.matmul(tf.pow(X, 2), tf.pow(tf.transpose(V), 2))), 
                                    1, keepdims=True))
Y_hat = tf.add(linear_terms, pair_interactions)
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W,2)))+tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V,2)))
error = tf.reduce_mean(tf.square(tf.subtract(Y, Y_hat)))
loss =  tf.add(error, l2_norm)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)



    # In[77]: train process

epoches = 10
batch_size = 100

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for epoch in tqdm(range(epoches), unit='epoch'):
    perm = np.random.permutation(x_train.shape[0])
    for bx, by in batcher(x_train[perm], y_train[perm], batch_size):
        sess.run(optimizer, feed_dict={
            X:bx.reshape(-1, p),
            Y:by.reshape(-1, 1)
        })


# In[81]: test process


errors = []
for bx, by in batcher(x_test, y_test):
    errors.append(sess.run(error, feed_dict={
        X:bx.reshape(-1, p),
        Y:by.reshape(-1, 1)
    }))
RMSE = np.sqrt(np.array(errors).mean())
print(RMSE)

sess.close()


