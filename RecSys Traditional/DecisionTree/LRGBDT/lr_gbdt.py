
# coding: utf-8

# In[50]:


from sklearn.datasets import load_iris
import numpy as np
import lightgbm as lgb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[51]:


## build data
iris = pd.DataFrame(load_iris().data)
iris.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
iris['Species'] = load_iris().target%2


# In[52]:


## train test split
train=iris[0:130]
test=iris[130:]
X_train=train.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
X_test=test.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
y_train=train[[train.Species.name]]
y_test=test[[test.Species.name]]


# In[105]:


## build lgb model
lgb_train = lgb.Dataset(X_train.as_matrix(), 
                        y_train.values.reshape(y_train.shape[0],))
lgb_eval = lgb.Dataset(X_test.as_matrix(), 
                       y_test.values.reshape(y_test.shape[0],), 
                       reference=lgb_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 16,
    'num_trees':100,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params=params,
                train_set=lgb_train,
                num_boost_round=3000,
                valid_sets=None)



# In[110]:


# build train matrix
num_leaf = 16

y_pred = gbm.predict(X_train,raw_score=False,pred_leaf=True)

transformed_training_matrix = np.zeros([len(y_pred),
                                        len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)

for i in range(0,len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i]);
    transformed_training_matrix[i][temp] += 1


# In[111]:


print(y_pred[0], y_pred.shape)


# In[68]:


# build test matrix
y_pred = gbm.predict(X_test,pred_leaf=True)
transformed_testing_matrix = np.zeros([len(y_pred),
                                       len(y_pred[0]) * num_leaf],
                                      dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
	transformed_testing_matrix[i][temp] += 1


# In[69]:


# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

label_train = y_train.values.reshape(y_train.shape[0],)
label_test = y_test.values.reshape(y_test.shape[0],)

c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
for t in range(0,len(c)):
    lm = LogisticRegression(penalty='l2',C=c[t]) # logestic model construction
    lm.fit(transformed_training_matrix,y_train.values.reshape(y_train.shape[0],))  # fitting the data
    y_pred_est = lm.predict(transformed_testing_matrix)   # Give the probabilty on each label
    acc =accuracy_score(label_test, y_pred_est)
    print('Acc of test', acc)


