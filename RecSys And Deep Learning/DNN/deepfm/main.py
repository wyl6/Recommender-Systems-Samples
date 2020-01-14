import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt
from config import IGNORE_COLS, NUMERIC_COLS

import config
from DeepFM import DeepFM


dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":10,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "random_seed":config.RANDOM_SEED
}


def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    df = pd.concat([dfTrain,dfTest])
    
    feature_dict = {}
    total_feature = 0
    
    for col in df.columns:
        if col in IGNORE_COLS:
            continue
        elif col in NUMERIC_COLS:
            feature_dict[col] = total_feature
            total_feature += 1
        else:
            unique_val = df[col].unique()
            feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
            total_feature += len(unique_val)
    dfm_params['feature_size'] = total_feature
    # 转化训练集
    train_y = dfTrain[['target']].values.tolist()
    dfTrain.drop(['target','id'],axis=1,inplace=True)
    train_feature_index = dfTrain.copy()
    train_feature_value = dfTrain.copy()
    
    for col in train_feature_index.columns:
        if col in IGNORE_COLS:
            train_feature_index.drop(col,axis=1,inplace=True)
            train_feature_value.drop(col,axis=1,inplace=True)
            continue
        elif col in NUMERIC_COLS:
            train_feature_index[col] = feature_dict[col]
        else:
            train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
            train_feature_value[col] = 1
    dfm_params['field_size'] = len(train_feature_index.columns)

    train_y = np.reshape(np.array(train_y), (-1,1))
    return train_feature_index, train_feature_value, train_y



if __name__ == '__main__':
    Xi_train, Xv_train, y_train = load_data()
    model = DeepFM(**dfm_params)
    model.fit(Xi_train, Xv_train, y_train)