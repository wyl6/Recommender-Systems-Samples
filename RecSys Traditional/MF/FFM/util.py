# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
# In[1]:
class FieldHandler(object):
    def __init__(self, train_file_path, test_file_path=None, category_columns=[], continuation_columns=[]):
        self.train_file_path = None
        self.test_file_path = None
        self.feature_nums = 0
        self.field_dict = {}
        self.category_columns = category_columns
        self.continuation_columns = continuation_columns
        
        if(not isinstance(train_file_path, str)):
            raise ValueError('train file path must be str')
        if(os.path.exists(train_file_path)):
            self.train_file_path = train_file_path
        else:
            raise OSError('train file path isn\'t exist')
        if(test_file_path):
            if(os.path.exists(test_file_path)):
                self.test_file_path = test_file_path
            else:
                raise OSError('test file path isn\'t exist')
        
        self.read_data()
        self.df[category_columns].fillna('-1', inplace=True)
        self.build_field_dict()
        self.build_standard_scaler()
        self.field_nums = len(self.category_columns+self.continuation_columns)
        
    def build_field_dict(self):
        for col in self.df.columns:
            if(col in self.category_columns):
                cv = self.df[col].unique()
                self.field_dict[col] = dict(zip(cv, range(self.feature_nums, self.feature_nums+len(cv))))
                self.feature_nums += len(cv)
            else:
                self.field_dict[col] = self.feature_nums
                self.feature_nums += 1
                
    def read_data(self):
        if(self.train_file_path and self.test_file_path):
            train_df = pd.read_csv(self.train_file_path)[self.category_columns+self.continuation_columns]
            test_df = pd.read_csv(self.test_file_path)[self.category_columns+self.continuation_columns]
            self.df = pd.concat([train_df, test_df]) # default axis=0
        else:
            self.df = pd.read_csv(self.train_file_path)[self.category_columns+self.continuation_columns]
            
    def build_standard_scaler(self):
        if(self.continuation_columns):
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(self.df[self.continuation_columns].values)
        else:
            self.standard_scaler = None
            
def transformation_data(file_path:str, field_hander:FieldHandler, label=None):
    df_v = pd.read_csv(file_path)
    if(label):
        if(label in df_v.columns):
            labels = df_v[[label]].values.astype('float32')
        else:
            raise KeyError(f'label "{label} isn\'t exist')
            
    df_v = df_v[field_hander.category_columns+field_hander.continuation_columns]
    df_v[field_hander.category_columns].fillna('-1', inplace=True)
    df_v[field_hander.continuation_columns].fillna(-999, inplace=True)
    if(field_hander.standard_scaler):
        df_v[field_hander.continuation_columns] = field_hander.standard_scaler.transform(df_v[field_hander.continuation_columns])
    df_i = df_v.copy()
    
    for col in df_v.columns:
        if(col in field_hander.category_columns):
            df_i[col] = df_i[col].map(field_hander.field_dict[col])
            df_v[col] = 1
        else:
            df_i[col] = field_hander.field_dict[col]
    
    df_v = df_v.values.astype("float32")
    df_i = df_i.values.astype("int32")
    
    features = {
            'df_i':df_i,
            'df_v':df_v}
    
    if(label): 
        return features, labels
    return features, None

# In[2]:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

def dataGenerate(path="./Dataset/train.csv"):
    df = pd.read_csv(path)
    df = df[['Pclass',"Sex","SibSp","Parch","Fare","Embarked","Survived"]]
    class_columns = ['Pclass',"Sex","SibSp","Parch","Embarked"]
    continuous_columns = ['Fare']
    train_x = df.drop('Survived', axis=1)
    train_y = df['Survived'].values
    train_x = train_x.fillna("-1")
    le = LabelEncoder()
    oht = OneHotEncoder()
    files_dict = {}
    s = 0
    for index, column in enumerate(class_columns):
        try:
            train_x[column] =  le.fit_transform(train_x[column])
        except:
            pass
        ont_x = oht.fit_transform(train_x[column].values.reshape(-1,1)).toarray()
        for i in range(ont_x.shape[1]):
            files_dict[s] = index
            s +=1
        if index == 0:
            x_t = ont_x
        else:
            x_t = np.hstack((x_t, ont_x))
    x_t = np.hstack((x_t, train_x[continuous_columns].values.reshape(-1,1)))
    files_dict[s] = index + 1

    return x_t.astype("float32"), train_y.reshape(-1,1).astype("float32"), files_dict
