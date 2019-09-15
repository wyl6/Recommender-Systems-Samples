# !/usr/bin/env python
# -*- coding:utf-8 -*-

import gc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DataParse:
    def __init__(self, category_feature, continuous_feature, ignore_feature=[], feature_dict={}, feature_size=0,
                 field_size=0):
        self.feature_dict = feature_dict
        self.feature_size = feature_size
        self.field_size = field_size
        self.ignore_feature = ignore_feature
        self.category_feature = category_feature
        self.continuous_feature = continuous_feature

    def FeatureDictionary(self, train, test):
        '''
        目的是给每一个特征维度都进行编号。
        1. 对于离散特征，one-hot之后每一列都是一个新的特征维度(计算编号时，不算0)。所以，原来的一维度对应的是很多维度，编号也是不同的。
        2. 对于连续特征，原来的一维特征依旧是一维特征。
        返回一个feat_dict，用于根据原特征名称和特征取值 快速查询出 对应的特征编号。
        train: 原始训练集
        test:  原始测试集
        continuous_feature: 所有数值型特征
        ignore_feature: 所有忽略的特征. 除了数值型和忽略的，剩下的全部认为是离散型
        feat_dict, feat_size
             1. feat_size: one-hot之后总的特征维度。
             2. feat_dict是一个{}， key是特征string的col_name, value可能是编号（int），可能也是一个字典。
             如果原特征是连续特征： value就是int，表示对应的特征编号；
             如果原特征是离散特征：value就是dict，里面是根据离散特征的 实际取值 查询 该维度的特征编号。 因为离散特征one-hot之后，一个取值就是一个维度，
             而一个维度就对应一个编号。
        '''
        df = pd.concat([train, test], axis=0)
        feat_dict = {}
        total_cnt = 0

        for col in df.columns:
            # 连续特征只有一个编号
            if col in self.continuous_feature:
                feat_dict[col] = total_cnt
                total_cnt = total_cnt + 1

            # 离散特征，有多少个取值就有多少个编号
            elif col in self.category_feature:
                unique_vals = df[col].unique()
                unique_cnt = df[col].nunique()
                feat_dict[col] = dict(zip(unique_vals, range(total_cnt, total_cnt + unique_cnt)))
                total_cnt = total_cnt + unique_cnt

        self.feature_size = total_cnt
        self.feature_dict = feat_dict
        print('feat_dict=', feat_dict)
        print('=' * 20)
        print('feature_size=', total_cnt)

    def parse(self, df):
        '''
        获得list形式的特征下标和特征值
        dfi的每一行代表一个样本各特征对应的编号，0,1,2,3,...
        dfv的每一行代表一个样本各特征的取值，离散特征功能取值为1，连续特征取值不变
        '''
        dfi = df.copy()
        dfv = df.copy()
        for col in dfi.columns:
            if col in self.ignore_feature:
                dfi.drop([col], axis=1, inplace=True)
                dfv.drop([col], axis=1, inplace=True)

            elif col in self.continuous_feature:  # 连续特征1个维度，对应1个编号，这个编号是一个定值
                dfi[col] = self.feature_dict[col]

            elif col in self.category_feature:  # 离散特征。不同取值对应不同的特征维度，编号也是不同的。
                dfi[col] = dfi[col].map(self.feature_dict[col])
                dfv[col] = 1.0
        feature_index = dfi.values.tolist()
        feature_val = dfv.values.tolist()
        self.field_size = len(feature_index[0])
        del dfi, dfv
        gc.collect()

        return feature_index, feature_val

if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                        'native_country']
    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    feature_index, feature_val = dataParse.parse(train)
    print(feature_index[0], feature_val[0])
