# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:47:22 2018

@author: test
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

iris = load_iris()   # 这里是sklearn中自带的一部分数据
df = pd.DataFrame(iris.data, columns=iris.feature_names) # 格式化数据
trainData=pd.read_csv('D:\\code\\pycode\\lipstick\\train_data.csv')
trainDataLable=pd.read_csv('D:\\code\\pycode\\lipstick\\train_labels.csv')

trainData=trainData.join(trainDataLable)

print (df)          # 
trainData['is_train'] = np.random.uniform(0, 1, len(trainData)) <= .75
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

train, test = trainData[trainData['is_train']==True], trainData[trainData['is_train']==False]

features = trainData.columns[:263]
clf = RandomForestClassifier(n_jobs=2)
clf.fit(train[features], train['label'])  # 用train来训练样本

test_pred=clf.predict(test[features])   #用测试数据来做预测
print(classification_report(test['label'], test_pred))
