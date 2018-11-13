# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:57:19 2018

@author: test
"""

from sklearn.decomposition import PCA
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
f = open('./img2array.bin','rb')
dic = pickle.load(f)
arr = dic['data']
label=dic['label']
pca = PCA(n_components=0.95)
pca.fit(arr)
print ("explained_variance_ratio_",pca.explained_variance_ratio_)
print ("explained_variance_",pca.explained_variance_)
print ("pca.n_components_",pca.n_components_)

trainData=pca.transform(arr)

clf = RandomForestClassifier(n_jobs=111)
clf.fit(trainData,dic['label'] ) 