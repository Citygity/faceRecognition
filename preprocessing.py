# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:59:00 2018

@author: test
"""
import numpy as np
import pickle
from PIL import Image
import os
database='D:\\code\\pycode\\faceRecognition\\database\\'
img_rows, img_cols = 640,480
face_data = np.empty((777, img_rows*img_cols))
arr=[[]]
label=[]
for dir in os.listdir(database):
    picFolderPath=os.path.join(database,dir)
    for picname in os.listdir(picFolderPath):
        if(picname.endswith('JPG')):
            img = Image.open(os.path.join(picFolderPath,picname))
            print("dir {} {} ".format(dir,picname))
            img = img.resize((224, 224))
            R,G,B=img.split()
            r_array = np.array(R).reshape([224*224])/255
            g_array = np.array(G).reshape([224*224])/255
            b_array = np.array(B).reshape([224*224])/255
            merge_array = np.concatenate((r_array,g_array,b_array))
            if arr == [[]]:
                arr = [merge_array]
                continue
            arr = np.concatenate((arr, [merge_array]),axis=0)
            label.append(int(dir))

dic = {'data': arr,'label':label}
f = open('./img2array.bin','wb')
pickle.dump(dic,f)          