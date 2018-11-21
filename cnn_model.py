# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:31:42 2018

@author: test
"""

#CNN网络模型类            
class Model:
    def __init__(self):
        self.model = None 
        
    #建立模型
    def build_model(self, dataset, nb_classes = 2):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential() 
        
        #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', 
                                     input_shape = dataset.input_shape))    #1 2维卷积层
        self.model.add(Activation('relu'))                                  #2 激活函数层
        
        self.model.add(Convolution2D(32, 3, 3))                             #3 2维卷积层                             
        self.model.add(Activation('relu'))                                  #4 激活函数层
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #5 池化层
        self.model.add(Dropout(0.25))                                       #6 Dropout层

        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         #7  2维卷积层
        self.model.add(Activation('relu'))                                  #8  激活函数层
        
        self.model.add(Convolution2D(64, 3, 3))                             #9  2维卷积层
        self.model.add(Activation('relu'))                                  #10 激活函数层
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                      #11 池化层
        self.model.add(Dropout(0.25))                                       #12 Dropout层

        self.model.add(Flatten())                                           #13 Flatten层
        self.model.add(Dense(512))                                          #14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))                                  #15 激活函数层   
        self.model.add(Dropout(0.5))                                        #16 Dropout层
        self.model.add(Dense(nb_classes))                                   #17 Dense层
        self.model.add(Activation('softmax'))                               #18 分类层，输出最终结果
        
        #输出模型概况
        self.model.summary()