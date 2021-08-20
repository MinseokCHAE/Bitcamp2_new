import numpy as np
from numpy import argmax
import time
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. Data Preprocessing
datagen_train = ImageDataGenerator(rescale=1./255, 
                                                        horizontal_flip=True, 
                                                        vertical_flip=False, 
                                                        width_shift_range=0.1, 
                                                        height_shift_range=0.1,
                                                        rotation_range=5, 
                                                        zoom_range=0.2, 
                                                        shear_range=0.5, 
                                                        fill_mode='nearest')
datagen_test = ImageDataGenerator(rescale=1./255)

x_train = np.load('./_save/_npy/study/keras59_CD_x_train.npy')   
y_train = np.load('./_save/_npy/study/keras59_CD_y_train.npy')    
x_test = np.load('./_save/_npy/study/keras59_CD_x_test.npy')   
y_test = np.load('./_save/_npy/study/keras59_CD_y_test.npy')

augment_size=1000 # 증폭 사이즈

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150,150,3)
x_train = x_train.reshape(x_train.shape[0], 150,150,3)
x_test = x_test.reshape(x_test.shape[0], 150,150,3)

x_augmented = datagen_train.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False, #save_to_dir='d:/bitcamp2/_temp/')
).next()[0]

x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. Modeling
from tensorflow.keras.applications import MobileNetV3Small

input = Input((150,150,3))
xx = MobileNetV3Small(weights='imagenet', include_top=False)(input)
xx = GlobalAveragePooling2D()(xx)
output = Dense(1, activation='sigmoid')(xx)
model = Model(inputs=input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=16, batch_size=512, validation_split=0.01)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
deep learning
loss =  0.6057469248771667
acc =  0.6712802648544312
val_acc =  0.5027624368667603

MobileNetV3Small
loss =  0.8230592012405396
accuracy =  0.5002471804618835
'''
