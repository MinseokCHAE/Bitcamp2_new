import numpy as np
from numpy import argmax
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
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

x_train = np.load('./_save/_npy/study/keras59_MW_x_train.npy')   
y_train = np.load('./_save/_npy/study/keras59_MW_y_train.npy')    
x_test = np.load('./_save/_npy/study/keras59_MW_x_val.npy')   
y_test = np.load('./_save/_npy/study/keras59_MW_y_val.npy')

x_train = x_train.reshape(2648, 150*150*3)
x_test = x_test.reshape(661, 150*150*3)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape, x_train_noised.shape, x_test_noised.shape)   # (2648, 67500) (661, 67500) (2648, 67500) (661, 67500)
# print(y_train.shape, y_test.shape)  # (2648, 2) (661, 2)

def autoencoder(hidden_layer_size):
    input = Input((67500,))
    xx = Dense(units=hidden_layer_size, activation='relu')(input)
    output = Dense(1, activation='sigmoid')(xx)
    model = Model(input, output)
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95%
model.compile(loss='mse', optimizer='adam')
model.fit(x_train_noised, y_train, epochs=8, batch_size=1024, validation_split=0.01)
# img_decoded = model.predict(x_test_noised)

#4. Evaluating, Prediction
loss = model.evaluate(x_test_noised, y_test)

print('loss = ', loss[0])
print('acc = ', loss[1])

'''
loss =  0.6232322454452515
acc =  0.6747352480888367
val_acc =  0.6301369667053223
'''
