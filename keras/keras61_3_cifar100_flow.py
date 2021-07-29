import numpy as np
from numpy import argmax
import time
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. Data Preproccessing
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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

augment_size=10000 # 증폭 사이즈

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 32,32,3)
x_train = x_train.reshape(x_train.shape[0], 32,32,3)
x_test = x_test.reshape(x_test.shape[0], 32,32,3)

x_augmented = datagen_train.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False, #save_to_dir='d:/bitcamp2/_temp/')
).next()[0]

x_train = np.concatenate((x_train, x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
input = Input((32, 32, 3))
x = Conv2D(32, (1,1), activation='relu')(input)
x = MaxPooling2D((1,1))(x)
x = Conv2D(64, (1,1), activation='relu')(x)
x = MaxPooling2D((1,1))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
output = Dense(100, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=8, batch_size=128, verbose=1, validation_split=0.02)

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)
val_acc = hist.history['val_acc']

print('loss = ', loss[0])
print('acc = ', loss[1])
print('val_acc = ', val_acc[-1])

'''
loss =  4.603000164031982
acc =  0.011300000362098217
val_acc =  0.08833333104848862
'''

