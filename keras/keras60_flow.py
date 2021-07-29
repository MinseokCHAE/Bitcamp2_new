import numpy as np
from numpy import argmax
import time
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. Data Preproccessing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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
'''
flow_from_directory() :
  파일에서 load
  x, y가 튜플형태로 묶임

flow() :
  데이터에서 load
  x, y가 나뉨 

x_data = datagen_train.flow(
    np.tile(x_train[0].reshape(28*28), 100).reshape(-1, 28, 28, 1), # x_train[0] 에 해당하는 이미지 
    np.zeros(100),  # y: 임의로 0을 100번채움
    batch_size=100,
    shuffle=True).next()
    # augment_size = 100

.next() 적용 전
print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
print(type(x_data[0]))  # <class 'tuple'>
print(type(x_data[0][0]))   # <class 'numpy.ndarray'>
print(x_data[0][0].shape)   # (100, 28, 28, 1) x값
print(x_data[0][1].shape)    # (100, ) y값
  
.next 적용 후 -> 한 개씩 밀림

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
'''

augment_size=40000 # 증폭 사이즈

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(x_train.shape[0]) # 60000
# print(randidx, randidx.shape)  # [40935 15948 31644 ... 10904  3018 50310]  (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28,28,1)
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

x_augmented = datagen_train.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False).next()[0]
# print(x_augmented.shape)    # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(9):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()
'''
#2. Modeling
input = Input((28, 28, 1))
x = Conv2D(32, (1,1), activation='relu')(input)
x = MaxPooling2D((1,1))(x)
x = Conv2D(64, (1,1), activation='relu')(x)
x = MaxPooling2D((1,1))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=16, batch_size=16, verbose=1, validation_split=0.02)

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)
val_acc = hist.history['val_acc']

print('loss = ', loss[0])
print('acc = ', loss[1])
print('val_acc = ', val_acc[-1])

'''
loss =  0.3549745976924896
acc =  0.8712000250816345
val_acc =  0.7789999842643738
'''

