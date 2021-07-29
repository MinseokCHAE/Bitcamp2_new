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
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. Data Preprocessing
datagen_train = ImageDataGenerator(rescale=1./255, 
                                                        horizontal_flip=True, 
                                                        vertical_flip=True, 
                                                        width_shift_range=0.1, 
                                                        height_shift_range=0.1,
                                                        rotation_range=5, 
                                                        zoom_range=1.2, 
                                                        shear_range=0.7, 
                                                        fill_mode='nearest')
datagen_test = ImageDataGenerator(rescale=1./255)

xy_train = datagen_train.flow_from_directory('../_data/cat_dog/training_set/training_set', 
                                                                target_size=(150, 150),
                                                                batch_size=8005,
                                                                class_mode='binary',
                                                                shuffle=True)
xy_test = datagen_test.flow_from_directory('../_data/cat_dog/test_set/test_set', 
                                                                target_size=(150, 150),
                                                                batch_size=2023,
                                                                class_mode='binary',
                                                                shuffle=True)

# Found 8005 images belonging to 2 classes.
# Found 2023 images belonging to 2 classes.

# np.save('./_save/_npy/CD_x_train.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/CD_y_train.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/CD_x_test.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/CD_y_test.npy', arr=xy_test[0][1])

x_train = np.load('./_save/_npy/CD_x_train.npy')   
y_train = np.load('./_save/_npy/CD_y_train.npy')    
x_test = np.load('./_save/_npy/CD_x_test.npy')   
y_test = np.load('./_save/_npy/CD_y_test.npy')

#2. Modeling
input = Input((150, 150, 3))
x = Conv2D(32, (2,2), activation='relu')(input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (2,2), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (2,2), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (2,2), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (2,2), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=16, batch_size=16, verbose=1, validation_split=0.02)

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)
val_acc = hist.history['val_acc']

print('loss = ', loss[0])
print('acc = ', loss[1])
print('val_acc = ', val_acc[-1])

'''
loss =  0.8473806381225586
acc =  0.6258032917976379
val_acc =  0.6086956262588501
'''
