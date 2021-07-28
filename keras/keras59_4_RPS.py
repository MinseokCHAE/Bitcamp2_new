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
                                                        fill_mode='nearest',
                                                        validation_split=0.1)
datagen_val = ImageDataGenerator(rescale=1./255,
                                                                validation_split=0.1)

xy_train = datagen_train.flow_from_directory('../_data/rps', 
                                                                target_size=(150, 150),
                                                                batch_size=32,
                                                                class_mode='categorical',
                                                                subset='training')
xy_val = datagen_val.flow_from_directory('../_data/rps',
                                                                target_size=(150, 150),
                                                                batch_size=32,
                                                                class_mode='categorical',
                                                                subset='validation')

# Found 2520 images belonging to 3 classes.
# print(xy_data[0][0].shape, xy_data[0][1].shape) # (2520, 150, 150, 3) (2520, 3)

# np.save('./_save/_npy/rps_x.npy', arr=xy_data[0][0])
# np.save('./_save/_npy/rps_y.npy', arr=xy_data[0][1])

# x = np.load('./_save/_npy/rps_x.npy')    # (2520, 150, 150, 3)
# y = np.load('./_save/_npy/rps_y.npy')    # (2520, 3)

# print(np.unique(y)) 
# y = to_categorical(y)
# print(y.shape)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

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
output = Dense(3, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, epochs=16, verbose=1, validation_data=xy_val)

#4. Evaluating, Prediction
# loss = model.evaluate(x_test, y_test)

# print('loss = ', loss[0])
# print('acc = ', loss[1])

acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc =', acc[-1])
print('val_acc = ', val_acc[-1])

'''
acc = 0.7614638209342957
val_acc =  0.9007936716079712
'''
