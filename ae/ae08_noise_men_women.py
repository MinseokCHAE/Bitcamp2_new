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

'''
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
                                                        validation_split=0.2)
datagen_pred = ImageDataGenerator(rescale=1./255)

xy_train = datagen_train.flow_from_directory('../_data/men_women', 
                                                                target_size=(150, 150),
                                                                batch_size=3500,
                                                                class_mode='binary',
                                                                shuffle=True,
                                                                subset='training')
xy_val = datagen_train.flow_from_directory('../_data/men_women', 
                                                                target_size=(150, 150),
                                                                batch_size=3500,
                                                                class_mode='binary',
                                                                shuffle=True,
                                                                subset='validation')
x_pred = datagen_pred.flow_from_directory('../_data/men_women_pred', 
                                                                target_size=(150, 150),
                                                                batch_size=3500,
                                                                class_mode='binary')

# Found 2648 images belonging to 2 classes.
# Found 661 images belonging to 2 classes.
# Found 1 images belonging to 1 classes.

# np.save('./_save/_npy/MW_x_train.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/MW_y_train.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/MW_x_val.npy', arr=xy_val[0][0])
# np.save('./_save/_npy/MW_y_val.npy', arr=xy_val[0][1])
# np.save('./_save/_npy/MW_x_pred.npy', arr=x_pred[0][0])
'''
x_train = np.load('./_save/_npy/study/keras59_MW_x_train.npy')   
y_train = np.load('./_save/_npy/study/keras59_MW_y_train.npy')    
x_test = np.load('./_save/_npy/study/keras59_MW_x_val.npy')   
y_test = np.load('./_save/_npy/study/keras59_MW_y_val.npy')
x_pred = np.load('./_save/_npy/study/keras59_MW_x_pred.npy')
# print(x_train.shape, x_val.shape, x_pred.shape, y_train.shape, y_val.shape)
# (2648, 150, 150, 3) (661, 150, 150, 3) (1, 150, 150, 3) (2648,) (661,)

x_train = x_train.reshape(2648, 150*150*3)
x_test = x_test.reshape(661, 150*150*3)
x_pred = x_pred.reshape(1, 150*150*3)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_pred_noised = x_pred + np.random.normal(0, 0.1, size=x_pred.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
x_pred_noised = np.clip(x_pred_noised, a_min=0, a_max=1)

def autoencoder(hidden_layer_size):
    input = Input((150*150*3,))
    xx = Dense(units=hidden_layer_size, activation='relu')(input)
    output = Dense(150*150*3, activation='sigmoid')(xx)
    model = Model(input, output)
    return model

autoencoder = autoencoder(hidden_layer_size=154)  # pca 95%
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(x_train_noised, x_train, epochs=8, batch_size=512, validation_split=0.01)

x_train_decoded = autoencoder.predict(x_train_noised)
x_test_decoded = autoencoder.predict(x_test_noised)
x_pred_decoded = autoencoder.predict(x_pred_noised)

x_train_decoded = x_train_decoded.reshape(2648, 150, 150, 3)
x_test_decoded = x_test_decoded.reshape(661, 150, 150, 3)
x_pred_decoded = x_pred_decoded.reshape(1, 150, 150, 3)

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
hist = model.fit(x_train_decoded, y_train, epochs=16, batch_size=16, verbose=1, validation_split=0.02)

#4. Evaluating, Prediction
loss = model.evaluate(x_test_decoded, y_test)
val_acc = hist.history['val_acc']
prediction = model.predict(x_pred_decoded)
result = (1-prediction) * 100

print('loss = ', loss[0])
print('acc = ', loss[1])
print('val_acc = ', val_acc[-1])
print('남자일 확률 (%) = ', result)

'''
[before noised]
loss =  0.7983900904655457
acc =  0.6414523720741272
val_acc =  0.6792452931404114
남자일 확률 (%) =  [[85.25112]]

[after noised]
loss =  0.6818958520889282
acc =  0.5748865604400635
val_acc =  0.5660377144813538
남자일 확률 (%) =  [[42.695]]
'''
