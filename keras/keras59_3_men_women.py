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
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten
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

xy_data = datagen_train.flow_from_directory('../_data/men_women', 
                                                                target_size=(150, 150),
                                                                batch_size=3500,
                                                                class_mode='binary')
pred_data = datagen_test.flow_from_directory('../_data/men_women_pred', 
                                                                target_size=(150, 150),
                                                                batch_size=3500,
                                                                class_mode='binary')

# Found 3309 images belonging to 2 classes.
# Found 1 images belonging to 1 classes.
# print(xy_data[0][0].shape, xy_data[0][1].shape) # (3309, 150, 150, 3) (3309,)
# print(pred_data[0][0].shape)    # (1, 150, 150, 3)

# np.save('./_save/_npy/MW_x.npy', arr=xy_data[0][0])
# np.save('./_save/_npy/MW_y.npy', arr=xy_data[0][1])

x = np.load('./_save/_npy/MW_x.npy')    # (3309, 150, 150, 3)
y = np.load('./_save/_npy/MW_y.npy')    # (3309,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

#2. Modeling
input = Input((150, 150, 3))
c = Conv2D(32, (2,2))(input)
f = Flatten()(c)
d = Dropout(0.7)(f)
output = Dense(1, activation='sigmoid')(d)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=8, batch_size=128, verbose=1, validation_split=0.01)

#4. Evaluating, Prediction
loss = model.evaluate(x_test, y_test)
x_pred = pred_data[0][0]
prediction = model.predict(x_pred)
result = (1-prediction) * 100

print('loss = ', loss[0])
print('acc = ', loss[1])
print('남자일 확률 (%) = ', result)

'''
loss =  0.7722522020339966
acc =  0.555891215801239
남자일 확률 (%) =  [[52.427853]]
'''
