import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) # (60000,) (10000,)
x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 28 * 28 )
# print(x.shape) # (70000, 784)
y = np.append(y_train, y_test, axis=0)

pca = PCA(n_components=441) 
x = pca.fit_transform(x)
# print(x.shape) # (70000, 441)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = x.reshape(70000, 21, 21)
# print(x.shape) # (70000, 21, 21)
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

input = Input((21, 21))
l = Conv1D(64, (2,), activation='relu')(input)
l = Conv1D(128, (2,), activation='relu')(l)
l = MaxPooling1D()(l)
l = Conv1D(128, (2,), activation='relu')(l)
l = Conv1D(256, (2,), activation='relu')(l)
l= Dropout(0.2)(l)
l = Conv1D(128, (2,), activation='relu')(l)
l = Conv1D(32, (2,), activation='relu')(l)
l = GlobalAveragePooling1D()(l)
output = Dense(10, activation='softmax')(l)
model = Model(inputs=input, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=4, batch_size=128, validation_split=0.001)
loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
loss =  0.16357393562793732
accuracy =  0.9552381038665771
'''
