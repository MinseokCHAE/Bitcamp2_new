import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape((50000, 32 * 32 * 3))
x_test = x_test.reshape((10000, 32 * 32 * 3))
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape((50000, 32, 32, 3))
x_test = x_test.reshape((10000, 32, 32, 3))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input = Input((32,32,3))
xx = EfficientNetB0(weights='imagenet', include_top=False)(input)
xx = GlobalAveragePooling2D()(xx)
output = Dense(10, activation='softmax')(xx)
model = Model(inputs=input, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=16, batch_size=512, validation_split=0.01)

loss = model.evaluate(x_test, y_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])
# loss =  4.201725482940674
# accuracy =  0.38370001316070557


