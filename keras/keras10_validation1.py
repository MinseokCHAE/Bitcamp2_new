from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7]) # 훈련
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10]) # 평가
y_test = np.array([8,9,10])
x_val = np.array([11,12,13]) #검증(훈련시)
y_val = np.array([11,12,13])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1234, batch_size=1, 
                    validation_data=(x_val, y_val))

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('loss = ', loss)
print('result = ', result)
'''
epochs = 1234
loss =  5.153803165486304e-12
result =  [[10.999999]]
'''
