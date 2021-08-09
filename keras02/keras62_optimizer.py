import numpy as np

#1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. Modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input = Input((1,))
d = Dense(1000)(input)
d = Dense(100)(d)
d = Dense(100)(d)
d = Dense(100)(d)
output = Dense(1)(d)
model = Model(inputs=input, outputs=output)

#3. Compiling, Training
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = Adam() # default learning_rate = 0.001
'''
optimizer = Adagrad() # default learning_rate = 0.001
optimizer = Adamax() # default learning_rate = 0.001
optimizer = Adadelta() # default learning_rate = 0.001
optimizer = RMSprop() # default learning_rate = 0.001
optimizer = SGD() # default learning_rate = 0.01
optimizer = Nadam() # default learning_rate = 0.001
'''
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluating, Prediction
loss, mse = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])

print('loss = ', loss)
print('y_pred = ', y_pred)

'''
loss =  0.0001566986902616918
y_pred =  [[10.976078]]
'''
