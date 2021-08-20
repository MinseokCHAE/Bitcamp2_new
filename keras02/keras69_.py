import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

input = Input((1,))
d = Dense(3)(input)
d = Dense(2)(d)
output = Dense(1)(d)
model = Model(inputs=input, outputs=output)

model.summary()

