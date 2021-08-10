import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#1. Data
x = [[0,0], [0,1], [1,0], [1,1]]
y = [0,1,1,0]

#2. Modeling
model = LinearSVC()

#3. Traning
model.fit(x,y)

#4. Evaluating
score = model.score(x,y)
print('score = ', score)
# score =  0.25, 0.5, 0.75 => Linear로는 불가

from sklearn.svm import SVC # 다층모델
model = SVC()
model.fit(x,y)
score = model.score(x,y)
print('score = ', score)
# score =  1.0

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# deep learning 'Linear' modeling
input = Input((2,))
output = Dense(1, activation='sigmoid')(input)
model = Model(inputs=input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=100, batch_size=1)
evaluate = model.evaluate(x,y)
print('acc = ', evaluate[1])
# acc =  0.75 

# deep learning '' modeling

