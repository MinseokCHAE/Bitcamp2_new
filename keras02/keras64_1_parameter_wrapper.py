import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

def build_model(rate=0.5, optimizer='adam'):
    input = Input((28*28))
    xx = Dense(512, activation='relu')(input)
    xx = Dropout(rate)(xx)
    xx = Dense(256, activation='relu')(xx)
    xx = Dropout(rate)(xx)
    xx = Dense(128, activation='relu')(xx)
    xx = Dropout(rate)(xx)
    output = Dense(10, activation='softmax')(xx)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_parameter():
    batch_size = [8, 16, 32, 64, 128]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    rate = [0.1, 0.2, 0.4]
    return {'batch_size' : batch_size, 'optimizer' : optimizer, 'rate' : rate}

parameter = create_parameter()
model = KerasClassifier(build_fn=build_model, verbose=1)

random = RandomizedSearchCV(model, parameter, cv=5)
random.fit(x_train, y_train, epochs=8, verbose=1)

best_par = random.best_params_
best_est = random.best_estimator_
best_score = random.best_score_
score = random.score(x_test, y_test)

print('best parameter = ', best_par)
print('best estimator = ', best_est)
print('best score = ', best_score)
print('score = ', score)
'''
best parameter =  {'rate': 0.1, 'optimizer': 'adam', 'batch_size': 64} 
best estimator =  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000014EE72D0CA0>
best score =  0.9768499851226806
score =  0.9765999913215637
'''
