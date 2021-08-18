import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28 *28))
x_test = x_test.reshape((10000, 28* 28))
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape((60000, 28, 28))
x_test = x_test.reshape((10000, 28, 28))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

def build_model(node1, node2, node3, rate, opt, lr):
    input = Input((28, 28))
    xx = Conv1D(node1, 2, activation='relu')(input)
    xx = MaxPooling1D()(xx)
    xx = Conv1D(node2, 2, activation='relu')(xx)
    xx = Dropout(rate)(xx)
    xx = Conv1D(node3, 2, activation='relu')(xx)
    xx = GlobalAveragePooling1D()(xx)
    output = Dense(10, activation='softmax')(xx)
    model = Model(inputs=input, outputs=output)

    model.compile(optimizer=opt(learning_rate=lr), metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_parameter():
    batch_size = [128, 512, 2048]
    rate = [0.1, 0.2, 0.4]
    optimizer = [Adam, Adadelta]
    lr = [0.01, 0.005, 0.001]
    node1 = [8, 16, 32]
    node2 = [8, 16, 32]
    node3 = [8, 16, 32]
    return {'batch_size' : batch_size, 'rate' : rate, 'opt' : optimizer, 'lr' : lr, 
                    'node1' : node1, 'node2' : node2, 'node3' : node3}

parameter = create_parameter()
model = KerasClassifier(build_fn=build_model, verbose=1)

random = RandomizedSearchCV(model, parameter, cv=2)
random.fit(x_train, y_train, epochs=2, validation_split=0.01, verbose=1)

best_par = random.best_params_
best_est = random.best_estimator_
best_score = random.best_score_
score = random.score(x_test, y_test)

print('best parameter = ', best_par)
print('best estimator = ', best_est)
print('best score = ', best_score)
print('score = ', score)

'''
best parameter =  {'rate': 0.1, 'opt': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'lr': 0.01, 'layer3': 8, 'layer2': 32, 'layer1': 32, 'batch_size': 128}
best estimator =  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000287A0933550>
best score =  0.8995833396911621
score =  0.9057999849319458
'''
