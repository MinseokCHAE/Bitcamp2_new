import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. data preprocessing
datasets = load_iris()
x = datasets.data
y = datasets.target
'''
print(x.shape) # (150, 4)
print(y.shape) # (150, )
print(y[:20]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(np.unique(y)) # [0 1 2]
'''
scaler = StandardScaler()
scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

#2. modeling
from sklearn.svm import LinearSVC
model = LinearSVC()

#3. compiling, training
model.fit(x_train, y_train)

#4. evaluating, prediction
from sklearn.metrics import accuracy_score
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('score = ', score)
print('acc = ', acc)

'''
score =  0.9666666666666667
acc =  0.9666666666666667
'''
