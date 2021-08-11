import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)
parameter = [
    {'n_estimators': [100,200]},
    {'max_depth': [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 7, 10]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]

model = RandomForestRegressor()
grid = GridSearchCV(model, parameter, cv=kfold)
grid.fit(x_train, y_train)

best_estimator = grid.best_estimator_
best_score = grid.best_score_
# y_pred = grid.predict(x_test)
# acc_score = accuracy_score(y_test, y_pred)
grid_score = grid.score(x_test, y_test)

print('best parameter = ', best_estimator)
print('best score = ', best_score)
# print('acc score = ', acc_score)
print('grid score = ', grid_score)

# best parameter =  RandomForestRegressor(min_samples_split=5)
# best score =  0.830591307770115
# grid score =  0.8783616408326427

