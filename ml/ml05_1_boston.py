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
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21) 
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = SVC() 
# 안됨

# model = LinearSVC()
# 안됨

# model = KNeighborsRegressor()
# score =  [0.53153447 0.46279674 0.45686016 0.59738263 0.54965116]
# score_avg =  0.5196450309808538

# model = LinearRegression()
# score =  [0.71493642 0.70586907 0.6797936  0.77347282 0.68069476]
# score_avg =  0.7109533328842113

# model = DecisionTreeRegressor()
# score =  [0.82261656 0.66697441 0.74552615 0.75168817 0.6662955 ]
# score_avg =  0.7306201586443425

# model = RandomForestRegressor()
# score =  [0.89058548 0.8805761  0.81841422 0.88760869 0.84869411]
# score_avg =  0.8651757171018204

score = cross_val_score(model, x, y, cv=kfold)
print('score = ', score)
print('score_avg = ', sum(score)/n_splits)
