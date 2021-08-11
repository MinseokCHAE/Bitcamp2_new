import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
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
# score =  [0.88596491 0.88596491 0.93859649 0.90350877 0.9380531 ]
# score_avg =  0.9104176370128861

# model = LinearSVC()
# score =  [0.90350877 0.9122807  0.9122807  0.93859649 0.91150442]
# score_avg =  0.9156342182890856

# model = KNeighborsRegressor()
# score =  [0.69911795 0.74365079 0.70290678 0.80958242 0.84228997]
# score_avg =  0.7595095830686741

# model = LinearRegression()
# score =  [0.78223897 0.7053208  0.71370461 0.68059011 0.76797577]
# score_avg =  0.7299660503771175

# model = DecisionTreeRegressor()
# score =  [0.72717949 0.6984127  0.77146676 0.82103611 0.7703252 ]
# score_avg =  0.7576840502716117

# model = RandomForestRegressor()
# score =  [0.85641456 0.79834365 0.7987422  0.84988509 0.91515047]
# score_avg =  0.8437071947912533

score = cross_val_score(model, x, y, cv=kfold)
print('score = ', score)
print('score_avg = ', sum(score)/n_splits)
