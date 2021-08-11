import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_wine()
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = SVC() 
# score =  [0.61111111 0.52777778 0.75       0.65714286 0.77142857]
# score_avg =  0.6634920634920635

# model = LinearSVC()
# score =  [0.94444444 0.61111111 0.33333333 0.94285714 0.88571429]      
# score_avg =  0.7434920634920634

# model = KNeighborsClassifier()
# score =  [0.66666667 0.63888889 0.77777778 0.68571429 0.65714286]
# score_avg =  0.6852380952380952

# model = LogisticRegression()
# score =  [1.         0.91666667 0.91666667 0.94285714 0.94285714]
# score_avg =  0.9438095238095239

# model = DecisionTreeClassifier()
# score =  [0.83333333 0.80555556 0.88888889 0.91428571 0.91428571]
# score_avg =  0.8712698412698412

# model = RandomForestClassifier()
# score =  [1.         1.         0.94444444 0.94285714 0.97142857]
# score_avg =  0.9717460317460318

score = cross_val_score(model, x, y, cv=kfold)
print('score = ', score)
print('score_avg = ', sum(score)/n_splits)
