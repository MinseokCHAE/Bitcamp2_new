import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
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
# score =  [0.         0.01123596 0.01136364 0.01136364 0.02272727]
# score_avg =  0.011338100102145046

# model = LinearSVC()
# score =  [0.         0.         0.01136364 0.01136364 0.01136364]
# score_avg =  0.006818181818181818

# model = KNeighborsRegressor()
# score =  [0.26641449 0.27583291 0.46434366 0.42364649 0.52268373]
# score_avg =  0.3905842563397755

# model = LinearRegression()
# score =  [0.42795359 0.45619816 0.53278477 0.48758196 0.54201609]
# score_avg =  0.4893069117795578

# model = DecisionTreeRegressor()
# score =  [-0.09139515 -0.04189804 -0.07291704 -0.06685284 -0.31308971]
# score_avg =  -0.11723055617487885

# model = RandomForestRegressor()
# score =  [0.28576225 0.45660995 0.45382272 0.47270233 0.42272566]
# score_avg =  0.41832458141993734

score = cross_val_score(model, x, y, cv=kfold)
print('score = ', score)
print('score_avg = ', sum(score)/n_splits)
