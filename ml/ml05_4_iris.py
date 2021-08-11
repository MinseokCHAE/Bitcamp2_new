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
scaler = StandardScaler()
scaler.fit_transform(x)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)


from sklearn.model_selection import KFold, cross_val_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC() 
# score =  [0.9        1.         0.96666667 1.         1.        ]
# score_avg =  0.9733333333333334

# model = LinearSVC()
# score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.9       ]      
# score_avg =  0.9466666666666667

# model = KNeighborsClassifier()
# score =  [0.96666667 1.         0.96666667 1.         1.        ]
# score_avg =  0.9866666666666667

# model = LogisticRegression()
# score =  [0.93333333 0.96666667 0.96666667 1.         0.93333333]
# score_avg =  0.96

# model = DecisionTreeClassifier()
# score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.93333333]
# score_avg =  0.9533333333333334

# model = RandomForestClassifier()
# score =  [0.93333333 0.96666667 0.96666667 1.         0.93333333]
# score_avg =  0.96

score = cross_val_score(model, x, y, cv=kfold)
print('score = ', score)
print('score_avg = ', sum(score)/n_splits)
