import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#1. data preprocessing
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)
# parameter = [
#     {'svc__C': [1,10,100,1000], 'svc__kernel': ['linear']},
#     {'svc__C': [1,10,100], 'svc__kernel': ['rbf'], 'svc__gamma': [0.001, 0.0001]},
#     {'svc__C': [1,10,100,1000], 'svc__kernel': ['sigmoid'], 'svc__gamma': [0.001, 0.0001]}
# ]
parameter = [
    {'model__C': [1,10,100,1000], 'model__kernel': ['linear']},
    {'model__C': [1,10,100], 'model__kernel': ['rbf'], 'model__gamma': [0.001, 0.0001]},
    {'model__C': [1,10,100,1000], 'model__kernel': ['sigmoid'], 'model__gamma': [0.001, 0.0001]}
]
scaler = StandardScaler()
model = SVC()
# pipe = make_pipeline(scaler, model)
pipe = Pipeline([('scaler', scaler), ('model', model)])
grid = GridSearchCV(pipe, parameter, cv=kfold, verbose=1)
grid.fit(x_train, y_train)

best_estimator = grid.best_estimator_
best_score = grid.best_score_
y_pred = grid.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
grid_score = grid.score(x_test, y_test)

print('best parameter = ', best_estimator)
print('best score = ', best_score)
print('acc score = ', acc_score)
print('grid score = ', grid_score)

'''
best parameter =  Pipeline(steps=[('scaler', StandardScaler()),
                ('model', SVC(C=100, gamma=0.001, kernel='sigmoid'))]) 
best score =  0.9833333333333334
acc score =  0.9
grid score =  0.9
'''
