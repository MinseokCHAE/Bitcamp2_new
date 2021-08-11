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
scaler = StandardScaler()
scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)
parameter = [
    {'C': [1,10,100,1000], 'kernel': ['linear']},
    {'C': [1,10,100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1,10,100,1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
]

model = SVC()
random = RandomizedSearchCV(model, parameter, cv=kfold, verbose=1)
random.fit(x_train, y_train)

best_estimator = random.best_estimator_
best_score = random.best_score_
y_pred = random.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)
random_score = random.score(x_test, y_test)

print('best parameter = ', best_estimator)
print('best score = ', best_score)
print('acc score = ', acc_score)
print('random score = ', random_score)

# best parameter =  SVC(C=1000, gamma=0.001, kernel='sigmoid')
# best score =  0.9666666666666668
# acc score =  0.9666666666666667
# random score =  0.9666666666666667
