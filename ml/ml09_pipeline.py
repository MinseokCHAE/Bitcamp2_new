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

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline

scaler = StandardScaler()
model = SVC()
pipe = make_pipeline(scaler, model)
pipe.fit(x_train, y_train)

score = pipe.score(x_test, y_test)
print('score = ', score)
# score =  0.9333333333333333
