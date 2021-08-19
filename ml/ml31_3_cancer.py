import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']
# print(y.shape)  # (569,)

count_value = pd.Series(y).value_counts()
# print(count_value)
# 1    357
# 0    212

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=21, stratify=y)

smote = SMOTE(random_state=21, k_neighbors=2)
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

model = XGBClassifier(n_jobs=-1)
model.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')
score = model.score(x_test, y_test)
# print('score = ', score)   # score =  0.9736842105263158

