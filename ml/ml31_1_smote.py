from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target
# print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48

x = x[:-30]
y = y[:-30]
# print(pd.Series(y).value_counts())
#  1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=21, stratify=y)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
# print('score = ', score)   # 1.0

smote = SMOTE(random_state=21)
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

model_smote = XGBClassifier(n_jobs=-1)
model_smote.fit(x_smote_train, y_smote_train)
score_smote = model_smote.score(x_test, y_test)
# print('score = ', score_smote)   # 0.9667

