import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score


datasets = pd.read_csv('../_data/study/winequality-white.csv',
                                index_col=None, header=0, sep=';')

# print(datasets.head())
# print(datasets.shape)   # (4898, 12)

count_data = datasets.groupby('quality')['quality'].count()
# print(count_data)
plt.bar(count_data.index, count_data)
# plt.show()
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

datasets = datasets.values
x = datasets[:, :11]
y = datasets[:, 11]
# print(y.shape)  # (4898,)
count_value = pd.Series(y).value_counts()
# print(count_value)
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

for index, value in enumerate(y):
    if value == 3:
        y[index] = 5
    elif value == 4:
        y[index] = 5
    elif value == 7:
        y[index] = 6
    elif value == 9:
        y[index] = 8

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=21, stratify=y)

smote = SMOTE(random_state=21, k_neighbors=3)
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

model = XGBClassifier(n_jobs=-1)
model.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')
score = model.score(x_test, y_test)

y_pred = model.predict(x_test)
f1_score = f1_score(y_test, y_pred, average='macro')
# print('score = ', score)   # score =  0.7969387755102041
print('f1 score = ', f1_score)   # f1 score =  0.6793478632461666

