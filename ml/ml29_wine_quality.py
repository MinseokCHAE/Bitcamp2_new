import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier

datasets = pd.read_csv('../_data/study/winequality-white.csv',
                                index_col=None, header=0, sep=';')

# print(datasets.head())
# print(datasets.shape)   # (4898, 12)

count_data = datasets.groupby('quality')['quality'].count()
# print(count_data)
plt.bar(count_data.index, count_data)
# plt.show()
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''

datasets = datasets.values
x = datasets[:, :11]
y = datasets[:, 11]
# print(y.shape)  # (4898,)

newlist = []
for i in list(y):
    if i<=4 :
        newlist += [0]
    elif i<=7 :
        newlist += [1]
    else:
        newlist += [2]
y = np.array(newlist)
# print(y.shape)  # (4898,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
# print('score = ', score)     
# # score =  0.6653061224489796 
# -> score =  0.9551020408163265

