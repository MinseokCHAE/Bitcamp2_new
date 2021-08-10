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
'''
print(x.shape) # (150, 4)
print(y.shape) # (150, )
print(y[:20]) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print(np.unique(y)) # [0 1 2]
'''
scaler = StandardScaler()
scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

#2. modeling
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이름만 봐선 회귀모델 같지만 분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = SVC() # model.score =  0.9
# model = KNeighborsClassifier() # model.score =  0.9666666666666667
# model = LogisticRegression() # model.score =  0.9333333333333333
# model = DecisionTreeClassifier() # model.score =  0.9333333333333333
# model = RandomForestClassifier() # model.score =  0.9333333333333333

#3. compiling, training
model.fit(x_train, y_train)

#4. evaluating, prediction
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
score = model.score(x_test, y_test)

print('acc_score = ', acc)
print('model.score = ', score)
'''
acc_score =  0.9666666666666667
model.score =  0.9666666666666667
'''
