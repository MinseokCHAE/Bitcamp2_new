from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

datasets = load_iris()
x = datasets['data']
y = datasets['target']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=1)
model.fit(x_train, y_train, verbose=1, 
    eval_metric='mlogloss',
    eval_set=[(x_train, y_train), (x_test, y_test)]
)
score = model.score(x_test, y_test)
print('score = ', score)
# score =  0.9333333333333333
