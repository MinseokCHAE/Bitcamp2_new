from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

datasets = load_boston()
x = datasets['data']
y = datasets['target']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = XGBRegressor(n_estimators=20, learning_rate=0.05, n_jobs=1)
model.fit(x_train, y_train, verbose=1, 
    eval_metric=['rmse', 'mae', 'logloss'],
    eval_set=[(x_train, y_train), (x_test, y_test)],
    early_stopping_rounds=10
)
score = model.score(x_test, y_test)
eval = model.evals_result()
# print('score = ', score)
# print('eval = ', eval)
# score =  0.8832469797675666

'''
import matplotlib.pyplot as plt
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()
'''
