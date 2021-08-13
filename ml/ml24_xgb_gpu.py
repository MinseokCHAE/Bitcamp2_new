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

model = XGBRegressor(n_estimators=1000, learning_rate=0.01, 
    n_jobs=1,
    tree_method='gpu_hist'
)

model.fit(x_train, y_train, verbose=1, 
    eval_metric=['rmse'],
    eval_set=[(x_train, y_train), (x_test, y_test)],
)

