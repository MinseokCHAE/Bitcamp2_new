import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

model = XGBClassifier()
model.fit(x_train, y_train)

plot_importance(model) # 평가기준 = F score
plt.show()
