from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

model = RandomForestClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
fi = model.feature_importances_
# print('score = ', score)
# print('feature importance = ', fi) 
# score =  0.9333333333333333
# feature importance =  [0.11325239 0.02089316 0.51868277 0.34717168]

def plot_feature_importance(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('feature importance')
    plt.ylabel('feature')
    plt.ylim(-1, n_features)

plot_feature_importance(model)
# plt.show() # sepal width column 제거

