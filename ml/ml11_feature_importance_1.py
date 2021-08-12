from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# datasets = load_iris()
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=21)

# model = DecisionTreeClassifier()
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
fi = model.feature_importances_
# print('acc = ', acc) 
# print('feature importance = ', fi) 
'''
acc =  0.9333333333333333
feature importance =  [0.         0.01877738 0.93038628 0.05083633]

acc =  0.8167302364614968
feature importance =  [4.65337120e-02 1.79528192e-03 1.22284369e-03 1.06296573e-03
 4.46883841e-02 5.76213490e-01 9.16744492e-03 7.56311344e-02
 4.89577237e-04 1.51935555e-02 1.30359731e-02 2.18025495e-02
 1.93163088e-01]
 '''

# model = RandomForestClassifier()
model = RandomForestRegressor()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
fi = model.feature_importances_
# print('acc = ', acc) 
# print('feature importance = ', fi) 
'''
acc =  0.9333333333333333
feature importance =   [0.10247514 0.02934314 0.49668215 0.37149957]

acc =  0.8966366963060146
feature importance =  [0.04427398 0.00098628 0.00759437 0.0014595  0.01810554 0.32585296
 0.0153736  0.04736119 0.00295252 0.01678977 0.01899875 0.01028392
 0.48996762]
'''

# model = XGBClassifier()
model = XGBRegressor() # boston 안됨
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
fi = model.feature_importances_
# print('acc = ', acc) 
# print('feature importance = ', fi) 
'''
acc =  0.9333333333333333
feature importance =  [0.00592343 0.02190936 0.89838314 0.07378403]
'''

# model = GradientBoostingClassifier()
model = GradientBoostingRegressor()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
fi = model.feature_importances_
# print('acc = ', acc) 
# print('feature importance = ', fi) 
'''
acc =  0.9333333333333333
feature importance =  [0.00089294 0.01479436 0.60549753 0.37881517]

acc =  0.9117851258014603
feature importance =  [0.03226745 0.00184143 0.00438353 0.00126158 0.0298447  0.36397503
 0.00693764 0.06977711 0.00182096 0.01733549 0.02541375 0.00726426
 0.43787707]
'''
