import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=7) # column 7개로 압축
x = pca.fit_transform(x)
# print(x.shape) # (442, 7)

model = XGBRegressor()
model.fit(x, y)
score = model.score(x, y)
# print('score = ', score) 
# score =  0.999990274544785
# score =  0.9999349120798557
