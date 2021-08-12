import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=7) # column 10 -> 7개로 압축
x = pca.fit_transform(x)
# print(x.shape) # (442, 7)

pca_evr = pca.explained_variance_ratio_
# print(pca_evr) 
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192  0.05365605]
# print(sum(pca_evr)) # 0.9479436357350414, n_components=10 일때 sum=1 (축소X)

cumsum = np.cumsum(pca_evr)
# print(cumsum) 
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759  0.94794364]
# 누적합 -> n_components 구분 기준 설정

n_component = np.argmax(cumsum >= 0.94) + 1
# print(n_component) # 7

plt.plot(cumsum)
plt.grid()
plt.show()

