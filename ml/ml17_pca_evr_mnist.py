import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape) # (60000,) (10000,)
x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 28 * 28 )
y = np.append(y_train, y_test, axis=0)
# print(x.shape) # (70000, 784)

pca = PCA(n_components=154) 
x = pca.fit_transform(x)
# print(x.shape) # (70000, 154)

pca_evr = pca.explained_variance_ratio_
print(sum(pca_evr)) # 0.9499943055215889

cumsum = np.cumsum(pca_evr)
n_component = np.argmax(cumsum >= 0.95) + 1
# print(n_component) # 154

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (49000, 154) (21000, 154) (49000,) (21000,)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
# print('score = ', score) # score =  0.8456927962656833
