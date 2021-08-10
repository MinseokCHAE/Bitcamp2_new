import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, x_test = x_train.reshape(50000, 32*32*3) / 255, x_test.reshape(10000, 32*32*3) / 255
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = SVC() # score =  
# model = KNeighborsClassifier() # score =  0.1511
# model = LogisticRegression() # score =  0.1547
# model = DecisionTreeClassifier() # score =  0.0872
# model = RandomForestClassifier() # score =  

#3. compiling, training
model.fit(x_train, y_train)

#4. evaluating, prediction
score = model.score(x_test, y_test)
print('score = ', score)

