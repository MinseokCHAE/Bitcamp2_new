import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


submission = pd.read_csv('../_data/dacon/climatetech_grouping/sample_submission.csv', header=0)
x = np.load('./_save/_npy/dacon/climatetech_grouping/CTG_x_vector.npy')
y = np.load('./_save/_npy/dacon/climatetech_grouping/CTG_y_vector.npy')
x_pred = np.load('./_save/_npy/dacon/climatetech_grouping/CTG_x_pred_vector.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = XGBClassifier(n_estimators=200, tree_method='gpu_hist')
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric='mlogloss' , early_stopping_rounds=20)
score = model.score(x_test, y_test)

print('score = ', score)
# score =  0.9151487335417802

prediction = model.predict(x_pred)
index = np.array([range(174304, 217880)])
index = np.transpose(index)
index = index.reshape(43576, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('../_data/dacon/climatetech_grouping/CTG.csv', header=['index', 'label'], index=False)
