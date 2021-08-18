
#fingerfrint name to use
ffpp = "pattern"

# read csv
import numpy as np
import pandas as pd
train = pd.read_csv("../_data/dacon/polymer_properties/train.csv")
dev = pd.read_csv("../_data/dacon/polymer_properties/dev.csv")
test = pd.read_csv("../_data/dacon/polymer_properties/test.csv")
ss = pd.read_csv("../_data/dacon/polymer_properties/sample_submission.csv")

import rdkit
print('rdkit version :', rdkit.__version__)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

# npy load
np_train_fps_array = np.load('./_save/_npy/dacon/polymer_properties/x_train.npy')
np_test_fps_array = np.load('./_save/_npy/dacon/polymer_properties/x_test.npy')
train_y = np.load('./_save/_npy/dacon/polymer_properties/y_train.npy')
test_y = np.load('./_save/_npy/dacon/polymer_properties/y_test.npy')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

X, Y = np_train_fps_array , train_y

#validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=21)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

model = create_deep_learning_model()
model.fit(X, Y, epochs = 256)
test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y
ss.to_csv("../_data/dacon/polymer_properties/rdkit.csv", index=False)

