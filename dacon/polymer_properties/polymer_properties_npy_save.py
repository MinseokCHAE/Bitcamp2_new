
#fingerfrint name to use
ffpp = "pattern"

# read csv
import numpy as np
import pandas as pd
train = pd.read_csv("../_data/dacon/polymer_properties/train.csv")
dev = pd.read_csv("../_data/dacon/polymer_properties/dev.csv")
test = pd.read_csv("../_data/dacon/polymer_properties/test.csv")
ss = pd.read_csv("../_data/dacon/polymer_properties/sample_submission.csv")

train = pd.concat([train,dev])
train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

import rdkit
print('rdkit version :', rdkit.__version__)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

### train ###
import math
train_fps = [] #train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)
train_y_array = np.array(train_y)
# print(np_train_fps_array.shape) # (30345, 2048)
# print(train_y_array.shape) # (30345,)

### test ###
test_fps = [] #test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)
test_y_array = np.array(test_y)

# print(np_test_fps_array.shape)  # (602, 2048)
# print(test_y_array.shape)  # (0,)

np.save('./_save/_npy/dacon/polymer_properties/x_train.npy', arr=np_train_fps_array)
np.save('./_save/_npy/dacon/polymer_properties/x_test.npy', arr=np_test_fps_array)
np.save('./_save/_npy/dacon/polymer_properties/y_train.npy', arr=train_y_array)
np.save('./_save/_npy/dacon/polymer_properties/y_test.npy', arr=test_y_array)

