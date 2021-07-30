import time
import datetime
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. Data Preprocessing
# read_sql
path = sqlite3.connect('../_data/soccer_prediction/database.sqlite')   # 데이터파일 경로설정
player = pd.read_sql_query('SELECT * FROM Player', path)
player_attributes = pd.read_sql_query('SELECT * FROM Player_Attributes', path)


