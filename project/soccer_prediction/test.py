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

#1-1. 데이터 추출 및 정보확인
#   1) read_sql
path = sqlite3.connect('../_data/soccer_prediction/database.sqlite')   # 데이터베이스 경로설정
player = pd.read_sql_query('SELECT * FROM Player', path) # SELECT 구문 이용 필요 데이터셋 추출 (Player, Player_Attributes)
player_attributes = pd.read_sql_query('SELECT * FROM Player_Attributes', path)

#   2) null 값 drop
player = player.dropna(axis=0) 
player_attributes = player_attributes.dropna(axis=0)

#   3) column 명 확인
'''for columns in player.columns: 
    print(columns)'''
# [id  player_api_id   player_name player_fifa_api_id  birthday    height  weight]
'''for columns in player_attributes.columns: 
    print(columns)'''
# [id player_fifa_api_id  player_api_id   date    overall_rating  potential   preferred_foot  attacking_work_rate
# defensive_work_rate crossing    finishing   heading_accuracy    short_passing   volleys dribbling   curve
# free_kick_accuracy  long_passing    ball_control    acceleration    sprint_speed    agility reactions   balance
# shot_power  jumping stamina strength    long_shots  aggression  interceptions   positioning vision  penalties
# marking standing_tackle sliding_tackle  gk_diving   gk_handling gk_kicking  gk_positioning  gk_reflexes]

#   4) 데이터 shape 확인 및 초기 5행 시범출력
'''print(player.shape, player.head()) # (11060, 7) '''
# id  player_api_id         player_name  ...             birthday  height  weight
# 0   1         505942  Aaron Appindangoye  ...  1992-02-29 00:00:00  182.88     187
# 1   2         155782     Aaron Cresswell  ...  1989-12-15 00:00:00  170.18     146
# 2   3         162549         Aaron Doran  ...  1991-05-13 00:00:00  170.18     163
# 3   4          30572       Aaron Galindo  ...  1982-05-08 00:00:00  182.88     198
# 4   5          23780        Aaron Hughes  ...  1979-11-08 00:00:00  182.88     154
# [11060 rows x 7 columns]    
'''print(player_attributes.shape, player_attributes.head()) # (183978, 42)'''
# id  player_fifa_api_id  player_api_id  ... gk_kicking  gk_positioning  gk
# 0   1              218353         505942  ...       10.0             8.0          8.0
# 1   2              218353         505942  ...       10.0             8.0          8.0
# 2   3              218353         505942  ...       10.0             8.0          8.0
# 3   4              218353         505942  ...        9.0             7.0          7.0
# 4   5              218353         505942  ...        9.0             7.0          7.0
# [183978 rows x 42 columns]

#1-2. column 정리

#   1) 불필요한 column drop ( id, player_name, player_fifa_api_id, date )
player = player.drop(['id', 'player_name', 'player_fifa_api_id'], axis=1)
player_attributes = player_attributes.drop(['id', 'player_fifa_api_id', 'date'], axis=1)

#   2) birthday column 으로부터 birth_year column 생성 (age) 
#   [참고] https://hiio.tistory.com/30
'''print(player.info()) '''
# #   Column         Non-Null Count  Dtype
#  1   birthday       non-null  11060 object
player['birthday'] = pd.to_datetime(player['birthday']) # apply() 적용을 위해 dtype 변환 object -> datetime64
'''print(player.info())'''
# #   Column         Non-Null Count  Dtype
#  1   birthday       non-null 11060  datetime64[ns]
player['birth_year'] = player['birthday'].apply(lambda x:x.year) # 출생년도만 추출
player = player.drop('birthday', axis=1) # 기존 birthday column drop
'''print(player['birth_year'].head(5)) # result check'''
# 0        1992
# 1        1989
# 2        1991
# 3        1982
# 4        1979

