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

#   2) null 값 dropping
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

#1-2. merging ( player + player_attributes )

#   player dataset: 11060명의 선수 기본 정보 
#   player_attributes dataset: 각 선수 세부 스탯 시즌별로(date) 나열
#   동일 player_api_id 데이터(= 동일 선수 시즌별 데이터)를 평균값으로 단일화 시킨후 player dataset 에 추가(단일 dataset으로 통일)
#   평균값 계산을 위해 일부 데이터 수치화 필요 ( preferred_foot, attacking_work_rate, defensive_work_rate )

#   1) binary encoding ( preferred_foot )
player_attributes['preferred_foot'] = player_attributes['preferred_foot'].replace({'left':0, 'right':1}) # 주발
'''print(player_attributes['preferred_foot'].head())'''
# 0    1
# 1    1
# 2    1
# 3    1
# 4    1

#   2) get_dummies ( attacking_work_rate, defensive_work_rate )
#   [참고1] https://stackoverflow.com/questions/58101126/using-scikit-learn-onehotencoder-with-a-pandas-dataframe
#   [참고2] https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.get_dummies.html
#   'Series' object has no attribute 'to_categorical'  : pandas_series는 to_categorical 또는 onehotencode 적용불가
player_attributes['attacking_work_rate'] = pd.get_dummies(
    player_attributes['attacking_work_rate'], 
    prefix=['attacking_work_rate'], 
    columns=['attacking_work_rate'], drop_first=True)
'''print(player_attributes['attacking_work_rate'].head(10))'''
# 0    0
# 1    0
# 2    0
# 3    0
# 4    0
# 5    1
# 6    1
# 7    1
# 8    1
# 9    1

# onehotencode쓰는방법추가

#   3) 합치는거


#1-3. column 정리

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
'''print(player['birth_year'].head(5))'''
# 0        1992
# 1        1989
# 2        1991
# 3        1982
# 4        1979
player = player.drop('birthday', axis=1) # 기존 birthday column drop

#1-4. 
