import pandas as pd
import numpy as np
import lightgbm
from tqdm import tqdm
import warnings

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings(action='ignore')

def preprocessing(temp_df, pum, len_lag) :
    # p_lag, q_lag 추가
    for lag in range(1,len_lag+1) :
      temp_df[f'p_lag_{lag}'] = -1
      temp_df[f'q_lag_{lag}'] = -1
      for index in range(lag, len(temp_df)) :
        temp_df.loc[index, f'p_lag_{lag}'] = temp_df[f'{pum}_가격(원/kg)'][index-lag] #1일전, 2일전, ... 가격을 feature로 추가
        temp_df.loc[index, f'q_lag_{lag}'] = temp_df[f'{pum}_거래량(kg)'][index-lag] #1일전, 2일전, ... 거래량을 feature로 추가

    # month 추가
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df['month'] = temp_df['date'].dt.month

    # 예측 대상(1w,2w,4w) 추가
    for week in ['1_week','2_week','4_week'] :
      temp_df[week] = 0
      n_week = int(week[0])
      for index in range(len(temp_df)) :
        try : temp_df[week][index] = temp_df[f'{pum}_가격(원/kg)'][index+7*n_week]
        except : continue

    # 불필요한 column 제거        
    temp_df = temp_df.drop(['date',f'{pum}_거래량(kg)',f'{pum}_가격(원/kg)'], axis=1)
    
    return temp_df

train = pd.read_csv('../_data/dacon/farm_price/public_data/train.csv')

# # preprocessing 함수 예시
# pum = '배추'
# temp_df = train[['date',f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']]
# preprocessing(temp_df, pum, len_lag=28)

def nmae(week_answer, week_submission):
    answer = week_answer
    target_idx = np.where(answer!=0)
    true = answer[target_idx]
    pred = week_submission[target_idx]
    score = np.mean(np.abs(true-pred)/true)
    
    return score


def at_nmae(pred, dataset):
    y_true = dataset.get_label()
    week_1_answer = y_true[0::3]
    week_2_answer = y_true[1::3]
    week_4_answer = y_true[2::3]
    
    week_1_submission = pred[0::3]
    week_2_submission = pred[1::3]
    week_4_submission = pred[2::3]
    
    score1 = nmae(week_1_answer, week_1_submission)
    score2 = nmae(week_2_answer, week_2_submission)
    score4 = nmae(week_4_answer, week_4_submission)
    
    score = (score1+score2+score4)/3
    
    return 'score', score, False

'''
0.001 5 128 0.8 0.8 5 21 7 = 0.23015
0.01 5 128 0.8 0.8 7 21 9 = 0.22934
0.01 5 128 0.7 0.7 7 21 7 = 0.22652
0.05 7 128 0.7 0.7 7 21 7 = 0.22831
0.021 3 128 0.7 0.7 7 21 7 = 0.22995
0.01 3 128 0.7 0.7 7 21 7 = 0.22536
0.01 2 128 0.7 0.7 7 21 7 = 0.22979
0.01 3 128 0.5 0.5 5 21 5 = 0.23142
0.01 3 128 0.6 0.6 6 21 7 = 0.22743
0.01 3 128 0.8 0.8 8 21 7 = 0.22731
0.01 2 128 0.8 0.8 8 21 7 = 0.23532
0.01 4 128 0.8 0.8 8 21 7 = 0.22919
0.01 3 128 0.7 0.7 8 21 7 = 0.22565
0.01 3 128 0.7 0.7 9 21 7 = 0.22618
0.01 3 128 0.7 0.7 6 21 7 = 0.22654

lr, depth, threads 고정
fraction, frequency 조절
'''

def model_train(x_train, y_train, x_valid, y_valid) :
    params = {'learning_rate': 0.01, 
              'max_depth': 3, 
              'boosting': 'gbdt', 
              'objective': 'regression',  
              'is_training_metric': True, 
              'num_leaves': 128, 
              'feature_fraction': 0.7, 
              'bagging_fraction': 0.7, 
              'bagging_freq': 6, 
              'seed':21,
              'num_threads': 7
             }

    model = lightgbm.train(params, 
                   train_set = lightgbm.Dataset(data = x_train, label = y_train),
                   num_boost_round = 10000, 
                   valid_sets = lightgbm.Dataset(data = x_valid, label = y_valid), 
                   init_model = None, 
                   early_stopping_rounds = 100,
                   feval = at_nmae,
                   verbose_eval = False,
                    )
    
    return model

unique_pum = [
    '배추', '무', '양파', '건고추','마늘',
    '대파', '얼갈이배추', '양배추', '깻잎',
    '시금치', '미나리', '당근',
    '파프리카', '새송이', '팽이버섯', '토마토',
]

unique_kind = [
    '청상추', '백다다기', '애호박', '캠벨얼리', '샤인마스캇'
]

model_dict = {}
split = 28 #validation

for pum in tqdm(unique_pum + unique_kind):
    # 품목 품종별 전처리
    temp_df = train[['date',f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']]
    temp_df = preprocessing(temp_df, pum, len_lag=28)
    
    # 주차별(1,2,4w) 학습
    for week_num in [1,2,4] :
        x = temp_df[temp_df[f'{week_num}_week']>0].iloc[:,:-3]
        y = temp_df[temp_df[f'{week_num}_week']>0][f'{week_num}_week']
        
        #train, test split
        x_train = x[:-split]
        y_train = y[:-split]
        x_valid = x[-split:]
        y_valid = y[-split:]
        
        model_dict[f'{pum}_model_{week_num}'] = model_train(x_train, y_train, x_valid, y_valid)

submission = pd.read_csv('../_data/dacon/farm_price/sample_submission.csv')
public_date_list = submission[submission['예측대상일자'].str.contains('2020')]['예측대상일자'].str.split('+').str[0].unique()
# ['2020-09-29', ...]

for date in tqdm(public_date_list) :
    test = pd.read_csv(f'../_data/dacon/farm_price/public_data/test_files/test_{date}.csv')
    for pum in unique_pum + unique_kind:
        # 예측기준일에 대해 전처리
        temp_test = pd.DataFrame([{'date' : date}]) #예측기준일
        alldata = pd.concat([train, test, temp_test], sort=False).reset_index(drop=True)
        alldata = alldata[['date', f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']].fillna(0)
        alldata = alldata.iloc[-28:].reset_index(drop=True)
        alldata = preprocessing(alldata, pum, len_lag=28)
        temp_test = alldata.iloc[-1].astype(float)[:-3]
        
        # 개별 모델을 활용하여 1,2,4주 후 가격 예측
        for week_num in [1,2,4] :
            temp_model = model_dict[f'{pum}_model_{week_num}']
            result = temp_model.predict(temp_test)
            condition = (submission['예측대상일자']==f'{date}+{week_num}week')
            idx = submission[condition].index
            submission.loc[idx, f'{pum}_가격(원/kg)'] = result[0]

submission.to_csv('../_data/dacon/farm_price/111.csv',index=False)

