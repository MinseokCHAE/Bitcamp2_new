import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


#1. Data Preprocessing
# read_csv 
datasets_train = pd.read_csv('../_data/dacon/climatetech_grouping/train.csv', header=0)
datasets_test = pd.read_csv('../_data/dacon/climatetech_grouping/test.csv', header=0)

# null값 제거
datasets_train = datasets_train.dropna(axis=0)
datasets_test = datasets_test.dropna(axis=0)
# print(datasets_train.shape, datasets_test.shape)    # (171138, 13) (42793, 12)

# x, y, x_pred 분류 
x = datasets_train.iloc[:, -3:-2]
x_pred = datasets_test.iloc[:, -2:-1]
y = datasets_train.iloc[:, -1]
# print(x.head(), y.head(), x_pred.head())

# 불필요한 특수문자 및 기호 삭제
def text_cleaning(input):
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", input)
    return text
x['요약문_한글키워드'] = x['요약문_한글키워드'].apply(lambda y : text_cleaning(y))
# x['요약문_영문키워드'] = x['요약문_영문키워드'].apply(lambda y : text_cleaning(y))
x_pred['요약문_한글키워드'] = x_pred['요약문_한글키워드'].apply(lambda y : text_cleaning(y))
# x_pred['요약문_영문키워드'] = x_pred['요약문_영문키워드'].apply(lambda y : text_cleaning(y))
# print(x.head(), x_pred.head())

# x, x_pred 토큰화 및 sequence화
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)
# vector = TfidfVectorizer(min_df=0.0, analyzer='char', sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
# count = CountVectorizer(analyzer='word', max_features=5000)
# vector.fit(x)
# x = vector.transform(x)
# x_pred = vector.transform(x_pred)
# x = x.toarray()
# x_pred = x_pred.toarray()
print(x.shape, x_pred.shape)
'''
# x, x_pred padding
max_len1 = max(len(i) for i in x)
avg_len1 = sum(map(len, x)) / len(x)
max_len2 = max(len(i) for i in x_pred)
avg_len2 = sum(map(len, x_pred)) / len(x_pred)
# print(max_len1, max_len2) # 2 2
# print(avg_len1, avg_len2) # 2 2
x = pad_sequences(x, padding='pre', maxlen=4)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=4)
print(x.shape, x_pred.shape)
print(np.unique(x), np.unique(x_pred)) # 0~76528

# 전처리 데이터 npy저장
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_x_vector.npy', arr=x)
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_y_vector.npy', arr=y)
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_x_pred_vector.npy', arr=x_pred)
'''