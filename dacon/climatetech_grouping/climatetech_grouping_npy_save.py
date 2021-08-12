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
datasets_train = datasets_train.fillna('가나다')
datasets_test = datasets_test.fillna('가나다')
# print(datasets_train.shape, datasets_test.shape)    # (174304, 13) (43576, 12)

# x, y, x_pred 분류 
x = datasets_train['과제명'] + datasets_train['요약문_한글키워드']
x_pred = datasets_test['과제명'] + datasets_test['요약문_한글키워드']
y = datasets_train.iloc[:, -1]
# print(x.shape, y.shape, x_pred.shape) # (174304,) (174304,) (43576,)
# print(x[:5], x_pred[:5])

# 불필요한 특수문자 및 기호 삭제
# def text_cleaning(input):
#     text = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", input)
#     return text
# x = text_cleaning(x)
# x_pred = text_cleaning(x_pred)
# print(x.shape, x_pred.shape)

# tokenizer, vectorization
# token = Tokenizer()
# token.fit_on_texts(x)
# x = token.texts_to_sequences(x)
# x_pred = token.texts_to_sequences(x_pred)
# vector = TfidfVectorizer(min_df=0.0, analyzer='char', sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
count = CountVectorizer(tokenizer = lambda a: a, lowercase=False)
count.fit(x)
x = count.transform(x)
x_pred = count.transform(x_pred)
x = x.toarray()
x_pred = x_pred.toarray()
# print(x[:5], x_pred[:5])
# print(x.shape, x_pred.shape) # (174304, 2047) (43576, 2047)

# x, x_pred padding
# max_len1 = max(len(i) for i in x)
# avg_len1 = sum(map(len, x)) / len(x)
# max_len2 = max(len(i) for i in x_pred)
# avg_len2 = sum(map(len, x_pred)) / len(x_pred)
# # print(max_len1, max_len2) # 59 61
# # print(avg_len1, avg_len2) # 15.186507481182302 14.416444832017625
# x = pad_sequences(x, padding='pre', maxlen=60)
# x_pred = pad_sequences(x_pred, padding='pre', maxlen=60)
# print(x.shape, x_pred.shape) # (174304, 60) (43576, 60)
# print(np.unique(x), np.unique(x_pred)) # 0~316992

# 전처리 데이터 npy저장
np.save('./_save/_npy/dacon/climatetech_grouping/CTG_x_count.npy', arr=x)
np.save('./_save/_npy/dacon/climatetech_grouping/CTG_y_count.npy', arr=y)
np.save('./_save/_npy/dacon/climatetech_grouping/CTG_x_pred_count.npy', arr=x_pred)
