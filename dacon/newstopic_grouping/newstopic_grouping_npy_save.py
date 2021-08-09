import time
import datetime
import re
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#1. Data Preprocessing
# read_csv 
datasets_train = pd.read_csv('../_data/dacon/newstopic_grouping/train_data.csv', header=0)
datasets_test = pd.read_csv('../_data/dacon/newstopic_grouping/test_data.csv', header=0)

# null값 제거
datasets_train = datasets_train.dropna(axis=0)
datasets_test = datasets_test.dropna(axis=0)
# print(datasets_train.shape, datasets_test.shape)    # (45654, 3) (9131, 2)

# 불필요한 특수문자 및 기호 삭제
def text_cleaning(input):
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", input)
    return text
datasets_train['title'] = datasets_train['title'].apply(lambda x : text_cleaning(x))
datasets_test['title'] = datasets_test['title'].apply(lambda x : text_cleaning(x))

# x, y, x_pred 분류 
# x = datasets_train.iloc[:, -2]
# x_pred = datasets_test.iloc[:, -1]
# y = datasets_train.iloc[:, -1]
# print(x.head(), y.head(), x_pred.head())

# word2vec 사용
x_list = list(datasets_train['title'])
x_pred_list = list(datasets_test['title'])
y = datasets_train.iloc[:, -1]
x = []
x_pred = []
for x_list in x_list:
    x.append(x_list.split())
for x_pred_list in x_pred_list:
    x_pred.append(x_pred_list.split())

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
model_x = word2vec.Word2Vec(x, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
model_x_pred = word2vec.Word2Vec(x_pred, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    num_words = 0
    # 어휘사전 준비
    index2word_set = set(model.wv.index2word)
    for w in words:
        if w in index2word_set:
            num_words +=1
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            feature_vector = np.add(feature_vector, model[w])
    # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

def get_dataset(reviews, model, num_features):
    dataset = list()
    for s in reviews:
        dataset.append(get_features(s, model, num_features))
    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs

x = get_dataset(x, model_x, num_features)
x_pred = get_dataset(x_pred, model_x_pred, num_features)

print(x.shape, x_pred.shape)

'''
# x, x_pred 토큰화 및 sequence화
# token = Tokenizer()
# token.fit_on_texts(x)
# x = token.texts_to_sequences(x)
# x_pred = token.texts_to_sequences(x_pred)
vector = TfidfVectorizer(min_df=0.0, analyzer='char', sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
vector.fit(x)
x = vector.transform(x)
x_pred = vector.transform(x_pred)
x = x.toarray()
x_pred = x_pred.toarray()
# print(x, x_pred)

# x, x_pred padding
# max_len1 = max(len(i) for i in x)
# avg_len1 = sum(map(len, x)) / len(x)
# max_len2 = max(len(i) for i in x_pred)
# avg_len2 = sum(map(len, x_pred)) / len(x_pred)
# print(max_len1, max_len2) # 13 12
# print(avg_len1, avg_len2) # 6.883098961755816 5.913152995290767
# x = pad_sequences(x, padding='pre', maxlen=13)
# x_pred = pad_sequences(x_pred, padding='pre', maxlen=13)
print(x.shape, x_pred.shape) # (45654, 13) (9131, 13) //  (45654, 5000) (9131, 5000)
# print(np.unique(x), np.unique(x_pred)) # 0~76528
'''
# 전처리 데이터 npy저장
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_x_vector.npy', arr=x)
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_y_vector.npy', arr=y)
# np.save('./_save/_npy/dacon/newstopic_grouping/NTG_x_pred_vector.npy', arr=x_pred)
