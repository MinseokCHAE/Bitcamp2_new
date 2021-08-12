import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm
from konlpy.tag import Okt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


train=pd.read_csv('../_data/dacon/climatetech_grouping/train.csv')
test=pd.read_csv('../_data/dacon/climatetech_grouping/test.csv')
sample_submission=pd.read_csv('../_data/dacon/climatetech_grouping/sample_submission.csv')

train=train[['요약문_한글키워드','label']]
test=test[['요약문_한글키워드']]

def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
    word_text=okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review

stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
okt=Okt()
clean_train_text=[]
clean_test_text=[]

for text in tqdm.tqdm(train['요약문_한글키워드']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['요약문_한글키워드']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

print(len(clean_train_text), len(clean_test_text))

from sklearn.feature_extraction.text import CountVectorizer
