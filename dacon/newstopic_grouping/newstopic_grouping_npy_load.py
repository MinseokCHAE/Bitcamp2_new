import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


submission = pd.read_csv('../_data/dacon/newstopic_grouping/sample_submission.csv', header=0)
x = np.load('./_save/_npy/dacon/newstopic_grouping/NTG_x_vector.npy')
y = np.load('./_save/_npy/dacon/newstopic_grouping/NTG_y_vector.npy')
x_pred = np.load('./_save/_npy/dacon/newstopic_grouping/NTG_x_pred_vector.npy')

# # x, x_pred scaling -> 효과X
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)
# print(x.shape, x_pred.shape) # (45654, 14) (9131, 14)

# y to categorical -> Stratified Kfold는 model.fit 에서 적용, LogisticRegressor는 생략
# print(np.unique(y)) # 0~6
# y = to_categorical(y)
# print(np.unique(y)) # 0, 1

# x, y train_test_split -> Stratified KFold는 생략
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

# #2. Modeling
# input = Input((13, ))
# d = Embedding(76529, 256)(input)
# d = Dropout(0.7)(d)
# d = LSTM(128, return_sequences=True, activation='relu')(d)
# d = Dropout(0.7)(d)
# d = Conv1D(64, 2, activation='relu')(d)
# # d = Dense(512, activation='relu')(input)
# # d = Dropout(0.5)(d)
# # d = Dense(256, activation='relu')(d)
# d = Dropout(0.7)(d)
# d = Flatten()(d)
# output = Dense(7, activation='softmax')(d)
# model = Model(inputs=input, outputs=output)
# model = RandomForestRegressor()
model = LogisticRegression(class_weight = 'balanced')


#3. Compiling, Training
# optimizer = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

# date = datetime.datetime.now()
# date_time = date.strftime('%m%d_%H%M')
# path = './_save/_mcp/dacon/newstopic_grouping/'
# info = '{val_acc:.4f}'
# filepath = ''.join([path, date_time, '_', info, '.hdf5'])
# cp = ModelCheckpoint(monitor='val_acc', save_best_only=True, mode='max', verbose=1, filepath=filepath)
# es = EarlyStopping(monitor='val_acc', restore_best_weights=False, mode='max', verbose=1, patience=8)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode='max', verbose=1, patience=2, factor=0.5)

# start_time = time.time()
# model.fit(x_train, y_train, epochs=16, batch_size=256, verbose=1, validation_split=0.1, callbacks=[es, cp, reduce_lr])
# #Stratified KFold 적용
# # new_tech = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
# # for i, (i_train, i_val) in enumerate(new_tech.split(x, y), 1):
# #     print(f'training model for CV #{i}')
# #     model.fit(x[i_train], to_categorical(y[i_train]), epochs=8, batch_size=512, verbose=1, validation_data=(x[i_val], to_categorical(y[i_val])), callbacks=[es, cp])
# end_time = time.time() - start_time
model.fit(x_train, y_train) # RandomForestRegressor, LogisticRegressor 적용


# best weight model loading
# model = load_model('./_save/_mcp/dacon/newstopic_grouping/0.7720.hdf5')

#4. Evaluating
# loss = model.evaluate(x_test, y_test)

# print('loss = ', loss[0])
# print('acc = ', loss[1])
# print('time taken(s) = ', end_time)
print("Accuracy: {}".format(model.score(x_test, y_test))) # LogisticRegressor 적용

'''
loss =  1.4483565092086792
acc =  0.7688095569610596
time taken(s) =  283.4025468826294
'''

#5. Prediction
# prediction = np.zeros((x_pred.shape[0], 7)) # predict 값 저장할 곳 생성
# # prediction += model.predict(x_pred) # / 5 Stratified KFold 검증횟수(n_splits)로 나누기
# topic_idx = []
# for i in range(len(prediction)):
#     topic_idx.append(np.argmax(prediction[i]))  # reverse to_categorical 적용후 리스트에 저장

# 제출파일형식 맞추기
# submission['topic_idx'] = topic_idx
# submission.to_csv('../_data/dacon/newstopic_grouping/NTG.csv', index=False)

prediction = model.predict(x_pred)
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('../_data/dacon/newstopic_grouping/NTG.csv', header=['index', 'topic_idx'], index=False)
