from xgboost import XGBRFRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

x, y = load_boston(return_X_y=True)
# print(x.shape, y.shape) # (506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

model = XGBRFRegressor(n_jobs=8)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
# print('score = ', score)    # score =  0.8871407181522256

from sklearn.feature_selection import SelectFromModel
threshold = np.sort(model.feature_importances_)

for thresh in threshold:
    # print(thresh)
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRFRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score1 = r2_score(y_test, y_pred)
    score2 = selection_model.score(select_x_test, y_test)

    print('Thresh=%.3f, n=%d, r2=%.2f%%' %(thresh, select_x_train.shape[1], 
    score1*100))




