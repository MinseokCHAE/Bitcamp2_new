from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import numpy as np

x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) # (506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
'''
parameter = [
    {'gamma': [0,10,100,1000], 'max_depth': [2,4,6,8,10]},
]

model = XGBRegressor()
random = RandomizedSearchCV(model, parameter, verbose=1)
random.fit(x_train, y_train)

best_estimator = random.best_estimator_
best_score = random.best_score_
# print('best parameter = ', best_estimator)
# print('best score = ', best_score)
'''
from sklearn.feature_selection import SelectFromModel

model = XGBRegressor(
    base_score=0.5, booster='gbtree', colsample_bylevel=1,
    colsample_bynode=1, colsample_bytree=1, gamma=1000, gpu_id=-1,
    importance_type='gain', interaction_constraints='',       
    learning_rate=0.300000012, max_delta_step=0, max_depth=8, 
    min_child_weight=1, monotone_constraints='()',
    n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
    tree_method='exact', validate_parameters=1, verbosity=None
)
model.fit(x_train, y_train)

threshold = np.sort(model.feature_importances_)
for thresh in threshold:
    # print(thresh)
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(
        base_score=0.5, booster='gbtree', colsample_bylevel=1,
    colsample_bynode=1, colsample_bytree=1, gamma=1000, gpu_id=-1,
    importance_type='gain', interaction_constraints='',       
    learning_rate=0.300000012, max_delta_step=0, max_depth=8, 
    min_child_weight=1, monotone_constraints='()',
    n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
    tree_method='exact', validate_parameters=1, verbosity=None
    )
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score1 = r2_score(y_test, y_pred)
    score2 = selection_model.score(select_x_test, y_test)

    print('Thresh=%.3f, n=%d, r2=%.2f%%' %(thresh, select_x_train.shape[1], 
    score1*100))




