import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling, training, evaluating
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('number of models = ', len(allAlgorithms))
for (name, model) in allAlgorithms:
    try:
        model = model()
        model.fit(x_train, y_train)
        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        score = model.score(x_test, y_test)
        # print(name, 'acc = ', acc)
        print(name, 'score = ', score)
    except:
        print(name, 'error')
'''
number of models =  54
ARDRegression score =  0.4116695011812993
AdaBoostRegressor score =  0.30929402364928815
BaggingRegressor score =  0.21010847791719323
BayesianRidge score =  0.4234092956487918
CCA score =  0.41023938531496085
DecisionTreeRegressor score =  -0.21956940869052977
DummyRegressor score =  -0.0007401508457360872
ElasticNet score =  0.10914454073828517
ElasticNetCV score =  0.4076635866455003
ExtraTreeRegressor score =  -0.24920787994237537
ExtraTreesRegressor score =  0.3585545502878905
GammaRegressor score =  0.06301818165783968
GaussianProcessRegressor score =  -14.65976364832664
GradientBoostingRegressor score =  0.2786318882900152
HistGradientBoostingRegressor score =  0.20673827909193998
HuberRegressor score =  0.43043185329856315
IsotonicRegression error
KNeighborsRegressor score =  0.325464359089926
KernelRidge score =  0.4141041692185645
Lars score =  0.4279535865623666
LarsCV score =  0.41124150277857985
Lasso score =  0.4061117996157274
LassoCV score =  0.41261750091744986
LassoLars score =  0.33765187564450716
LassoLarsCV score =  0.41124150277857985
LassoLarsIC score =  0.41212622412644273
LinearRegression score =  0.4279535865623664
LinearSVR score =  0.1361408387509523
MLPRegressor score =  -0.4603192611616549
MultiOutputRegressor error
MultiTaskElasticNet error
MultiTaskElasticNetCV error
MultiTaskLasso error
MultiTaskLassoCV error
NuSVR score =  0.117340649185673
OrthogonalMatchingPursuit score =  0.2666454404916777
OrthogonalMatchingPursuitCV score =  0.41094666145543046
PLSCanonical score =  -1.5678159123408917
PLSRegression score =  0.4284155087406303
PassiveAggressiveRegressor score =  0.2756429676292048
PoissonRegressor score =  0.406104113794147
RANSACRegressor score =  0.057203599817576056
RadiusNeighborsRegressor score =  0.13857392327205165
RandomForestRegressor score =  0.2981973568480448
RegressorChain error
Ridge score =  0.42057322943901065
RidgeCV score =  0.420573229439011
SGDRegressor score =  0.41129992022394024
SVR score =  0.11419380420917769
StackingRegressor error
TheilSenRegressor score =  0.4244300660180227
TransformedTargetRegressor score =  0.4279535865623664
TweedieRegressor score =  0.06734739970098069
VotingRegressor error
'''