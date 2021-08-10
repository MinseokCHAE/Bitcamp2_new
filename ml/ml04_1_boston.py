import numpy as np
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
boston = load_boston()
x = boston.data
y = boston.target
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
ARDRegression score =  0.714221932620104
AdaBoostRegressor score =  0.7914930216945788
BaggingRegressor score =  0.8599670317103763
BayesianRidge score =  0.7127858580055613
CCA score =  0.6939976152679337
DecisionTreeRegressor score =  0.6955710366871195
DummyRegressor score =  -0.0007717096547503743
ElasticNet score =  0.1583818752675482
ElasticNetCV score =  0.7091902587026315
ExtraTreeRegressor score =  0.7656255266585601
ExtraTreesRegressor score =  0.8866528313164188
GammaRegressor score =  0.18958413252962247
GaussianProcessRegressor score =  -3.8676338657588785
GradientBoostingRegressor score =  0.9112991466064837
HistGradientBoostingRegressor score =  0.8581364427387908
HuberRegressor score =  0.7230951003208502
IsotonicRegression error
KNeighborsRegressor score =  0.7259504857913879
KernelRidge score =  0.6582151957294136
Lars score =  0.7139731116682013
LarsCV score =  0.7144801712120168
Lasso score =  0.2198783627670533
LassoCV score =  0.714047842604663
LassoLars score =  -0.0007717096547503743
LassoLarsCV score =  0.7151200626345506
LassoLarsIC score =  0.715160678276884
LinearRegression score =  0.7149364161392221
LinearSVR score =  0.6119154203052605
MLPRegressor score =  0.2392334931764627
MultiOutputRegressor error
MultiTaskElasticNet error
MultiTaskElasticNetCV error
MultiTaskLasso error
MultiTaskLassoCV error
NuSVR score =  0.586283435122708
OrthogonalMatchingPursuit score =  0.5254511161568489
OrthogonalMatchingPursuitCV score =  0.6820245796272115
PLSCanonical score =  -2.1349617255297355
PLSRegression score =  0.7000162981466617
PassiveAggressiveRegressor score =  0.7198599357834239
PoissonRegressor score =  0.6008111644235149
RANSACRegressor score =  -0.19405879385722757
RadiusNeighborsRegressor score =  0.34995122282473956
RandomForestRegressor score =  0.8883168599683604
RegressorChain error
Ridge score =  0.7043991328091959
RidgeCV score =  0.7139980049047578
SGDRegressor score =  0.6850574392379839
SVR score =  0.6296893340457187
StackingRegressor error
TheilSenRegressor score =  0.723727261733905
TransformedTargetRegressor score =  0.7149364161392221
TweedieRegressor score =  0.18180020556944754
VotingRegressor error
'''