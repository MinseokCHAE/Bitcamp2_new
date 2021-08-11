import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21) 
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. modeling, training, evaluating
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='regressor') 

from sklearn.model_selection import KFold, cross_val_score
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=21)

print('number of models = ', len(allAlgorithms))
for (name, model) in allAlgorithms:
    try:
        model = model()
        score = cross_val_score(model, x, y, cv=kfold)
        print(name, 'score = ', score)
        print(name, 'score_avg = ', sum(score)/n_splits)
    except:
        print(name, 'error')

'''
number of models =  54
ARDRegression score =  [0.76673037 0.67634234 0.68505898 0.64198225 0.72640206]
ARDRegression score_avg =  0.6993031999442378
AdaBoostRegressor score =  [0.93223083 0.84291974 0.84416184 0.85413347 0.92640239]
AdaBoostRegressor score_avg =  0.8799696551939856
BaggingRegressor score =  [0.8328     0.79454365 0.79241564 0.82497331 
0.92765244]
BaggingRegressor score_avg =  0.8344770077410113
BayesianRidge score =  [0.73668792 0.70494965 0.70144943 0.71313796 0.7541334 ]
BayesianRidge score_avg =  0.7220716725910166
CCA score =  [0.46921508 0.50138157 0.31197538 0.2518386  0.39126496]
CCA score_avg =  0.3851351176972565
DecisionTreeRegressor score =  [0.72717949 0.54761905 0.77146676 0.67786499 0.80860434]
DecisionTreeRegressor score_avg =  0.70654692375121
DummyRegressor score =  [-0.00645453 -0.00011645 -0.00113604 -0.02090825 -0.00064046]
DummyRegressor score_avg =  -0.005851144649421958
ElasticNet score =  [0.64012302 0.5865359  0.59158748 0.61688424 0.63857962]
ElasticNet score_avg =  0.6147420495341744
ElasticNetCV score =  [0.6833487  0.6228952  0.60461246 0.66404369 0.68383434]
ElasticNetCV score_avg =  0.6517468799254302
ExtraTreeRegressor score =  [0.68820513 0.84920635 0.77146676 0.78524333 0.80860434]
ExtraTreeRegressor score_avg =  0.7805451794637515
ExtraTreesRegressor score =  [0.90159364 0.86430079 0.84180929 0.86564107 0.93655617]
ExtraTreesRegressor score_avg =  0.8819801911662942
GammaRegressor score =  [nan nan nan nan nan]
GammaRegressor score_avg =  nan
GaussianProcessRegressor score =  [-1.92307652 -1.71428571 -1.78041903 
-1.3265304  -1.75602862]
GaussianProcessRegressor score_avg =  -1.7000680558486223
GradientBoostingRegressor score =  [0.85299604 0.78726406 0.8115936  0.84972923 0.90638474]
GradientBoostingRegressor score_avg =  0.8415935345054478
HistGradientBoostingRegressor score =  [0.89319806 0.82583784 0.80278218 0.82945831 0.91231934]
HistGradientBoostingRegressor score_avg =  0.8527191461186282
HuberRegressor score =  [0.43244813 0.42856946 0.43898351 0.47585024 0.51169808]
HuberRegressor score_avg =  0.45750988437848206
IsotonicRegression score =  [nan nan nan nan nan]
IsotonicRegression score_avg =  nan
KNeighborsRegressor score =  [0.69911795 0.74365079 0.70290678 0.80958242 0.84228997]
KNeighborsRegressor score_avg =  0.7595095830686741
KernelRidge score =  [0.66145691 0.58433245 0.67926894 0.61043562 0.69015082]
KernelRidge score_avg =  0.6451289473022802
Lars score =  [ -701.02169977  -197.70151639 -1281.13855426  -252.9537446
  -427.1807126 ]
Lars score_avg =  -571.9992455251956
LarsCV score =  [0.12524979 0.08001882 0.54571679 0.26506913 0.08277946]
LarsCV score_avg =  0.2197667991919922
Lasso score =  [0.57219708 0.54609251 0.4970622  0.55237909 0.57162818]Lasso score_avg =  0.5478718116407958
LassoCV score =  [0.68350724 0.62307638 0.60466279 0.66425631 0.68410627]
LassoCV score_avg =  0.6519217989088277
LassoLars score =  [-0.00645453 -0.00011645 -0.00113604 -0.02090825 -0.00064046]
LassoLars score_avg =  -0.005851144649421958
LassoLarsCV score =  [0.78590522 0.70426617 0.72624878 0.70774277 0.77104923]
LassoLarsCV score_avg =  0.7390424344699728
LassoLarsIC score =  [0.7620744  0.68213628 0.68586846 0.67151573 0.73357283]
LassoLarsIC score_avg =  0.7070335385683398
LinearRegression score =  [0.78223897 0.7053208  0.71370461 0.68059011 
0.76797577]
LinearRegression score_avg =  0.7299660503771175
LinearSVR score =  [ 0.37007502 -0.06030505  0.06702961  0.43774733  0.27512518]
LinearSVR score_avg =  0.21793441724681212
MLPRegressor score =  [-27.52387812  -2.69890746 -72.93209753  -0.95392131  -0.14495641]
MLPRegressor score_avg =  -20.850752166792155
MultiOutputRegressor error
MultiTaskElasticNet score =  [nan nan nan nan nan]
MultiTaskElasticNet score_avg =  nan
MultiTaskElasticNetCV score =  [nan nan nan nan nan]
MultiTaskElasticNetCV score_avg =  nan
MultiTaskLasso score =  [nan nan nan nan nan]
MultiTaskLasso score_avg =  nan
MultiTaskLassoCV score =  [nan nan nan nan nan]
MultiTaskLassoCV score_avg =  nan
NuSVR score =  [0.6789913  0.69102062 0.76562284 0.70073398 0.77130333]NuSVR score_avg =  0.7215344116483389
OrthogonalMatchingPursuit score =  [0.76295098 0.65323369 0.60403326 0.68559169 0.70145494]
OrthogonalMatchingPursuit score_avg =  0.6814529130872321
OrthogonalMatchingPursuitCV score =  [0.76143054 0.66085586 0.6223647  
0.65870965 0.71982052]
OrthogonalMatchingPursuitCV score_avg =  0.6846362525716564
PLSCanonical score =  [-7.3968505  -7.5109187  -6.78709919 -6.86621725 
-7.44588075]
PLSCanonical score_avg =  -7.201393276998668
PLSRegression score =  [0.73644701 0.66134668 0.65537358 0.69077488 0.72310979]
PLSRegression score_avg =  0.6934103884138609
PassiveAggressiveRegressor score =  [-0.10015344  0.10179477 -0.32432913 -0.16677978 -0.98876442]
PassiveAggressiveRegressor score_avg =  -0.29564639860294106
PoissonRegressor score =  [-4.16982118e-03 -7.43169956e-05 -7.28008326e-04 -1.29539291e-02
 -4.09803494e-04]
PoissonRegressor score_avg =  -0.0036671758241189067
RANSACRegressor score =  [nan nan nan nan nan]
RANSACRegressor score_avg =  nan
RadiusNeighborsRegressor score =  [-1.92307692 -1.71428571 -1.7804878  
-1.32653061 -1.75609756]
RadiusNeighborsRegressor score_avg =  -1.7000957230922389
RandomForestRegressor score =  [0.86459918 0.8249248  0.80184263 0.85645306 0.9198167 ]
RandomForestRegressor score_avg =  0.8535272751301732
RegressorChain error
Ridge score =  [0.73999367 0.70028449 0.7051493  0.71336119 0.75170059]Ridge score_avg =  0.7220978504903719
RidgeCV score =  [0.7651764  0.70617163 0.72328116 0.72078193 0.76099429]
RidgeCV score_avg =  0.7352810815544194
SGDRegressor score =  [-2.22298466e+29 -4.39811772e+29 -3.28540774e+29 
-1.24387912e+30
 -2.25598233e+29]
SGDRegressor score_avg =  -4.920256722746502e+29
SVR score =  [0.68572601 0.69939645 0.76728642 0.71250847 0.78248399]
SVR score_avg =  0.7294802694751977
StackingRegressor error
TheilSenRegressor score =  [0.66050048 0.56559997 0.62739414 0.6214738 
 0.65717701]
TheilSenRegressor score_avg =  0.6264290809434667
TransformedTargetRegressor score =  [0.78223897 0.7053208  0.71370461 0.68059011 0.76797577]
TransformedTargetRegressor score_avg =  0.7299660503771175
TweedieRegressor score =  [0.61686688 0.53671434 0.52981652 0.57706003 
0.60671294]
TweedieRegressor score_avg =  0.5734341430059272
VotingRegressor error

'''
