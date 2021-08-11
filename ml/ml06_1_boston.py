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
ARDRegression score =  [0.71421746 0.69346113 0.64672971 0.77135566 0.65585406]
ARDRegression score_avg =  0.6963236048351433
AdaBoostRegressor score =  [0.79890561 0.90552913 0.79675605 0.81678232 0.84049207]
AdaBoostRegressor score_avg =  0.8316930359811497
BaggingRegressor score =  [0.89714953 0.85292705 0.79153766 0.88573647 
0.82738773]
BaggingRegressor score_avg =  0.850947688014994
BayesianRidge score =  [0.70838247 0.6863053  0.67003199 0.75075272 0.69043366]
BayesianRidge score_avg =  0.7011812291099918
CCA score =  [0.69399762 0.66165719 0.55063977 0.73724156 0.618069  ]
CCA score_avg =  0.6523210265204022
DecisionTreeRegressor score =  [0.70571217 0.69145133 0.73286325 0.77103803 0.74383102]
DecisionTreeRegressor score_avg =  0.7289791616845696
DummyRegressor score =  [-0.00077171 -0.00118678 -0.00101423 -0.00838146 -0.01215177]
DummyRegressor score_avg =  -0.004701188531834655
ElasticNet score =  [0.65029654 0.64920684 0.63176807 0.70344103 0.69533231]
ElasticNet score_avg =  0.6660089590203387
ElasticNetCV score =  [0.63453717 0.63347307 0.61753723 0.69023669 0.68219997]
ElasticNetCV score_avg =  0.6515968275578726
ExtraTreeRegressor score =  [0.73810985 0.65964035 0.67613076 0.79470969 0.69257532]
ExtraTreeRegressor score_avg =  0.7122331952595545
ExtraTreesRegressor score =  [0.88912761 0.90845731 0.84310145 0.90897127 0.90242001]
ExtraTreesRegressor score_avg =  0.8904155312106408
GammaRegressor score =  [-0.00076689 -0.00135794 -0.00100112 -0.00798047 -0.01335018]
GammaRegressor score_avg =  -0.004891318376723408
GaussianProcessRegressor score =  [-5.09309503 -5.64625069 -8.03102191 
-5.51644025 -5.93929455]
GaussianProcessRegressor score_avg =  -6.045220485911516
GradientBoostingRegressor score =  [0.91170316 0.91853332 0.82181752 0.9075479  0.88255024]
GradientBoostingRegressor score_avg =  0.8884304285320648
HistGradientBoostingRegressor score =  [0.86647845 0.89117542 0.80027571 0.88807516 0.90773708]
HistGradientBoostingRegressor score_avg =  0.8707483647815677
HuberRegressor score =  [0.71931702 0.53173947 0.72017988 0.73222515 0.61609249]
HuberRegressor score_avg =  0.6639108006270913
IsotonicRegression score =  [nan nan nan nan nan]
IsotonicRegression score_avg =  nan
KNeighborsRegressor score =  [0.53153447 0.46279674 0.45686016 0.59738263 0.54965116]
KNeighborsRegressor score_avg =  0.5196450309808538
KernelRidge score =  [0.71318402 0.65750221 0.62261363 0.76319609 0.63974352]
KernelRidge score_avg =  0.6792478956908097
Lars score =  [0.71397311 0.70395769 0.6797936  0.77347282 0.67141629]
Lars score_avg =  0.7085227016343094
LarsCV score =  [0.7032078  0.67278925 0.64460721 0.7661506  0.66053489]
LarsCV score_avg =  0.6894579510861198
Lasso score =  [0.6385523  0.64688578 0.62861685 0.70277375 0.69213953]Lasso score_avg =  0.6617936410970364
LassoCV score =  [0.63298626 0.66383039 0.60098111 0.71787133 0.70618609]
LassoCV score_avg =  0.6643710350222165
LassoLars score =  [-0.00077171 -0.00118678 -0.00101423 -0.00838146 -0.01215177]
LassoLars score_avg =  -0.004701188531834655
LassoLarsCV score =  [0.70322436 0.68353668 0.64460721 0.76972788 0.66150634]
LassoLarsCV score_avg =  0.692520494018266
LassoLarsIC score =  [0.71516068 0.69979139 0.67650079 0.77359532 0.66080012]
LassoLarsIC score_avg =  0.7051696591904933
LinearRegression score =  [0.71493642 0.70586907 0.6797936  0.77347282 
0.68069476]
LinearRegression score_avg =  0.7109533328842113
LinearSVR score =  [ 0.39830628  0.44762759  0.51138457 -0.89903866 -0.05416749]
LinearSVR score_avg =  0.08082246087236813
MLPRegressor score =  [0.41538031 0.3510339  0.47757038 0.56667171 0.61879119]
MLPRegressor score_avg =  0.4858894984194054
MultiOutputRegressor error
MultiTaskElasticNet score =  [nan nan nan nan nan]
MultiTaskElasticNet score_avg =  nan
MultiTaskElasticNetCV score =  [nan nan nan nan nan]
MultiTaskElasticNetCV score_avg =  nan
MultiTaskLasso score =  [nan nan nan nan nan]
MultiTaskLasso score_avg =  nan
MultiTaskLassoCV score =  [nan nan nan nan nan]
MultiTaskLassoCV score_avg =  nan
NuSVR score =  [0.27187434 0.15120976 0.32833268 0.20634417 0.22061373]NuSVR score_avg =  0.2356749355593375
OrthogonalMatchingPursuit score =  [0.52545112 0.54037336 0.4635281  0.58285394 0.55839723]
OrthogonalMatchingPursuit score_avg =  0.5341207486165975
OrthogonalMatchingPursuitCV score =  [0.67699522 0.67532314 0.57645448 
0.72186546 0.62607884]
OrthogonalMatchingPursuitCV score_avg =  0.6553434269544749
PLSCanonical score =  [-2.13496173 -2.13601816 -2.77841906 -2.28355661 
-1.65380379]
PLSCanonical score_avg =  -2.197351868556098
PLSRegression score =  [0.7000163  0.68661117 0.66332479 0.75625075 0.62589243]
PLSRegression score_avg =  0.6864190875901961
PassiveAggressiveRegressor score =  [-0.15451942  0.14658663 -6.96067229  0.3319265   0.31181267]
PassiveAggressiveRegressor score_avg =  -1.264973183852669
PoissonRegressor score =  [0.75090256 0.73880299 0.68232602 0.82374084 
0.77541141]
PoissonRegressor score_avg =  0.7542367651930366
RANSACRegressor score =  [0.11979229 0.42582708 0.68028939 0.27272349 0.0055397 ]
RANSACRegressor score_avg =  0.3008343900700326
RadiusNeighborsRegressor score =  [nan nan nan nan nan]
RadiusNeighborsRegressor score_avg =  nan
RandomForestRegressor score =  [0.89204553 0.88581946 0.81440087 0.89427956 0.8593451 ]
RandomForestRegressor score_avg =  0.8691781063018233
RegressorChain error
Ridge score =  [0.71499353 0.69860228 0.67609811 0.76830654 0.68411714]Ridge score_avg =  0.7084235186738977
RidgeCV score =  [0.7152345  0.70497445 0.67959711 0.77295647 0.68141453]
RidgeCV score_avg =  0.7108354121454284
SGDRegressor score =  [-1.72047389e+26 -3.98078375e+26 -5.19788027e+26 
-5.59383530e+25
 -5.66863353e+25]
SGDRegressor score_avg =  -2.4050769594241673e+26
SVR score =  [0.25312932 0.11418755 0.28673406 0.16480141 0.18900806]
SVR score_avg =  0.20157207842280522
StackingRegressor error
TheilSenRegressor score =  [0.71348793 0.6341025  0.70071305 0.74254847 0.62869863]
TheilSenRegressor score_avg =  0.6839101149564857
TransformedTargetRegressor score =  [0.71493642 0.70586907 0.6797936  0.77347282 0.68069476]
TransformedTargetRegressor score_avg =  0.7109533328842113
TweedieRegressor score =  [0.63183805 0.63866268 0.60463105 0.69694137 
0.67193309]
TweedieRegressor score_avg =  0.6488012502184606
VotingRegressor error
'''
