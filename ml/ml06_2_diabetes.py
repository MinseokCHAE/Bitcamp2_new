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
ARDRegression score =  [0.41166708 0.45593267 0.52742293 0.48641814 0.5368448 ]
ARDRegression score_avg =  0.48365712325058724
AdaBoostRegressor score =  [0.30670831 0.49070823 0.38045219 0.47021483 0.43692446]
AdaBoostRegressor score_avg =  0.4170016038937397
BaggingRegressor score =  [0.22014777 0.4218327  0.40379857 0.41670044 
0.40972451]
BaggingRegressor score_avg =  0.3744407973888408
BayesianRidge score =  [0.42367535 0.45832082 0.53323775 0.48430705 0.53797237]
BayesianRidge score_avg =  0.4875026665653702
CCA score =  [0.41023939 0.43871016 0.53521706 0.40314861 0.48121268]
CCA score_avg =  0.4537055784292295
DecisionTreeRegressor score =  [-0.28534112 -0.12247678 -0.13228048 -0.0677934  -0.32201825]
DecisionTreeRegressor score_avg =  -0.18598200635687237
DummyRegressor score =  [-0.00074015 -0.01626389 -0.0041107  -0.0077899  -0.00225822]
DummyRegressor score_avg =  -0.0062325724442134686
ElasticNet score =  [ 0.00707447 -0.00765485  0.00412687  0.00162718  0.00731334]
ElasticNet score_avg =  0.0024973997011079607
ElasticNetCV score =  [0.35621397 0.42497021 0.44121926 0.47465703 0.49664882]
ElasticNetCV score_avg =  0.43874185990983944
ExtraTreeRegressor score =  [-0.46341347 -0.038206    0.11828954 -0.10217675 -0.16304343]
ExtraTreeRegressor score_avg =  -0.12971002163500742
ExtraTreesRegressor score =  [0.33470109 0.39951837 0.43372636 0.47254689 0.46992446]
ExtraTreesRegressor score_avg =  0.422083434487799
GammaRegressor score =  [ 0.00447972 -0.00887984  0.00165697 -0.00045815  0.00453024]
GammaRegressor score_avg =  0.000265787130283357
GaussianProcessRegressor score =  [-11.8531252   -9.04519487  -8.30195296 -10.3189341  -13.42564455]
GaussianProcessRegressor score_avg =  -10.588970338392585
GradientBoostingRegressor score =  [0.28059557 0.36969445 0.42061137 0.46262613 0.3518016 ]
GradientBoostingRegressor score_avg =  0.37706582312378084
HistGradientBoostingRegressor score =  [0.20673828 0.46093754 0.41358534 0.43042839 0.41856182]
HistGradientBoostingRegressor score_avg =  0.38605027423091426
HuberRegressor score =  [0.43073345 0.44975566 0.54555734 0.46419922 0.52562183]
HuberRegressor score_avg =  0.48317349974487345
IsotonicRegression score =  [nan nan nan nan nan]
IsotonicRegression score_avg =  nan
KNeighborsRegressor score =  [0.26641449 0.27583291 0.46434366 0.42364649 0.52268373]
KNeighborsRegressor score_avg =  0.3905842563397755
KernelRidge score =  [-3.91353342 -3.90330524 -2.89307995 -3.44646867 -3.7582174 ]
KernelRidge score_avg =  -3.5829209359636964
Lars score =  [ 0.42795359  0.45619816  0.53278477  0.15762329 -0.63899654]
Lars score_avg =  0.18711265199947297
LarsCV score =  [0.41117409 0.46627534 0.48887074 0.50035722 0.5372608 
]
LarsCV score_avg =  0.4807876371282007
Lasso score =  [0.32152731 0.35211304 0.35205153 0.35797934 0.35154017]Lasso score_avg =  0.34704227877425636
LassoCV score =  [0.41151975 0.45992556 0.5346741  0.48536309 0.53801482]
LassoCV score_avg =  0.4858994643415627
LassoLars score =  [0.33765188 0.376078   0.37600324 0.37846384 0.36986942]
LassoLars score_avg =  0.36761327565116286
LassoLarsCV score =  [0.41117409 0.46104413 0.53467696 0.48197355 0.53812772]
LassoLarsCV score_avg =  0.4853992901872326
LassoLarsIC score =  [0.41212622 0.4665654  0.53033874 0.49290435 0.53825909]
LassoLarsIC score_avg =  0.48803875964731525
LinearRegression score =  [0.42795359 0.45619816 0.53278477 0.48758196 
0.54201609]
LinearRegression score_avg =  0.4893069117795578
LinearSVR score =  [-0.37500922 -0.50087783 -0.25411147 -0.30116138 -0.43369491]
LinearSVR score_avg =  -0.37297096140912483
MLPRegressor score =  [-3.23882663 -3.04418447 -2.50303725 -2.86123761 
-3.24847339]
MLPRegressor score_avg =  -2.979151871684892
MultiOutputRegressor error
MultiTaskElasticNet score =  [nan nan nan nan nan]
MultiTaskElasticNet score_avg =  nan
MultiTaskElasticNetCV score =  [nan nan nan nan nan]
MultiTaskElasticNetCV score_avg =  nan
MultiTaskLasso score =  [nan nan nan nan nan]
MultiTaskLasso score_avg =  nan
MultiTaskLassoCV score =  [nan nan nan nan nan]
MultiTaskLassoCV score_avg =  nan
NuSVR score =  [0.12725541 0.13308921 0.16729823 0.19489124 0.16711328]NuSVR score_avg =  0.15792947554759665
OrthogonalMatchingPursuit score =  [0.26664544 0.35606425 0.34246154 0.36292548 0.34779376]
OrthogonalMatchingPursuit score_avg =  0.33517809549296496
OrthogonalMatchingPursuitCV score =  [0.41094666 0.45042521 0.52618448 
0.46611572 0.51011166]
OrthogonalMatchingPursuitCV score_avg =  0.47275674837493054
PLSCanonical score =  [-1.56781591 -1.18899951 -1.04740602 -1.00583869 
-1.32592458]
PLSCanonical score_avg =  -1.2271969430598824
PLSRegression score =  [0.42841551 0.44750558 0.52705672 0.49266417 0.53846024]
PLSRegression score_avg =  0.48682044355247533
PassiveAggressiveRegressor score =  [0.38864619 0.42509783 0.4850226  0.51102472 0.52351923]
PassiveAggressiveRegressor score_avg =  0.46666211568280663
PoissonRegressor score =  [0.27721833 0.31925393 0.31406583 0.36895707 
0.38296545]
PoissonRegressor score_avg =  0.33249212337620454
RANSACRegressor score =  [ 0.19687304  0.28477947  0.1326934   0.19777707 -0.02184218]
RANSACRegressor score_avg =  0.15805615972981651
RadiusNeighborsRegressor score =  [-0.00074015 -0.01626389 -0.0041107  
-0.0077899  -0.00225822]
RadiusNeighborsRegressor score_avg =  -0.0062325724442134686
RandomForestRegressor score =  [0.31734792 0.43616158 0.41279891 0.46389232 0.4301033 ]
RandomForestRegressor score_avg =  0.41206080540031464
RegressorChain error
Ridge score =  [0.34189835 0.40622993 0.4150199  0.45787601 0.4783576 ]Ridge score_avg =  0.41987636043771737
RidgeCV score =  [0.41764492 0.46066328 0.52508197 0.49145738 0.53962919]
RidgeCV score_avg =  0.486895348124774
SGDRegressor score =  [0.31753015 0.39388887 0.39253631 0.45940799 0.47013675]
SGDRegressor score_avg =  0.40670001702198144
SVR score =  [0.13008745 0.09560214 0.18716149 0.20297682 0.16329947]
SVR score_avg =  0.15582547564561877
StackingRegressor error
TheilSenRegressor score =  [0.43007956 0.44881631 0.54758856 0.44966547 0.51459461]
TheilSenRegressor score_avg =  0.47814890317921604
TransformedTargetRegressor score =  [0.42795359 0.45619816 0.53278477 0.48758196 0.54201609]
TransformedTargetRegressor score_avg =  0.4893069117795578
TweedieRegressor score =  [ 0.00479416 -0.00986478  0.00186007 -0.00059152  0.0050004 ]
TweedieRegressor score_avg =  0.00023966573664784273
VotingRegressor error
'''
