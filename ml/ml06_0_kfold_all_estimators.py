import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. data preprocessing
datasets = load_iris()
x = datasets.data
y = datasets.target
scaler = StandardScaler()
scaler.fit_transform(x)

#2. modeling, training, evaluating
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier') # regressor

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
number of models =  41
AdaBoostClassifier score =  [0.83333333 0.96666667 0.93333333 0.96666667 0.93333333]
AdaBoostClassifier score_avg =  0.9266666666666667
BaggingClassifier score =  [0.93333333 0.96666667 0.96666667 1.        
 0.93333333]
BaggingClassifier score_avg =  0.96
BernoulliNB score =  [0.23333333 0.3        0.26666667 0.26666667 0.23333333]
BernoulliNB score_avg =  0.26
CalibratedClassifierCV score =  [0.93333333 0.93333333 0.96666667 0.9  
      0.8       ]
CalibratedClassifierCV score_avg =  0.9066666666666666
CategoricalNB score =  [0.83333333 0.93333333 0.93333333 1.         0.96666667]
CategoricalNB score_avg =  0.9333333333333333
ClassifierChain error
ComplementNB score =  [0.6        0.7        0.7        0.66666667 0.66666667]
ComplementNB score_avg =  0.6666666666666666
DecisionTreeClassifier score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.93333333]
DecisionTreeClassifier score_avg =  0.9533333333333334
DummyClassifier score =  [0.23333333 0.3        0.26666667 0.26666667 0.23333333]
DummyClassifier score_avg =  0.26
ExtraTreeClassifier score =  [0.93333333 0.93333333 0.86666667 1.      
   0.86666667]
ExtraTreeClassifier score_avg =  0.9199999999999999
ExtraTreesClassifier score =  [0.93333333 0.96666667 0.96666667 1.     
    0.93333333]
ExtraTreesClassifier score_avg =  0.96
GaussianNB score =  [0.96666667 0.93333333 0.93333333 1.         0.93333333]
GaussianNB score_avg =  0.9533333333333334
GaussianProcessClassifier score =  [0.9        1.         0.96666667 1.         0.96666667]
GaussianProcessClassifier score_avg =  0.9666666666666666
GradientBoostingClassifier score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.93333333]
GradientBoostingClassifier score_avg =  0.9533333333333334
HistGradientBoostingClassifier score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.9       ]
HistGradientBoostingClassifier score_avg =  0.9466666666666667
KNeighborsClassifier score =  [0.96666667 1.         0.96666667 1.     
    1.        ]
KNeighborsClassifier score_avg =  0.9866666666666667
LabelPropagation score =  [0.96666667 0.96666667 0.96666667 1.
0.9       ]
LabelPropagation score_avg =  0.96
LabelSpreading score =  [0.96666667 0.96666667 0.96666667 1.         0.93333333]
LabelSpreading score_avg =  0.9666666666666666
LinearDiscriminantAnalysis score =  [0.96666667 0.96666667 1.         1.         0.96666667]
LinearDiscriminantAnalysis score_avg =  0.9800000000000001
LinearSVC score =  [0.93333333 0.96666667 0.96666667 0.96666667 0.9    
   ]
LinearSVC score_avg =  0.9466666666666667
LogisticRegression score =  [0.93333333 0.96666667 0.96666667 1.       
  0.93333333]
LogisticRegression score_avg =  0.96
LogisticRegressionCV score =  [0.93333333 0.96666667 1.         1.     
    0.96666667]
LogisticRegressionCV score_avg =  0.9733333333333333
MLPClassifier score =  [0.96666667 0.96666667 1.         0.96666667 0.96666667]
MLPClassifier score_avg =  0.9733333333333334
MultiOutputClassifier error
MultinomialNB score =  [0.73333333 0.96666667 1.         0.93333333 0.86666667]
MultinomialNB score_avg =  0.9
NearestCentroid score =  [0.9        0.9        0.93333333 0.96666667 0.93333333]
NearestCentroid score_avg =  0.9266666666666667
NuSVC score =  [0.9        1.         0.96666667 1.         1.        ]NuSVC score_avg =  0.9733333333333334
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier score =  [0.96666667 0.83333333 0.8        
0.8        0.7       ]
PassiveAggressiveClassifier score_avg =  0.8200000000000001
Perceptron score =  [0.9        0.9        0.93333333 0.93333333 0.76666667]
Perceptron score_avg =  0.8866666666666667
QuadraticDiscriminantAnalysis score =  [0.96666667 0.93333333 1.       
  1.         0.96666667]
QuadraticDiscriminantAnalysis score_avg =  0.9733333333333333
RadiusNeighborsClassifier score =  [0.9        0.93333333 0.96666667 1.         0.96666667]
RadiusNeighborsClassifier score_avg =  0.9533333333333334
RandomForestClassifier score =  [0.93333333 0.96666667 0.93333333 0.96666667 0.93333333]
RandomForestClassifier score_avg =  0.9466666666666667
RidgeClassifier score =  [0.9        0.83333333 0.76666667 0.86666667 0.73333333]
RidgeClassifier score_avg =  0.82
RidgeClassifierCV score =  [0.9        0.83333333 0.76666667 0.86666667 0.73333333]
RidgeClassifierCV score_avg =  0.82
SGDClassifier score =  [0.9        0.93333333 0.8        0.96666667 0.9       ]
SGDClassifier score_avg =  0.9000000000000001
SVC score =  [0.9        1.         0.96666667 1.         1.        ]  
SVC score_avg =  0.9733333333333334
StackingClassifier error
VotingClassifier error
'''