import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, MaxPooling1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
datasets = load_wine()
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
allAlgorithms = all_estimators(type_filter='classifier')

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
AdaBoostClassifier score =  [0.94444444 0.80555556 0.94444444 0.88571429 0.94285714]
AdaBoostClassifier score_avg =  0.9046031746031746
BaggingClassifier score =  [1.         0.94444444 0.91666667 0.94285714 0.94285714]
BaggingClassifier score_avg =  0.9493650793650794
BernoulliNB score =  [0.27777778 0.36111111 0.47222222 0.42857143 0.45714286]
BernoulliNB score_avg =  0.39936507936507937
CalibratedClassifierCV score =  [0.94444444 0.88888889 0.94444444 0.88571429 0.94285714]
CalibratedClassifierCV score_avg =  0.9212698412698412
CategoricalNB score =  [nan nan nan nan nan]
CategoricalNB score_avg =  nan
ClassifierChain error
ComplementNB score =  [0.61111111 0.52777778 0.72222222 0.65714286 0.68571429]
ComplementNB score_avg =  0.6407936507936508
DecisionTreeClassifier score =  [0.83333333 0.83333333 0.88888889 0.91428571 0.94285714]
DecisionTreeClassifier score_avg =  0.8825396825396824
DummyClassifier score =  [0.27777778 0.36111111 0.47222222 0.42857143 0.45714286]
DummyClassifier score_avg =  0.39936507936507937
ExtraTreeClassifier score =  [0.88888889 0.77777778 0.77777778 0.8     
   0.97142857]
ExtraTreeClassifier score_avg =  0.8431746031746032
ExtraTreesClassifier score =  [1.         1.         0.94444444 0.97142857 1.        ]
ExtraTreesClassifier score_avg =  0.9831746031746033
GaussianNB score =  [1.         0.97222222 0.97222222 0.94285714 1.    
    ]
GaussianNB score_avg =  0.9774603174603176
GaussianProcessClassifier score =  [0.58333333 0.52777778 0.5        0.42857143 0.4       ]
GaussianProcessClassifier score_avg =  0.4879365079365079
GradientBoostingClassifier score =  [0.94444444 0.94444444 0.91666667 0.91428571 0.94285714]
GradientBoostingClassifier score_avg =  0.9325396825396824
HistGradientBoostingClassifier score =  [1.         1.         0.91666667 0.94285714 0.97142857]
HistGradientBoostingClassifier score_avg =  0.9661904761904762
KNeighborsClassifier score =  [0.66666667 0.63888889 0.77777778 0.68571429 0.65714286]
KNeighborsClassifier score_avg =  0.6852380952380952
LabelPropagation score =  [0.55555556 0.36111111 0.61111111 0.4        
0.51428571]
LabelPropagation score_avg =  0.48841269841269846
LabelSpreading score =  [0.55555556 0.36111111 0.61111111 0.4        0.51428571]
LabelSpreading score_avg =  0.48841269841269846
LinearDiscriminantAnalysis score =  [1.         0.97222222 1.         0.97142857 1.        ]
LinearDiscriminantAnalysis score_avg =  0.9887301587301588
LinearSVC score =  [0.80555556 0.91666667 0.97222222 0.91428571 0.8    
   ]
LinearSVC score_avg =  0.8817460317460318
LogisticRegression score =  [1.         0.91666667 0.91666667 0.94285714 0.94285714]
LogisticRegression score_avg =  0.9438095238095239
LogisticRegressionCV score =  [1.         0.94444444 0.91666667 0.94285714 0.94285714]
LogisticRegressionCV score_avg =  0.9493650793650794
MLPClassifier score =  [0.52777778 0.58333333 0.80555556 0.6        0.22857143]
MLPClassifier score_avg =  0.549047619047619
MultiOutputClassifier error
MultinomialNB score =  [0.75       0.88888889 0.77777778 0.88571429 0.91428571]
MultinomialNB score_avg =  0.8433333333333334
NearestCentroid score =  [0.63888889 0.75       0.75       0.71428571 0.77142857]
NearestCentroid score_avg =  0.724920634920635
NuSVC score =  [0.83333333 0.88888889 0.80555556 0.71428571 0.94285714]NuSVC score_avg =  0.836984126984127
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier score =  [0.38888889 0.44444444 0.86111111 
0.34285714 0.48571429]
PassiveAggressiveClassifier score_avg =  0.5046031746031746
Perceptron score =  [0.36111111 0.61111111 0.11111111 0.65714286 0.51428571]
Perceptron score_avg =  0.45095238095238094
QuadraticDiscriminantAnalysis score =  [1.         1.         1.       
  0.97142857 1.        ]
QuadraticDiscriminantAnalysis score_avg =  0.9942857142857143
RadiusNeighborsClassifier score =  [nan nan nan nan nan]
RadiusNeighborsClassifier score_avg =  nan
RandomForestClassifier score =  [1.         1.         0.94444444 0.97142857 0.97142857]
RandomForestClassifier score_avg =  0.9774603174603176
RidgeClassifier score =  [0.97222222 1.         0.97222222 1.         1.        ]
RidgeClassifier score_avg =  0.9888888888888889
RidgeClassifierCV score =  [0.97222222 1.         0.97222222 1.        
 1.        ]
RidgeClassifierCV score_avg =  0.9888888888888889
SGDClassifier score =  [0.61111111 0.52777778 0.69444444 0.51428571 0.6       ]
SGDClassifier score_avg =  0.5895238095238095
SVC score =  [0.61111111 0.52777778 0.75       0.65714286 0.77142857]
SVC score_avg =  0.6634920634920635
StackingClassifier error
VotingClassifier error
'''
