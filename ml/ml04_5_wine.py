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
allAlgorithms = all_estimators(type_filter='classifier')
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
number of models =  41
AdaBoostClassifier score =  0.9444444444444444
BaggingClassifier score =  1.0
BernoulliNB score =  0.2777777777777778
CalibratedClassifierCV score =  0.9722222222222222
CategoricalNB error
ClassifierChain error
ComplementNB score =  0.8888888888888888
DecisionTreeClassifier score =  0.8333333333333334
DummyClassifier score =  0.2777777777777778
ExtraTreeClassifier score =  0.9444444444444444
ExtraTreesClassifier score =  1.0
GaussianNB score =  1.0
GaussianProcessClassifier score =  1.0
GradientBoostingClassifier score =  0.9444444444444444
HistGradientBoostingClassifier score =  1.0
KNeighborsClassifier score =  0.9722222222222222
LabelPropagation score =  1.0
LabelSpreading score =  1.0
LinearDiscriminantAnalysis score =  1.0
LinearSVC score =  0.9722222222222222
LogisticRegression score =  1.0
LogisticRegressionCV score =  0.9722222222222222
MLPClassifier score =  1.0
MultiOutputClassifier error
MultinomialNB score =  0.9722222222222222
NearestCentroid score =  1.0
NuSVC score =  1.0
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier score =  0.9722222222222222
Perceptron score =  0.9722222222222222
QuadraticDiscriminantAnalysis score =  1.0
RadiusNeighborsClassifier score =  0.9444444444444444
RandomForestClassifier score =  1.0
RidgeClassifier score =  0.9722222222222222
RidgeClassifierCV score =  0.9722222222222222
SGDClassifier score =  0.9722222222222222
SVC score =  1.0
StackingClassifier error
VotingClassifier error
'''