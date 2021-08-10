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
AdaBoostClassifier score =  0.9824561403508771
BaggingClassifier score =  0.9473684210526315
BernoulliNB score =  0.6578947368421053
CalibratedClassifierCV score =  0.9824561403508771
CategoricalNB error
ClassifierChain error
ComplementNB score =  0.8859649122807017
DecisionTreeClassifier score =  0.9122807017543859
DummyClassifier score =  0.6578947368421053
ExtraTreeClassifier score =  0.9385964912280702
ExtraTreesClassifier score =  0.9736842105263158
GaussianNB score =  0.9385964912280702
GaussianProcessClassifier score =  0.9736842105263158
GradientBoostingClassifier score =  0.9649122807017544
HistGradientBoostingClassifier score =  0.9736842105263158
KNeighborsClassifier score =  0.9912280701754386
LabelPropagation score =  0.9736842105263158
LabelSpreading score =  0.9736842105263158
LinearDiscriminantAnalysis score =  0.9649122807017544
LinearSVC score =  1.0
LogisticRegression score =  0.9736842105263158
LogisticRegressionCV score =  1.0
MLPClassifier score =  0.9912280701754386
MultiOutputClassifier error
MultinomialNB score =  0.8245614035087719
NearestCentroid score =  0.9298245614035088
NuSVC score =  0.9385964912280702
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier score =  0.9649122807017544
Perceptron score =  0.9824561403508771
QuadraticDiscriminantAnalysis score =  0.9473684210526315
RadiusNeighborsClassifier error
RandomForestClassifier score =  0.9649122807017544
RidgeClassifier score =  0.9649122807017544
RidgeClassifierCV score =  0.9649122807017544
SGDClassifier score =  0.9824561403508771
SVC score =  0.9912280701754386
StackingClassifier error
VotingClassifier error
'''