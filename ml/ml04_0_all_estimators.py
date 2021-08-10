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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

#2. modeling, training, evaluating
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier') # regressor
# print(allAlgorithms)
print('number of models = ', len(allAlgorithms))
for (name, model) in allAlgorithms:
    try:
        model = model()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, 'acc = ', acc)
    except:
        print(name, 'error')

'''
number of models =  41
AdaBoostClassifier acc =  0.8333333333333334
BaggingClassifier acc =  0.9333333333333333
BernoulliNB acc =  0.23333333333333334
CalibratedClassifierCV acc =  0.9333333333333333
CategoricalNB acc =  0.8333333333333334
ClassifierChain error
ComplementNB acc =  0.6
DecisionTreeClassifier acc =  0.9333333333333333
DummyClassifier acc =  0.23333333333333334
ExtraTreeClassifier acc =  0.9
ExtraTreesClassifier acc =  0.9333333333333333
GaussianNB acc =  0.9666666666666667
GaussianProcessClassifier acc =  0.9
GradientBoostingClassifier acc =  0.9333333333333333
HistGradientBoostingClassifier acc =  0.9333333333333333
KNeighborsClassifier acc =  0.9666666666666667
LabelPropagation acc =  0.9666666666666667
LabelSpreading acc =  0.9666666666666667
LinearDiscriminantAnalysis acc =  0.9666666666666667
LinearSVC acc =  0.9333333333333333
LogisticRegression acc =  0.9333333333333333
LogisticRegressionCV acc =  0.9333333333333333
MLPClassifier acc =  0.9666666666666667
MultiOutputClassifier error
MultinomialNB acc =  0.7333333333333333
NearestCentroid acc =  0.9
NuSVC acc =  0.9
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier acc =  0.8
Perceptron acc =  0.7
QuadraticDiscriminantAnalysis acc =  0.9666666666666667
RadiusNeighborsClassifier acc =  0.9
RandomForestClassifier acc =  0.9333333333333333
RidgeClassifier acc =  0.9
RidgeClassifierCV acc =  0.9
SGDClassifier acc =  0.8666666666666667
SVC acc =  0.9
StackingClassifier error
VotingClassifier error
'''
