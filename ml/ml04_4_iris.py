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
AdaBoostClassifier score =  0.8333333333333334
BaggingClassifier score =  0.9333333333333333
BernoulliNB score =  0.23333333333333334
CalibratedClassifierCV score =  0.9333333333333333
CategoricalNB score =  0.8333333333333334
ClassifierChain error
ComplementNB score =  0.6
DecisionTreeClassifier score =  0.9333333333333333
DummyClassifier score =  0.23333333333333334
ExtraTreeClassifier score =  0.9333333333333333
ExtraTreesClassifier score =  0.9333333333333333
GaussianNB score =  0.9666666666666667
GaussianProcessClassifier score =  0.9
GradientBoostingClassifier score =  0.9333333333333333
HistGradientBoostingClassifier score =  0.9333333333333333
KNeighborsClassifier score =  0.9666666666666667
LabelPropagation score =  0.9666666666666667
LabelSpreading score =  0.9666666666666667
LinearDiscriminantAnalysis score =  0.9666666666666667
LinearSVC score =  0.9333333333333333
LogisticRegression score =  0.9333333333333333
LogisticRegressionCV score =  0.9333333333333333
MLPClassifier score =  0.9333333333333333
MultiOutputClassifier error
MultinomialNB score =  0.7333333333333333
NearestCentroid score =  0.9
NuSVC score =  0.9
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier score =  0.8666666666666667
Perceptron score =  0.7
QuadraticDiscriminantAnalysis score =  0.9666666666666667
RadiusNeighborsClassifier score =  0.9
RandomForestClassifier score =  0.9333333333333333
RidgeClassifier score =  0.9
RidgeClassifierCV score =  0.9
SGDClassifier score =  0.6666666666666666
SVC score =  0.9
StackingClassifier error
VotingClassifier error
'''