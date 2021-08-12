import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

datasets = load_iris()
x = datasets.data
y = datasets.target
# print(datasets.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']
# print(datasets.target_names)
# ['setosa' 'versicolor' 'virginica']

# df = pd.Dataframe(x, columns=datasets.feature_names)
df = pd.DataFrame(x, columns=datasets['feature_names']) # 동일 방법
df['Target'] = y
# print(df.head())
# print(df.corr())
'''
                                sepal length (cm)  sepal width (cm)  ...  petal width (cm)    Target
sepal length (cm)           1.000000         -0.117570  ...          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000  ...         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440  ...          0.962865  0.949035
petal width (cm)            0.817941         -0.366126  ...          1.000000  0.956547
Target                      0.782561         -0.426658  ...          0.956547  1.000000
'''

sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()
