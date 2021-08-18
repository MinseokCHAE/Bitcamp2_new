from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

dataset = load_iris()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=21)
kmean.fit(df)

df['cluster'] = kmean.labels_
df['target'] = dataset.target

# print(dataset.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
result = df.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(result)

