
x = [-3,  31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

import pandas as pd
df = pd.DataFrame({'X' : x, 'Y' : y})
# print(df, df.shape)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
x_train = x_train.values.reshape(len(x_train),1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
# print('score = ', score)    #   1.0

coef = model.coef_
bias = model.intercept_
# print('coef = ', coef)  # [2.]
# print('bias = ', bias)  # 3.0
