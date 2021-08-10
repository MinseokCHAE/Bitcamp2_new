import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#1. Data 
x = [[0,0], [0,1], [1,0], [1,1]]
y = [0,0,0,1]

#2. Modeling
model = LinearSVC()

#3. Traning
model.fit(x,y)

#4. Evaluating
score = model.score(x,y)
print('score = ', score)
# score =  1.0
