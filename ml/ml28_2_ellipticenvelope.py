import  numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
                                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
data = data.transpose()

from sklearn.covariance import EllipticEnvelope

outlier = EllipticEnvelope(contamination=.2)
outlier.fit(data)

result = outlier.predict(data)
print('result = ', result)

plt.boxplot(data, notch=True, vert=False)
plt.show()
