import  numpy as np
data = np.array([1,2,-100,4,5,6,7,8,90,100,500])

def outlier(data_out):
    q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
    print('quartile1 = ', q1)
    print('quartile2 = ', q2)
    print('quartile3 = ', q3)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outlier(data)
print('outlier = ', outlier_loc)

import matplotlib.pyplot as plt
plt.boxplot([data], notch=True, vert=False)
plt.show()
