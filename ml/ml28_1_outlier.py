import  numpy as np
data1 = np.array([1,2,-100,4,5,6,7,8,90,100,500])

def outlier_single(data_out):
    q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
    print('quartile1 = ', q1)
    print('quartile2 = ', q2)
    print('quartile3 = ', q3)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outlier_single(data1)
# print('outlier = ', outlier_loc)

import matplotlib.pyplot as plt
plt.boxplot([data1], notch=True, vert=False)
# plt.show()

data2 = np.array([[1,2,10000,3,4,6,7,8,90,100,5000],
                                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
data2 = data2.transpose()

def outlier_multi(data_out):
    list = []
    for i in range(data_out.shape[1]):
        q1, q2, q3 = np.percentile(data_out[:, i], [25, 50, 75])
        print('quartile1 = ', q1)
        print('quartile2 = ', q2)
        print('quartile3 = ', q3)
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        m = np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        n = np.count_nonzero((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        list.append(['columns', i+1, m, 'outlier_num :', n])
    return np.array(list)

outlier_loc = outlier_multi(data2)
print('outlier = ', outlier_loc)
plt.boxplot(data2, notch=True, vert=False)
plt.show()
