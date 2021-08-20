import numpy as np
import matplotlib.pyplot as plt

def elu(x, a):
        return (x>0) * x + (x<=0) * ( a * (np.exp(x) - 1 ))

a = 0.1
x = np.arange(-5, 5, 0.1)
y = elu(x, a)

plt.plot(x, y)
plt.grid()
plt.show()
