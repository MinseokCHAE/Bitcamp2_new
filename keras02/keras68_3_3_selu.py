import numpy as np
import matplotlib.pyplot as plt

def selu(x, a, b):
        return (x>0) * b * x + (x<=0) * b * ( a * (np.exp(x) - 1 ))

a = 0.1
b = 0.1
x = np.arange(-5, 5, 0.1)
y = selu(x, a, b)

plt.plot(x, y)
plt.grid()
plt.show()
