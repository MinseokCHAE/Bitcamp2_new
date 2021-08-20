import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, a):
    return np.maximum(a * x, x)

a = 0.1
x = np.arange(-5, 5, 0.1)
y = leaky_relu(x, a)

plt.plot(x, y)
plt.grid()
plt.show()
