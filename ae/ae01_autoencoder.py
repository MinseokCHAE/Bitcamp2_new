import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input = Input((784,))
encoded = Dense(64, activation='relu')(input)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input, decoded)
# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train, x_train, epochs=32, batch_size=128, validation_split=0.1)

img_decoded = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(img_decoded[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
