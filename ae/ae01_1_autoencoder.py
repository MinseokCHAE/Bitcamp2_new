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

# autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.fit(x_train, x_train, epochs=32, batch_size=128, validation_split=0.1)

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

# plt.show()


def autoencoder(hidden_layer_size):
    input = Input((784,))
    xx = Dense(units=hidden_layer_size, activation='relu')(input)
    output = Dense(784, activation='relu')(xx)
    model = Model(input, output)
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95%
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.01)
img_decoded = model.predict(x_test)

import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))
random_images = random.sample(range(img_decoded.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Input', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(img_decoded[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Output', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
