import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def autoencoder1(hidden_layer_size):
    input = Input((784,))
    xx = Dense(units=hidden_layer_size, activation='relu')(input)
    output = Dense(784, activation='relu')(xx)
    model = Model(input, output)
    return model

def autoencoder2(hidden_layer_size):
    input = Input((784,))
    xx = Dense(units=hidden_layer_size, activation='relu')(input)
    xx = Dense(16, activation='relu')(xx)
    xx = Dense(16, activation='relu')(xx)
    xx = Dense(16, activation='relu')(xx)
    xx = Dense(16, activation='relu')(xx)
    output = Dense(784, activation='relu')(xx)
    model = Model(input, output)
    return model

model = autoencoder1(hidden_layer_size=154)  # pca 95%
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.01)
img_decoded1 = model.predict(x_test)

model = autoencoder2(hidden_layer_size=154)  # pca 95%
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.01)
img_decoded2 = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))
random_images = random.sample(range(img_decoded1.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Input', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(img_decoded1[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Output1', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(img_decoded2[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('Output2', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
