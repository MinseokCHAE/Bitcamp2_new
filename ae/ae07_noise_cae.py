import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

x_train_noised = x_train_noised.reshape(60000, 28, 28)
x_test_noised = x_test_noised.reshape(10000, 28, 28)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten

def autoencoder(hidden_layer_size):
    input = Input((28, 28))
    xx = Conv1D(hidden_layer_size, 2, activation='relu')(input)
    xx = MaxPooling1D()(xx)
    xx = Conv1D(hidden_layer_size, 2, activation='relu')(xx)
    xx = Flatten()(xx)
    output = Dense(784, activation='sigmoid')(xx)
    model = Model(input, output)
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95%
model.compile(loss='mse', optimizer='adam')
model.fit(x_train_noised, x_train, epochs=10, batch_size=128, validation_split=0.01)
img_decoded = model.predict(x_test_noised)

import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 7))
random_images = random.sample(range(img_decoded.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('before_noised', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('after_noised', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(img_decoded[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('output', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
