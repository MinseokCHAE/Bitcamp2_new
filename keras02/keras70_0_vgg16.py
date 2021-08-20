import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16, VGG19

# model = VGG16()
# model = VGG19()
# model.summary()
'''
Model: "vgg16"
_________________________________________________________________      
Layer (type)                 Output Shape              Param #
=================================================================      
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________      
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________      
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________      
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________      
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________      
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________      
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________      
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________      
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________      
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________      
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________      
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________      
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________      
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________      
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________      
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________      
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________      
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________      
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________      
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________      
fc1 (Dense)                  (None, 4096)              102764544       
_________________________________________________________________      
fc2 (Dense)                  (None, 4096)              16781312        
_________________________________________________________________      
predictions (Dense)          (None, 1000)              4097000
=================================================================      
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
'''
model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
model.trainable=False
# model.summary()
# print(len(model.weights)) # 26
# print(len(model.trainable_weights)) # 26

input = Input((100,100,3))
xx = VGG16(weights='imagenet', include_top=False)(input)
xx = Flatten()(xx)
xx = Dense(10)(xx)
output = Dense(1)(xx)
model = Model(inputs=input, outputs=output)

# model.summary()
# print(len(model.weights)) # 30 -> Dense 2개층 추가 (+4)
# print(len(model.trainable_weights)) # 30

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer trainable'])

# print(results)
'''
                                                                                                                                        Layer Type Layer Name  Layer trainable
0  <tensorflow.python.keras.engine.input_layer.InputLayer object at 0x00000125D07FEE50>  input_2    True
1  <tensorflow.python.keras.engine.functional.Functional object at 0x00000125D956A220>   vgg16      True
2  <tensorflow.python.keras.layers.core.Flatten object at 0x00000125D94C6F70>            flatten    True
3  <tensorflow.python.keras.layers.core.Dense object at 0x00000125D956A880>              dense      True
4  <tensorflow.python.keras.layers.core.Dense object at 0x00000125D12EE7F0>              dense_1    True
'''
