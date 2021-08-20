from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7


model = Xception()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 236
# print(len(model.trainable_weights)) # 236
# Total params = 22,910,480

model = ResNet101()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 626
# print(len(model.trainable_weights)) # 418
# Total params = 44,707,176

model = ResNet101V2()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 544
# print(len(model.trainable_weights)) # 344
# Total params = 44,675,560

model = ResNet152()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 932
# print(len(model.trainable_weights)) # 622
# Total params = 60,419,944

model = ResNet152V2()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 
# print(len(model.trainable_weights)) # 
# Total params = 

model = ResNet50()
# model.trainable=False
model.summary()
print(len(model.weights))   # 544
print(len(model.trainable_weights)) # 344
# Total params = 44,675,560
