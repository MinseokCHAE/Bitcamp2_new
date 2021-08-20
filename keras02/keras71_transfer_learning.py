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
# print(len(model.weights))   # 816
# print(len(model.trainable_weights)) # 514
# Total params = 60,380,648

model = ResNet50()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 320
# print(len(model.trainable_weights)) # 214
# Total params = 25,636,712

model = ResNet50V2()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 272
# print(len(model.trainable_weights)) # 174
# Total params = 25,613,800

model = InceptionV3()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 378
# print(len(model.trainable_weights)) # 190
# Total params = 23,851,784

model = InceptionResNetV2()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 898
# print(len(model.trainable_weights)) # 490
# Total params = 55,873,736

model = MobileNet()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 137
# print(len(model.trainable_weights)) # 83
# Total params = 4,253,864

model = MobileNetV2()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 262
# print(len(model.trainable_weights)) # 158
# Total params =  3,538,984

model = MobileNetV3Large()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 266
# print(len(model.trainable_weights)) # 174
# Total params = 5,507,432

model = MobileNetV3Small()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 210
# print(len(model.trainable_weights)) # 142
# Total params = 2,554,968

model = DenseNet121()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 606
# print(len(model.trainable_weights)) # 364
# Total params = 8,062,504

model = DenseNet169()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 846
# print(len(model.trainable_weights)) # 508
# Total params = 14,307,880

model = DenseNet201()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 1006
# print(len(model.trainable_weights)) # 604
# Total params = 20,242,984

model = NASNetLarge()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 1546
# print(len(model.trainable_weights)) # 1018
# Total params = 88,949,818

model = NASNetMobile()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 1126
# print(len(model.trainable_weights)) # 742
# Total params = 5,326,716

model = EfficientNetB0()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 314
# print(len(model.trainable_weights)) # 213
# Total params = 5,330,571

model = EfficientNetB1()
# model.trainable=False
# model.summary()
# print(len(model.weights))   # 442
# print(len(model.trainable_weights)) # 301
# Total params = 7,856,239

model = EfficientNetB7()
# model.trainable=False
model.summary()
print(len(model.weights))   # 1040
print(len(model.trainable_weights)) # 711
# Total params = 66,658,687
