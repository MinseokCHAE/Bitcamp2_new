import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ======== 1 ========
# inputs = ak.ImageInput()
# outputs = ak.ImageBlock(
#     block_type='resnet',
#     normalize=True,
#     augment=False,
# )(inputs)
# outputs = ak.ClassificationHead()(outputs)

# model = ak.AutoModel(inputs=inputs, outputs=outputs, overwrite=True, max_trials=1)


# ======== 2 ========
# model = ak.ImageClassifier(
#     overwrite=True,
#     max_trials=2
# )


# ======== 3 ========
model = ak.StructuredDataClassifier(overwrite=True, max_trials=1) # classifier <=> regressor


model.fit(x_train, y_train, epochs=5,)

prediction = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)
print(evaluation)

model = model.export_model()
model.summary()

