import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

model.fit(x_train, y_train, epochs=5,)

prediction = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)
print(evaluation)

model = model.export_model()
model.summary()

