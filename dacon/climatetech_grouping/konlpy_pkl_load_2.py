import pickle
import tensorflow as tf


with open('./_save/_npy/dacon/climatetech_grouping/inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)

x = train_inputs
x_pred = test_inputs
y = labels

x0 = tf.convert_to_tensor(x[0].toarray())
