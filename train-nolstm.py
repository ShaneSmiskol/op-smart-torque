import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, PReLU, ELU, LeakyReLU
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model
import os
import seaborn as sns
from normalizer import normX
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

os.chdir(os.getcwd())

print("Loading data...", flush=True)
with open("data/x_train", "rb") as f:
    x_train = pickle.load(f)
with open("data/y_train", "rb") as f:
    y_train = pickle.load(f)


model_inputs = ['angle_steers', 'delta_desired', 'angle_offset']  # , 'v_ego']

print("Normalizing data...", flush=True)
# x_train, scales = normX(x_train, model_inputs)

x_train = normX(x_train, model_inputs)
y_train = np.interp(y_train, [np.min(y_train), np.max(y_train)], [0, 1])
# y_train = np.interp(y_train, [-1, 1], [0, 1])


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.07)
print(x_train.shape)

opt = keras.optimizers.Adam(lr=0.0001)
# opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
# opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.01)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
# opt = keras.optimizers.Adagrad()
opt = 'adam'

# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()

a_function = "relu"

model = Sequential()
model.add(Dense(4 + 1, activation=a_function, input_shape=(x_train.shape[1:])))
model.add(Dropout(0.2))
model.add(Dense(128, activation=a_function))
model.add(Dropout(0.2))
model.add(Dense(256, activation=a_function))
model.add(Dropout(0.2))
model.add(Dense(128, activation=a_function))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer=opt, metrics=['mae'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=64,
          epochs=1000,
          validation_data=(x_test, y_test))


seq_len = 100
plt.clf()
rand_start = random.randint(0, len(x_test) - seq_len)
x = range(seq_len)
y = y_test[rand_start:rand_start+seq_len]
y2 = [model.predict(np.array([i]))[0][0] for i in x_test[rand_start:rand_start+seq_len]]
plt.title("random samples")
plt.plot(x, y, label='ground truth')
plt.plot(x, y2, label='prediction')
plt.legend()
plt.pause(0.01)
plt.show()



# preds = model.predict([[x_test]]).reshape(1, -1)
# diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test)]
#
# print("Test accuracy: {}".format(np.interp(sum(diffs) / len(diffs), [0, 1], [1, 0], ext=True)))

for i in range(20):
    c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict(np.array([x_test[c]]))[0][0]))
    print()
