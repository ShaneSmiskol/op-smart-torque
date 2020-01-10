import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, PReLU, ELU, LeakyReLU, CuDNNGRU, CuDNNLSTM, BatchNormalization
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
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
# sns.distplot(data_here)


def show_pred(test=True, max_y=None):
    seq_len = 100
    x = range(seq_len)
    plt.clf()
    if test:
        x_ = x_test
        y_ = y_test
    else:
        x_ = x_train
        y_ = y_train

    if max_y is None:
        max_y = 1000
    samples = np.where(abs(np.interp(y_, [0, 1], scales['driver_torque'])) <= max_y)[0]
    samples = np.random.choice(samples, size=seq_len)
    x_ = x_[samples]
    y_ = y_[samples]

    rand_start = random.randint(0, len(x_) - seq_len)
    y = y_[rand_start:rand_start + seq_len]
    y2 = [model.predict(np.array([i]))[0][0] for i in x_[rand_start:rand_start + seq_len]]

    plt.title("random samples")
    plt.plot(x, [np.interp(i, [0, 1], scales['driver_torque']) for i in y], label='ground truth')
    plt.plot(x, [np.interp(i, [0, 1], scales['driver_torque']) for i in y2], label='prediction')
    plt.legend()
    plt.pause(0.01)
    plt.show()


def show_pred_seq():
    for i in range(20):
        plt.clf()
        rand_start = random.randrange(len(x_test))
        delta = x_test[rand_start][:-3]
        plt.plot(len(delta), x_test[rand_start][-1], 'ro', label='angle steers')
        plt.plot(range(len(delta)), delta, label='delta desired')
        plt.plot(len(delta), model.predict(np.array([x_test[rand_start]]))[0][0], 'bo', label='prediction')
        plt.legend()
        plt.pause(0.01)
        input()


os.chdir(os.getcwd())

print("Loading data...", flush=True)
with open("model_data/x_train", "rb") as f:
    x_train = pickle.load(f)
with open("model_data/y_train", "rb") as f:
    y_train = pickle.load(f)
with open("model_data/scales", "rb") as f:
    scales = pickle.load(f)

# x_train = np.array([np.hstack(i) for i in x_train])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15)
print(x_train.shape)

opt = keras.optimizers.Adam(lr=0.001)
# opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.01)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
opt = keras.optimizers.Adagrad()
# opt = 'adam'

# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()

a_function = "relu"
dropout = 0.1

model = Sequential()
# model.add(CuDNNGRU(128, return_sequences=True, input_shape=x_train.shape[1:]))
# model.add(CuDNNGRU(64, return_sequences=False))
model.add(Dense(204, input_shape=(x_train.shape[1:])))
model.add(BatchNormalization(scale=False))
model.add(Activation('relu'))
# model.add(Dropout(0.4))

model.add(Dense(128))
model.add(BatchNormalization(scale=False))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization(scale=False))
model.add(Activation('relu'))

# model.add(Dropout(0.2))
# model.add(Dense(32, activation=a_function))
# model.add(Dropout(0.25))
# model.add(Dense(32, activation=a_function))
# model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(loss='mse', optimizer=opt, metrics=['mae'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=2048,
          epochs=1000,
          validation_data=(x_test, y_test))


# preds = model.predict([Ztewrmin[x_test]]).reshape(1, -1)
# diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test)]
#
# print("Test accuracy: {}".format(np.interp(sum(diffs) / len(diffs), [0, 1], [1, 0], ext=True)))

for i in range(20):
    c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict(np.array([x_test[c]]))[0][0]))
    print()


def save_model(name='model'):
    model.save('models/h5_models/{}.h5'.format(name))
