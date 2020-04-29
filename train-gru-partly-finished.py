import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, Flatten, BatchNormalization, SimpleRNN, GRU, TimeDistributed
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os
import seaborn as sns
from utils.st_helper import STHelper
from utils.BASEDIR import BASEDIR
import matplotlib.gridspec as gridspec
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

support = STHelper()
support.init()
os.chdir(BASEDIR)


fig, ax = plt.subplots(3)


def show_pred_new(sample_idx=None):
  if sample_idx is None:
    sample_idx = random.randrange(len(x_test))
  x = x_test[sample_idx]
  y = y_test[sample_idx]

  input_labels = support.model_inputs
  output_labels = support.model_outputs
  colors = ['r', 'g', 'b']

  for idx, (lbl, clr) in enumerate(zip(input_labels, colors)):
    ax[idx].cla()
    # data = x[idx::3]
    data = np.take(x, axis=1, indices=idx)
    # ax[idx].plot(range(1, (len(x) // 3) + 1), data, clr)
    ax[idx].plot(range(1, len(x) + 1), data, clr)

  for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
    # data = y[idx::3]
    data = np.take(y, axis=1, indices=idx)
    # ax[idx].plot(range((len(x) // 3), ((len(x) + len(y)) // 3)), data, clr, label=lbl)
    ax[idx].plot(range(len(x), len(x) + len(y)), data, clr, label=lbl)

  pred = model.predict(np.array([x]))[0]
  for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
    # data = pred[idx::3]
    data = np.take(pred, axis=1, indices=idx)
    # ax[idx].plot(range((len(x) // 3), ((len(x) + len(y)) // 3)), data, label=lbl + ' pred')
    ax[idx].plot(range(len(x), len(x) + len(y)), data, label=lbl + ' pred')
    ax[idx].legend(loc='upper left')

  fig.show()
  plt.pause(0.01)


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.sample_idx = random.randrange(len(x_test))

  def on_epoch_end(self, epoch, logs=None):
    if True or epoch % 0 == 1:
      show_pred_new(self.sample_idx)


print("Loading data...", flush=True)
with open("model_data/x_train", "rb") as f:
  x_train = pickle.load(f)
with open("model_data/y_train", "rb") as f:
  y_train = pickle.load(f)
with open("model_data/scales", "rb") as f:
  scales = pickle.load(f)

x_train = x_train[:19000]
y_train = y_train[:19000]

# x_train = np.array([i.flatten() for i in x_train])
# y_train = np.array([i.flatten() for i in y_train])
# y_train = helper.unnorm(y_train, 'eps_torque')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)
print(y_train.shape)

# opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
# opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.01)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
# opt = keras.optimizers.Adagrad()
# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()
opt = keras.optimizers.Adam(lr=0.0006)
# opt = 'adam'


a_function = "relu"
dropout = 0.1
batch_size = 152

model = Sequential()
model.add(GRU(16, return_sequences=False, stateful=True, batch_input_shape=(batch_size, 200, 3)))
# model.add(GRU(8, stateful=True, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation=a_function, input_shape=x_train.shape[1:]))
# model.add(Dropout(0.1))
model.add(Dense(32, activation=a_function))
# model.add(Dense(64, activation=a_function))
# model.add(Dense(64, activation=a_function))
# model.add(Dense(y_train.shape[1]))
# model.add(TimeDistributed(Dense(3)))
model.add(Dense(3))

model.compile(loss='mse', optimizer=opt, metrics=['mae'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
show_predictions = ShowPredictions()
callbacks = [show_predictions]
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=batch_size,
          epochs=1000,
          validation_data=(x_test, y_test),
          callbacks=callbacks)


preds = model.predict(x_test).reshape(1, -1)
diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test[0])]

print("Test accuracy: {}%".format(round(np.interp(sum(diffs) / len(diffs), [0, 1], [1, 0]) * 100, 4)))

for i in range(20):
  c = random.randint(0, len(x_test))
  print('Ground truth: {}'.format(support.unnorm(y_test[c][0], 'eps_torque')))
  print('Prediction: {}'.format(support.unnorm(model.predict(np.array([x_test[c]]))[0][0], 'eps_torque')))
  print()


def save_model(name='model'):
  model.save('models/h5_models/{}.h5'.format(name))
