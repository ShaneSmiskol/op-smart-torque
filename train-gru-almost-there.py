import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, Flatten, BatchNormalization, SimpleRNN, GRU, BatchNormalization
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
from utils.tokenizer import split_list, tokenize
import ast
import matplotlib.gridspec as gridspec
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

support = STHelper()
support.init()
os.chdir(BASEDIR)


with open('data/smart_torque_data', 'r') as f:
  raw = f.read().split('\n')

keys = ast.literal_eval(raw[0])
data = []
print(keys)

for line in raw[1:]:
  if line != '':
    line = ast.literal_eval(line)
    line = {key: dat for key, dat in zip(keys, line) if key not in support.ignored_keys}
    del line['time']
    for key in line:
      line[key] = support.norm(line[key], key)
    x_ = [line[key] for key in support.model_inputs]
    y_ = [line[key] for key in support.model_outputs]
    data.append([x_, y_])
x_, y_ = zip(*data)
future_timesteps = 50
x_, y_ = tokenize(x_, (support.seq_len - 1) * future_timesteps), tokenize(y_, (support.seq_len - 1) * future_timesteps)
x_, y_ = np.array(x_), np.array(y_)


fig, ax = plt.subplots(2, 2)
ax = ax.flatten()

def show_pred_new(epoch, sample_idx=None):
  if sample_idx is None:
    sample_idx = random.randrange(len(x_train))
  seq_x = x_[sample_idx]
  seq_y = y_[sample_idx]
  # seq_x_to_pred = seq_x[:support.seq_len - support.y_future]
  seq_x_to_pred = np.array(np.split(seq_x, future_timesteps, axis=0))
  # seq_y_ground = seq_y[-support.seq_len - support.y_future:]
  seq_y_ground = np.array(np.split(seq_y, future_timesteps, axis=0))

  input_labels = support.model_inputs
  output_labels = support.model_outputs
  _scales = [scales[i] for i in support.model_inputs + support.model_outputs]
  colors = ['r', 'g', 'b', 'm', 'c', 'r']

  # print(seq_x[0].shape)
  for idx, (lbl, clr) in enumerate(zip(input_labels, colors)):
    ax[idx].cla()
    lst = support.unnorm(seq_x_to_pred[0].take(axis=1, indices=idx), lbl)
    if lbl == 'delta_desired':
      lst = np.degrees(lst * 17.8)
    ax[idx].plot(range(1, len(seq_x_to_pred[0]) + 1), lst, clr, label=lbl)
  ax[-1].cla()


  to_plot = np.concatenate(seq_y_ground[1:])
  # print(len(to_plot.take(axis=1, indices=3)))
  ax[-1].plot(range(len(seq_x_to_pred[0]), len(seq_x_to_pred[0]) + len(to_plot)), to_plot.take(axis=1, indices=3))

  pred = model.predict(np.array(seq_x_to_pred)).T
  # print(pred.shape)

  for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
    # lst = support.unnorm(pred[idx::len(output_labels)], lbl)
    lst = support.unnorm(pred[idx], lbl)
    if lbl == 'delta_desired':
      lst = np.degrees(lst * 17.8)
    # ax[idx].plot(range((len(x) // len(input_labels)), len(x) // len(input_labels) + len(y) // len(output_labels)), lst, label=lbl + ' pred')
    ax[idx].plot(range(len(seq_x_to_pred[0]), len(seq_x_to_pred[0]) + len(lst)), lst, label=lbl + ' pred')
    ax[idx].legend(loc='upper left')

  fig.show()
  plt.pause(0.01)
  plt.savefig('models/model_imgs/{}'.format(epoch))


# def show_pred_new(epoch, sample_idx=None):
#   if sample_idx is None:
#     sample_idx = random.randrange(len(x_train))
#   x = x_train[sample_idx]
#   y = y_train[sample_idx]
#
#   input_labels = support.model_inputs
#   output_labels = support.model_outputs
#   _scales = [scales[i] for i in support.model_inputs + support.model_outputs]
#   colors = ['r', 'g', 'b', 'm', 'c', 'r']
#
#   for idx, (lbl, clr) in enumerate(zip(input_labels, colors)):
#     ax[idx].cla()
#     lst = support.unnorm(x[idx::len(input_labels)], lbl)
#     if lbl == 'delta_desired':
#       lst = np.degrees(lst * 17.8)
#     ax[idx].plot(range(1, (len(x) // len(input_labels)) + 1), lst, clr)
#   ax[-1].cla()
#
#   for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
#     lst = support.unnorm(y[idx::len(output_labels)], lbl)
#     if lbl == 'delta_desired':
#       lst = np.degrees(lst * 17.8)
#     ax[idx].plot(range((len(x) // len(input_labels)), len(x) // len(input_labels) + len(y) // len(output_labels)), lst, clr, label=lbl)
#
#   pred = model.predict(np.array([x]))[0]
#   for idx, (lbl, clr) in enumerate(zip(output_labels, colors)):
#     lst = support.unnorm(pred[idx::len(output_labels)], lbl)
#     if lbl == 'delta_desired':
#       lst = np.degrees(lst * 17.8)
#     ax[idx].plot(range((len(x) // len(input_labels)), len(x) // len(input_labels) + len(y) // len(output_labels)), lst, label=lbl + ' pred')
#     ax[idx].legend(loc='upper left')
#
#   fig.show()
#   plt.pause(0.01)
#   plt.savefig('models/model_imgs/{}'.format(epoch))


class ShowPredictions(tf.keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self.sample_idx = random.randrange(len(x_test))

  def on_epoch_end(self, epoch, logs=None):
    if True or (epoch + 4) % 3 == 1:
      show_pred_new(epoch, self.sample_idx)


print("Loading data...", flush=True)
with open("model_data/x_train", "rb") as f:
  x_train = pickle.load(f)
with open("model_data/y_train", "rb") as f:
  y_train = pickle.load(f)
with open("model_data/scales", "rb") as f:
  scales = pickle.load(f)

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
opt = keras.optimizers.Adam(amsgrad=False)
# opt = 'adam'


a_function = "relu"
dropout = 0.1

model = Sequential()
model.add(GRU(64, return_sequences=True, input_shape=x_train.shape[1:]))
model.add(GRU(32, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(32, activation=a_function, input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
#
# model.add(Dense(16, activation=a_function))
# model.add(BatchNormalization())
#
# model.add(Dense(16, activation=a_function))
# model.add(BatchNormalization())

model.add(Dense(y_train.shape[1]))

model.compile(loss='mae', optimizer=opt, metrics=['mse'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
callbacks = [ShowPredictions()]
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=128,
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
