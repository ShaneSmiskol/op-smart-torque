import json
import os
import ast
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from utils.st_helper import STHelper
from utils.BASEDIR import BASEDIR


os.chdir(BASEDIR)
helper = STHelper()
helper.init()
model = load_model('models/test.h5')

with open('data/smart_torque_data', 'r') as f:
  raw = f.read().split('\n')

keys = ast.literal_eval(raw[0])
data = []
print(keys)

for line in raw[1:]:
  if line != '':
    line = ast.literal_eval(line)
    data.append(dict(zip(keys, line)))



delta_desired = helper.norm(list(map(lambda x: x['delta_desired'], data)), 'delta_desired')
angle_steers = helper.norm(list(map(lambda x: x['angle_steers'], data)), 'angle_steers')
v_ego = helper.norm(list(map(lambda x: x['v_ego'], data)), 'v_ego')
eps_torque = helper.norm(list(map(lambda x: x['eps_torque'], data)), 'eps_torque')

x_train = []
y_train = []

x_temp = []
for idx, sample in enumerate(zip(delta_desired, angle_steers, v_ego, eps_torque)):
  x_temp.append(sample[1:])
  if len(x_temp) == 200:
    x_train.append(x_temp)
    y_train.append(sample[-1])
    x_temp = []

x_train = np.array(x_train)
y_train = np.array(y_train)


def show_pred():
  plt.clf()
  seq_idx = np.random.choice(range(len(x_train)))
  x_ = x_train[seq_idx]
  y_ = helper.unnorm(y_train[seq_idx], 'eps_torque')
  pred = helper.unnorm(model.predict_on_batch([[x_]])[0][0], 'eps_torque')

  delta_desired = helper.unnorm([np.take(ts, axis=0, indices=0) for ts in x_], 'delta_desired')
  angle_steers = helper.unnorm([np.take(ts, axis=0, indices=1) for ts in x_], 'angle_steers')

  plt.plot(delta_desired, label='delta desired')
  plt.plot(angle_steers, label='angle steers')
  plt.plot(len(x_), pred, 'go', label='prediction')
  plt.plot(len(x_), y_, 'bo', label='ground')
  # plt.plot(eps_torque, label='eps torque')
  plt.legend()

# show_pred()
