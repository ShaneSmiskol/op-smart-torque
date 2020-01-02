import ast
import os
import time
import json
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from tokenizer import tokenize

os.chdir(os.getcwd())
with open('data/smart_torque_data', 'r') as f:
    data = f.read().split('\n')
keys, data = ast.literal_eval(data[0]), data[1:]
data = [dict(zip(keys, i)) for i in map(json.loads, data)]

model_inputs = ['v_ego', 'angle_steers', 'delta_desired', 'angle_offset', 'driver_torque', 'time']
data = np.array([[math.degrees(sample[key]) if key == 'delta_desired' else sample[key] for key in model_inputs] for sample in data])

print('Normalizing data...', flush=True)
d_t = data.T
scales = {}
for idx, inp in enumerate(model_inputs):
    if inp != 'time':
        scales[inp] = [np.amin(d_t[idx]), np.amax(d_t[idx])]
        d_t[idx] = np.interp(d_t[idx], scales[inp], [0, 1])

data = [dict(zip(model_inputs, i)) for i in d_t.T]

print('Splitting data by time...', flush=True)
data_split = [[]]
counter = 0
model_inputs.remove('time')
for idx, line in enumerate(data):
    if idx > 0:
        time_diff = line['time'] - data[idx - 1]['time']
        if abs(time_diff) > 0.05:
            counter += 1
            data_split.append([])
    data_split[counter].append([line[inp] for inp in model_inputs])  # removes time


avg_time = 0.01  # openpilot runs longcontrol at 100hz, so this makes sense

seq_time = 2.5
seq_len = round(seq_time / avg_time) + 1

print('Tokenizing data...', flush=True)
data_sequences = []
for i in data_split:
    data_sequences += tokenize(i, seq_len)

print('Formatting data for model...', flush=True)
# todo: speed up with numpy
# all but last, remove driver torque from each sample \/
x_train = np.array([[[point for idx, point in enumerate(sample) if idx != model_inputs.index('driver_torque')] for sample in seq[:-1]] for seq in data_sequences])
y_train = np.array([seq[-1][model_inputs.index('driver_torque')] for seq in data_sequences])  # last sample, but driver torque

# idx = 5488
# y = [i[model_inputs.index('v_ego')] for i in data_sequences[idx]]
# y2 = [i[model_inputs.index('angle_steers')] for i in data_sequences[idx]]
# y3 = [i[model_inputs.index('delta_desired')] * 17.8 for i in data_sequences[idx]]
# y4 = [i[model_inputs.index('driver_torque')] * .3 for i in data_sequences[idx]]
# # plt.plot(range(len(data_sequences[idx])), y, label='v_ego (mph)')
# plt.plot(range(len(data_sequences[idx])), y2, label='angle steers')
# plt.plot(range(len(data_sequences[idx])), y3, label='delta desired')
# plt.plot(range(len(data_sequences[idx])), y4, label='driver torque')
#
# plt.legend()

# y = [i['v_ego'] for i in data_sequences[0]]
# y2 = [i['angle_offset'] for i in data]
# y3 = [math.degrees(i['delta_desired'] * 18.24) for i in data_sequences[0]]
# y4 = [i['angle_steers'] for i in data_sequences[0]]
# y5 = [i['driver_torque'] for i in data_sequences[0]]
# y6 = [i['eps_torque'] for i in data]
# y7 = [i['time'] for i in data]

# plt.plot(range(len(data)), y, label='v_ego (mph)')
# plt.plot(range(len(data)), y2, label='angle_offset')
# plt.plot(range(len(data_sequences[0])), y3, label='delta desired')
# plt.plot(range(len(data_sequences[0])), y4, label='angle steers')
# plt.plot(range(len(data_sequences[0])), y5, label='driver torque')
# plt.plot(range(len(data)), y6, label='eps torque')
# plt.plot(range(len(data)), y6, label='eps torque')
# plt.plot(range(len(data)), y7, label='time')
# plt.legend()
# plt.show()

# x_keys = ['angle_steers', 'delta_desired', 'angle_offset', 'v_ego']
# x_train = [{key: i[key] for key in i if key in x_keys} for i in data]
# y_train = [i['driver_torque'] for i in data]

with open('data/x_train', 'wb') as f:
    pickle.dump(x_train, f)
with open('data/y_train', 'wb') as f:
    pickle.dump(y_train, f)
