import ast
import os
import time
import json
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from tokenizer import tokenize, split_list
import random

os.chdir(os.getcwd())

driving_data = []
for data_file in os.listdir('data/'):
    if 'smart_torque_data' in data_file:
        print(data_file)
        with open('data/{}'.format(data_file), 'r') as f:
            data = f.read().split('\n')
        keys, data = ast.literal_eval(data[0]), data[1:]
        driving_data += [dict(zip(keys, i)) for i in map(json.loads, data)]

model_inputs = ['v_ego', 'angle_steers', 'delta_desired', 'angle_offset', 'driver_torque', 'time']
data = np.array([[math.degrees(sample[key]) if key == 'delta_desired' else sample[key] for key in model_inputs] for sample in driving_data])  # todo: see if not converting to degrees will make the model learn quicker

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
        if abs(time_diff) > 0.035:  # account for lag when writing data (otherwise we would use 0.01)
            counter += 1
            data_split.append([])
    data_split[counter].append([line[inp] for inp in model_inputs])  # removes time


avg_time = 0.01  # openpilot runs longcontrol at 100hz, so this makes sense

y_time_in_future = 0.00  # how far into the future we want to be predicting, in seconds (0.01 is next sample)
y_future = round(y_time_in_future / avg_time)

seq_time = 1.0
seq_len = round(seq_time / avg_time) + y_future

print('Tokenizing data...', flush=True)
data_sequences = []
for i in data_split:
    data_sequences += tokenize(i, seq_len)  # todo: experiment with splitting list instead. lot less training data, but possibly less redundant data

print('Formatting data for model...', flush=True)
# todo: speed up with numpy
# all but last, remove driver torque from each sample \/
# x_train = np.array([[[point for idx, point in enumerate(sample) if idx != model_inputs.index('driver_torque')] for sample in seq[:-1]] for seq in data_sequences])

x_train = []
y_train = []
print(model_inputs)
for seq in data_sequences:
    if abs(np.interp(seq[-1][model_inputs.index('driver_torque')], [0, 1], scales['driver_torque'])) > 100:
        continue
    y_train.append(seq[-1][model_inputs.index('driver_torque')])
    if y_future != 0:
        seq = seq[:-y_future]

    v_ego = seq[-1][model_inputs.index('v_ego')]
    angle_offset = seq[-1][model_inputs.index('angle_offset')]
    angle_steers = seq[-1][model_inputs.index('angle_steers')]

    # seq = [[dat for idx, dat in enumerate(sample) if idx not in [model_inputs.index('v_ego'), model_inputs.index('angle_offset')]] for sample in seq]
    # seq = [[sample[model_inputs.index('delta_desired')], sample[model_inputs.index('angle_steers')]] for sample in seq]
    seq = [sample[model_inputs.index('delta_desired')] for sample in seq]

    # seq = [item for sublist in seq for item in sublist]
    seq += [v_ego, angle_offset, angle_steers]
    x_train.append(seq)

x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.array([seq[:-y_future] for seq in data_sequences])  # all but last x samples
# y_train = np.array([seq[-1][model_inputs.index('driver_torque')] for seq in data_sequences])  # last sample, but driver torque


def show_data():
    for i in range(20):
        idx = random.randrange(len(x_train))
        plt.clf()
        # y = [i[model_inputs.index('v_ego')] for i in x_train[idx]]
        y2 = [i[0] for i in x_train[idx][:-1]]
        y3 = [i[1] for i in x_train[idx][:-1]]
        y4 = [i[2] * .3 for i in x_train[idx][:-1]]
        # plt.plot(range(len(data_sequences[idx])), y, label='v_ego (mph)')
        plt.plot(range(len(y2)), y2, label='angle steers')
        plt.plot(range(len(y3)), y3, label='delta desired')
        plt.plot(len(x_train[idx]), y_train[idx] * .3, 'bo', label='driver torque future')
        plt.plot(range(len(y4)), y4, label='driver torque')

        plt.legend()
        plt.show()
        plt.pause(0.01)
        input()

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


print('Dumping data...', flush=True)
with open('model_data/x_train', 'wb') as f:
    pickle.dump(x_train, f)
with open('model_data/y_train', 'wb') as f:
    pickle.dump(y_train, f)
with open('model_data/scales', 'wb') as f:
    pickle.dump(scales, f)
