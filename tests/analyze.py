import json
import os
import ast
import matplotlib.pyplot as plt
import math
import numpy as np

os.chdir(os.getcwd())

with open('../data/smart_torque_data', 'r') as f:
  raw = f.read().split('\n')

keys = ast.literal_eval(raw[0])
data = []
print(keys)

for line in raw[1:]:
  if line != '':
    line = ast.literal_eval(line)
    data.append(dict(zip(keys, line)))

driver_torque = list(map(lambda x: x['driver_torque'], data))
eps_torque = list(map(lambda x: x['eps_torque'], data))
delta_desired = np.array(list(map(lambda x: x['delta_desired'], data)))
rate_desired = np.array(list(map(lambda x: x['rate_desired'], data)))
delta_desired_degrees = np.array(list(map(lambda x: math.degrees(x['delta_desired']), data)))

# plt.plot(driver_torque, label='driver torque')
plt.plot(np.interp(eps_torque, [-500, 500], [-1, 1]), label='eps torque')
plt.plot(delta_desired * 57, label='delta_desired')
# plt.plot(rate_desired * 57, label='rate_desired')
# plt.plot(delta_desired_degrees, label='delta_desired degrees')
plt.legend()

