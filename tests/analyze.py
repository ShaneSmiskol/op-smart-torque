import json
import os
import ast
import matplotlib.pyplot as plt

os.chdir(os.getcwd())

with open('../data/smart_torque_data_old', 'r') as f:
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

plt.plot(driver_torque, label='driver torque')
plt.plot(eps_torque, label='eps torque')
plt.legend()

