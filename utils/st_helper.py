import ast
import os
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from utils.tokenizer import tokenize, split_list
from utils.BASEDIR import BASEDIR

os.chdir(BASEDIR)


class STHelper:
    def __init__(self, process_data=False):
        self.driving_data = []
        self.scales = {}
        self.keys = None

        self.model_inputs = ['delta_desired', 'rate_desired', 'angle_steers', 'angle_steers_rate', 'v_ego']
        self.model_outputs = ['eps_torque']
        self.save_path = 'model_data'

        self.needs_to_be_degrees = ['delta_desired', 'rate_desired']
        self.sR = 17.8

        self.scale_to = [0, 1]
        self.avg_time = 0.01  # openpilot runs latcontrol at 100hz, so this makes sense
        self.y_future = round(0.0 / self.avg_time)  # how far into the future we want to be predicting, in seconds (0.01 is next sample)
        self.seq_len = round(0.5 / self.avg_time) + self.y_future  # how many seconds should the model see at any one time

        if process_data:
            self.start()
        else:
            self.load_scales()

    def start(self):
        self.setup_dirs()
        self.load_data()
        self.normalize_data()
        self.split_data()
        self.tokenize_data()
        self.format_data()
        self.finalize()

    def load_scales(self):
        if not self.setup_dirs:
            raise Exception('Error, need to run process.py first!')
        with open('model_data/scales', 'rb') as f:
            self.scales = pickle.loads(f.read())

    def setup_dirs(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            return True
        return False

    def load_data(self):
        print('Loading data...', flush=True)
        _data = []
        for data_file in os.listdir('data/'):
            if 'smart_torque_data' in data_file:
                print('Loading: {}'.format(data_file))
                with open('data/{}'.format(data_file), 'r') as f:
                    data = f.read().split('\n')

                keys, data = ast.literal_eval(data[0]), data[1:]
                if self.keys is not None:
                    if keys != self.keys:
                        raise Exception('Keys in files do not match each other!')

                self.keys = keys
                for line in data:
                    try:
                        _data.append(dict(zip(self.keys, ast.literal_eval(line))))
                    except:
                        print('Error parsing line: `{}`'.format(line))

        for sample in _data:
            for key in sample:
                if key in self.needs_to_be_degrees:
                    sample[key] = math.degrees(sample[key] * self.sR)
            self.driving_data.append(sample)

    def normalize_data(self):
        print('Normalizing data...', flush=True)
        data = np.array([[sample[key] for key in self.keys] for sample in self.driving_data])
        d_t = data.T
        for idx, inp in enumerate(self.keys):
            if inp != 'time':
                self.scales[inp] = [np.amin(d_t[idx]), np.amax(d_t[idx])]
                d_t[idx] = self.norm(d_t[idx], inp)

        data = [dict(zip(self.keys, i)) for i in d_t.T]
        times = [i['time'] for i in self.driving_data]
        assert len(data) == len(times) == len(self.driving_data), 'Length of data not equal'

        for idx, (t, sample) in enumerate(zip(times, data)):
            sample['time'] = t
            self.driving_data[idx] = sample

    def split_data(self):
        print('Splitting data by time...', flush=True)
        data_split = [[]]
        counter = 0
        for idx, line in enumerate(self.driving_data):
            if idx > 0:
                time_diff = line['time'] - self.driving_data[idx - 1]['time']
                if abs(time_diff) > 0.05:  # account for lag when writing data (otherwise we would use 0.01)
                    counter += 1
                    data_split.append([])

            data_split[counter].append(line)
        self.driving_data = data_split

    def tokenize_data(self):
        print('Tokenizing data...', flush=True)
        data_sequences = []
        for idx, seq in enumerate(self.driving_data):
            # todo: experiment with splitting list instead. lot less training data, but possibly less redundant data
            data_sequences += tokenize(seq, self.seq_len)
        self.driving_data = data_sequences

    def format_data(self):
        print('Formatting data for model...', flush=True)
        self.x_train = []
        self.y_train = []
        print(self.keys)
        for seq in self.driving_data:
            if abs(np.interp(seq[-1]['angle_steers'], [0, 1], self.scales['angle_steers'])) > 80:
                continue

            if self.y_future != 0:
                seq = seq[:-self.y_future]

            seq_in = [[sample[des_key] for des_key in self.model_inputs] for sample in seq]
            seq_out = [seq[-1][des_key] for des_key in self.model_outputs]

            self.x_train.append(seq_in)
            self.y_train.append(seq_out)

    def finalize(self):
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        print('Dumping data...', flush=True)
        with open('model_data/x_train', 'wb') as f:
            pickle.dump(self.x_train, f)
        with open('model_data/y_train', 'wb') as f:
            pickle.dump(self.y_train, f)
        with open('model_data/scales', 'wb') as f:
            pickle.dump(self.scales, f)

    def unnorm(self, x, name):
        return np.interp(x, self.scale_to, self.scales[name])

    def norm(self, x, name):
        return np.interp(x, self.scales[name], self.scale_to)


if __name__ == '__main__':
    helper = STHelper(process_data=True)
