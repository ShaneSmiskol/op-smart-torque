import numpy as np


def normX(data, model_inputs):
    data = np.array([[sample[key] for key in model_inputs] for sample in data])
    v1 = np.take(data, indices=0, axis=1)
    v2 = np.take(data, indices=1, axis=1)
    v3 = np.take(data, indices=2, axis=1)
    # v4 = np.take(data, indices=3, axis=1)

    v1 = np.interp(v1, [np.min(v1), np.max(v1)], [0, 1])
    v2 = np.interp(v2, [np.min(v2), np.max(v2)], [0, 1])
    v3 = np.interp(v3, [np.min(v3), np.max(v3)], [0, 1])
    # v4 = np.interp(v4, [np.min(v4), np.max(v4)], [0, 1])

    data = np.vstack((v1, v2, v3)).T
    return data
