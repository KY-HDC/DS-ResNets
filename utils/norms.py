import numpy as np
import random


def norm(data):
    return (data-min(data))/(max(data) - min(data))


def softmax(d):
    tmp = np.exp(d)
    # print(np.sum(tmp, axis=1).shape)
    tmp = tmp/np.sum(tmp, axis=-1).reshape(-1, d.shape[-2], 1)

    return tmp


def sigmoid(d):
    tmp = np.exp(d)
    tmp = tmp/(1 + tmp)
    return tmp


def harm_mean(d, axis=None):
    if axis is None:
        n = 1
        for di in d.shape:
            n *= di
    else:
        n = d.shape[axis]
    out = np.sum(np.power(d, -1), axis=axis)
    return np.power(out/n, -1)
