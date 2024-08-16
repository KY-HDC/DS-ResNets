import numpy as np
import torch
import numba


def minkovski(input1, input2, p):
    tmp = torch.pow(torch.abs(input1 - input2), p)
    tmp = torch.sum(tmp, dim=-1)

    return torch.pow(tmp, (1 / p))


def euclidean(input1, input2):
    tmp = np.power(np.abs(input1 - input2), 2)
    tmp = np.sum(tmp, axis=-1)

    return np.power(tmp, (1 / 2))


def manhattan(input1, input2):
    res = np.sum(np.abs(input1 - input2), axis=-1)

    return res


def chebychev(input1, input2):
    res = np.max(np.abs(input1 - input2), axis=-1)

    return res

