import numpy as np
import os


def task_old1(n_class, d_name, path="Result"):
    path_ = os.path.join(path, "task1_old")
    for i in range(n_class):
        if i == 0:
            data = np.load(f"{path_}/{d_name}_Class_{i}_dist_all.npy", allow_pickle=True)
        else:
            tmp = np.load(f"{path_}/{d_name}_Class_{i}_dist_all.npy", allow_pickle=True)
            data = np.concatenate((data, tmp), 0)
    return data


def task_new1(n_class, d_name):
    pass


def task2_read(d_name, model):
    a = np.load(f"task2/{d_name}_{model}_Targets.npy", allow_pickle=True)
    b = np.load(f"task2/{d_name}_{model}_Series.npy", allow_pickle=True)
    c = np.load(f"task2/{d_name}_{model}_SeqInfo.npy", allow_pickle=True)
    d = np.load(f"task2/{d_name}_{model}_MaxList.npy", allow_pickle=True)

    return [a, b, c, d]
