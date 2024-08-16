import numpy as np
import os
from utils.task import DistanceMeasure, task_2
from utils.builders import seq_builder, load_files
from utils.read import task_old1, task2_read
import sys
from utils.norms import softmax, sigmoid, norm, init_random
from utils.distance import minkovski
import torch


if __name__ == '__main__':

    wpath = "weightpath"
    out_dir = "dir_name"
    data_name = "data_name"
    f_path = wpath
    l_path = os.path.join(our_dir, data_name)
    n_blocks = 16

    for b in range(n_blocks):
        pth = f"{f_path}/{data_name}/{data_name}_block{b}.pt"
        x = torch.load(pth)
        if b == 0:
            hold = x.detach().numpy().reshape(x.shape[0], 1, x.shape[1])
            y = torch.load(f"{l_path}/test/{data_name}_label.pt")
        else:
            hold = np.concatenate((hold, x.detach().numpy().reshape(x.shape[0], 1, x.shape[1])), 1)
    
    feat_dict = DistanceMeasure(hold/1000, y, norm="softmax")
    feat_dict.task_1(data_name, "Densenet")
    sys.exit()
    seqs = seq_builder(hold, data_name, "Densenet", 4)
    best_stack, best_stack_mean = task_2(seqs, data_name)
    sys.exit()
    dim.class_wise(data_name, result_)
    old_task1 = task_old1(n_class, data_name, result_)

    f_name = f"Task_1_{data_name}"
    if not os.path.exists(f_name+'_9.npy'):
        dim.task_1(f_name)
    for key in np.arange(10):
        dist_vals = np.load(f'{f_name}_{key}.npy', allow_pickle=True)
        print(dist_vals.shape)
        print(np.max(dist_vals, 3).shape)
    
    if os.path.exists(f"Targets_{data_name}.npy"):
        seqs = task2_read(data_name)
    else:
        seqs = seq_builder(feats, labels)
    # distance = minkovski(seqs[0], seqs[1], 2)

    seq_max = np.max(seqs[3], axis=1)
    seq_mean = np.mean(seqs[3], axis=1)
    print(seq_max)
    print(seq_mean)
    # delta
    d1 = np.min(seq_max)
    d2 = np.min(seq_mean)

    if os.path.exists(f"Task2_{data_name}_best_max.npy"):
        best_stack = np.load(f"Task2_{data_name}_best_max.npy", allow_pickle=True)
        best_stack_mean = np.load(f"Task2_{data_name}_best_mean.npy", allow_pickle=True)
    else:
        best_stack, best_stack_mean = task_2(seqs, f"Task2_{data_name}")
    best = np.argsort(best_stack[:, 1])[0]
    best_mean = np.argsort(best_stack_mean[:, 1])[0]
    # epsilon
    e1 = np.min(best_stack[:, 1])
    e1s = np.argsort(best_stack[:, 1])[:10]
    e2 = np.min(best_stack_mean[:, 1])
    e2s = np.argsort(best_stack_mean[:, 1])[:10]

    tos = np.ndarray((2, 3), dtype=object)
    tos[0] = np.array([1, e1, d1])
    tos[1] = np.array([2, e2, d2])
    np.save("epdel12", tos)







