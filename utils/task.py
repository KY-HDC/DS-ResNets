import sys

from utils.distance import minkovski, hausdorff_distance
from utils.builders import create_dict
import os
import numpy as np
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DistanceMeasure:
    def __init__(self, data, labels, norm=None):
        super(DistanceMeasure, self).__init__()
        self.input_dict = create_dict(data, labels, norm=norm)
        for key in self.input_dict.keys():
            self.input_dict[key] = torch.from_numpy(self.input_dict[key])
        self.out_dict = dict()

    def single(self, in1=0, in2=4):
        print(minkovski(self.input_dict[0][in1], self.input_dict[0][in2], 2),
              np.argmax(self.input_dict[0][in1], 1),
              np.argmax(self.input_dict[0][in2], 1))

    def class_wise(self, d_name, path='Result'):
        path_ = os.path.join(path, "task1_old")
        if not os.path.exists(path_):
            os.makedirs(path_)

        for key in self.input_dict.keys():
            for i in range(self.input_dict[key][:].shape[0]):
                if i == 0:
                    tmp = minkovski(self.input_dict[key][i], self.input_dict[key], 2).reshape(1, -1)
                else:
                    tmp = np.concatenate((tmp, minkovski(self.input_dict[key][i],
                                                         self.input_dict[key], 2).reshape(1, -1)), 0)
            np.save(f"{path_}/{d_name}_Class_{key}_dist_all", tmp)
            tmp = None

    def task_1(self, d_name, m_name, path='Result', log=False, stat="", p=None):
        path_ = os.path.join(path, "task1")
        if not os.path.exists(path_):
            os.makedirs(path_)
        if p is None:
            p = f"{d_name}"
        key_list = self.input_dict.keys()
        print(key_list)
        for key in sorted(key_list):
            print(self.input_dict[key].shape, key)
            for key2 in sorted(key_list):
                min_dist = 999
                min_loc1 = 0
                min_loc2 = 0
                min_block = 0
                # if key == key2:
                #     continue
                for i in range(self.input_dict[key].shape[0]):
                    if key == key2:
                        # if i+1 == self.input_dict[key].shape[0]:
                        #     continue
                        ttt = self.input_dict[key].clone()
                        ttt[:i+1] *= 10
                        tps = minkovski(ttt, self.input_dict[key2][i], 2).to("cuda")
                    else:
                        tps = minkovski(self.input_dict[key], self.input_dict[key2][i], 2).to("cuda")
                   
                    if i == 0:
                        tmp = tps.reshape(1, 1, tps.shape[1], tps.shape[0])
                        # tmp = np.array([tps]).reshape(1, 1)
                    else:
                        tmp = torch.cat((tmp, tps.reshape(1, 1, tps.shape[1], tps.shape[0])), -1)
                        # tmp = np.concatenate((tmp, np.array([tps]).reshape(1, 1)), -1)
                if "tmp_" not in locals() or tmp_ is None:
                    tmp_ = tmp.to("cpu")
                else:
                    dif = tmp_.shape[-1] - tmp.shape[-1]
                    if dif != 0:
                        if dif > 0:
                            pad = torch.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3] + dif))
                            pad[:, :, :, :tmp.shape[3]] = tmp.to("cpu")
                            tmp_ = torch.cat((tmp_, pad), 1)
                        else:
                            pad = torch.zeros((tmp_.shape[0], tmp_.shape[1], tmp_.shape[2], tmp_.shape[3] - dif))
                            pad[:, :, :, :tmp_.shape[3]] = tmp_
                            tmp_ = torch.cat((pad, tmp.to("cpu")), 1)
                    else:
                        tmp_ = torch.cat((tmp_, tmp.to("cpu")), 0)
                tmp = None
            np.save(f"{path_}/{d_name}_{m_name}_Class_{key}_prob", tmp_.numpy())
            if log:
                print(key, tmp_.shape)
            
            tmp_ = None


def task_2(seq_set, d_name, path='Result', log=False):
    path_ = os.path.join(path, "task2")
    if not os.path.exists(path_):
        os.makedirs(path_)

    best_stack = []
    best_stack_mean = []
    count = 0
    for star in seq_set[0]:
        print(star.shape, seq_set[1].shape)
        distance = minkovski(star.reshape(1, star.shape[0], star.shape[1]), seq_set[1], 2)
        max_dist = np.min(distance, axis=1)
        mean_dist = np.mean(distance, axis=1)
        max_loc = np.argmin(max_dist)
        max_mean_loc = np.argmin(mean_dist)
        best_stack.append([max_loc, max_dist[max_loc]])
        best_stack_mean.append([max_mean_loc, mean_dist[max_mean_loc]])
        count += 1
        if count % 1000 == 0 and log:
            print(count/seq_set[0].shape[0])
    best_stack = np.array(best_stack, dtype=object)
    best_stack_mean = np.array(best_stack_mean, dtype=object)
    np.save(f"{path_}/{d_name}_best_", best_stack)
    np.save(f"{path_}/{d_name}_best_mean", best_stack_mean)

    return best_stack, best_stack_mean
