from models import resnet
from torch import nn


def model_(params, device):
    par = dict()
    par['model'] = params.model
    par['block'] = params.bc
    par['weight'] = params.weight

    if par['model'] == "densenet":
        mod = densenet.DenseNet(block_config=par['block'], device=device)
        if par['weight']:
            densenet._load_state_dict(mod, densenet.DenseNet121_Weights.DEFAULT, False)
    elif par['model'] == "resnet":
        mod = resnet.ResNet(block=resnet.Bottleneck, layers=par['block'])
        if par['weight']:
            mod.load_state_dict(resnet.Resnet50_Weights.DEFAULT, False)
    return mod
