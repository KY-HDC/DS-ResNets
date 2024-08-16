from models import resnet
from torch import nn


def model_(params, device):
    par = dict()
    par['model'] = params.model
    par['block'] = params.bc
    par['weight'] = params.weight

 
    mod = resnet.ResNet(block=resnet.Bottleneck, layers=par['block'])
    if par['weight']:
        mod.load_state_dict(resnet.Resnet50_Weights.DEFAULT, False)
        
    return mod
