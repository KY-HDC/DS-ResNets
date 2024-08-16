import sys
import os
import torch
import utils.models as ml
import argparse
from utils.ResNets import ResNet as Resnet, Bottleneck

import numpy as np

if __name__ == '__main__':
    param = argparse.ArgumentParser()
    param.add_argument( 
        '--model',
        default='resnet',   
        type=str
    )
    param.add_argument(
        '--bc',
        help='block configuration',
        default=[3, 4, 6, 3],
        type=list
    )
    param.add_argument(
        '--weight',
        default=False,
        type=bool
    )   
    params = param.parse_args()
    seed = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    resnet_blocks = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 18:[2, 2, 2, 2]}
    key = [50, 101, 18]
    
    data_name = 'data_name'
    
    init_rand(seed)
    train_dataset, test_dataset = load_data(data_name)
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=16,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,  
                                             batch_size=16,
                                             shuffle=False)


    
    model = ml.model_(params, device)
    ml.transfer(model, seed, 2, 2048 * 7 * 14)
    model.to(device)
    train(model, trainloader, testloader, device, epochs=100, es=20, lpth=f"Ressenet_{data_name}", lr=.00005)
    
    extractor = Resnet(block=Bottleneck, layers=params.bc)
    extractor = Exprob(2048*7*14, 2)
    
    trainloader = torch.utils.data.DataLoader(dataset=tt_set, batch_size=20, shuffle=False)
    extractor.load_state_dict(torch.load("model_weight_path"))
    extractor.to(device)
    extractor.eval()
    fb = True
    for xx, yy in testloader:
        xx = xx.to(device)
        out = extractor(xx).detach().cpu()
        if fb:
            xout = out
            fb = False
        else:
            xout = torch.cat((xout, out), axis=0)
        torch.save(xout, f"prob/{data_name}/{data_name}_block{b}.pt")

    extractor.load_state_dict(weigh)
    extractor.to(device)
    extractor.eval()
    ext = dict()
    print(len(testloader)) 
    for i, xy in enumerate(testloader):
        x, y = xy
        out = extractor(x.to(device))
        if i == 0:
            yout = y
        else:
            yout = torch.cat((yout, y), axis=0)
        for key in out.keys():
            torch.save(out[key].cpu(), f"pix/resnet/{data_name}/test/{data_name}_block{key}_{i}.pt")
    torch.save(yout, f"pix/resnet/{data_name}/test/{data_name}_label.pt")
