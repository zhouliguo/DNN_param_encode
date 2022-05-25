import numpy as np
import torch
import struct
import os

import torchvision.models as models
import time
import argparse

from compression import ResEntropy

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=str, default='0.01', help='learning rate when training these epochs')
    parser.add_argument('--epoch-interval', type=int, default=3, help='interval between two epochs')
    parser.add_argument('--epoch-first', type=int, default=21, help='first epoch')
    parser.add_argument('--epoch-last', type=int, default=100, help='last epoch')
    parser.add_argument('--dnn', type=str, default='yolo', help='dnn model')
    parser.add_argument('--method', type=str, default='ResEntropy', help='compression method')
    parser.add_argument('--path-pt', type=str, default='weights/yolov5n/', help='path for saved parameter files')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    
    dnn = opt.method       #network
    n = opt.epoch_interval #epoch interval
    method = opt.method    #ResEntropy, Float16, ResidualFloat16, ResEntropy16bits

    #f = open('results/yolo_lossless_res-0.001-3.csv', 'w')
    #f.write('epoch,origsize,compsize,ratio\n')

    map_location = torch.device('cpu')

    start = opt.epoch_first
    end = opt.epoch_last
    lr = opt.learning_rate
    path_pt = opt.path_pt

    for i in range(start, end-n, n):
        model_path1 = os.path.join(path_pt, str(i)+'_'+lr+'.pt')
        model_path2 = os.path.join(path_pt, str(i+n)+'_'+lr+'.pt')

        model1 = torch.load(model_path1, map_location=map_location)
        model2 = torch.load(model_path2, map_location=map_location)

        if dnn == 'resnet':
            net1 = models.__dict__['resnet18']()
            net1.load_state_dict(model1['state_dict'])
            net2 = models.__dict__['resnet18']()
            net2.load_state_dict(model2['state_dict'])
        else:
            net1 = model1['model']
            net2 = model2['model']

        #param1 = net1.state_dict()
        #param2 = net2.state_dict()

        param1 = net1.parameters()
        param2 = net2.parameters()

        start = time.time()
        originalsize, compressedsize = ResEntropy(param1, param2)
        #f.write(str(i+n)+','+str(sourcefile_size)+','+str(compressfile_size)+','+str(compressfile_size/sourcefile_size)+'\n')
        print('Epoch:', i+n, '-', i, '\tCompression Time:', np.around(time.time()-start, 2), 's\tOriginal Size:', originalsize, 'MB\tCompressed Size:', compressedsize, 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%')
