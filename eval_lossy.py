import numpy as np
import torch
import os
from collections import OrderedDict

import torchvision.models as models
import time
import argparse

from compression import Float16, ResidualFloat16, ResEntropy16bits

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=str, default='0.01', help='learning rate when training these epochs')
    parser.add_argument('--epoch-interval', type=int, default=3, help='interval between two epochs')
    parser.add_argument('--epoch-first', type=int, default=21, help='first epoch')
    parser.add_argument('--epoch-last', type=int, default=100, help='last epoch')
    parser.add_argument('--dnn', type=str, default='yolo', help='dnn model')
    parser.add_argument('--method', type=str, default='ResEntropy16bits', help='compression method: Float16, ResidualFloat16, ResEntropy16bits')
    parser.add_argument('--path-pt', type=str, default='weights/yolov5n/', help='path for saved parameter files')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    
    dnn = opt.dnn       # network
    n = opt.epoch_interval # epoch interval
    method = opt.method    # Float16, ResidualFloat16, ResEntropy16bits

    #f = open('results/yolo_lossless_res-0.001-3.csv', 'w')
    #f.write('epoch,origsize,compsize,ratio\n')

    map_location = torch.device('cpu')

    start = opt.epoch_first
    end = opt.epoch_last
    lr = opt.learning_rate
    path_pt = opt.path_pt

    param2_r = None  # reconstructed param

    print('Start compressing ...')
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

        # param2_r is the reconstructed parameters
        if param2_r is not None:
            p_start = 0

            state_dict = net1.state_dict()
            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                key_type = key.split('.')[-1]
                if key_type == 'weight' or key_type == 'bias':
                    num = value.numel()
                    shape = value.size()
                    state_dict[key] = torch.from_numpy(np.reshape(param2_r[p_start:p_start+num], shape))
                    p_start = p_start+num
            net1.load_state_dict(state_dict)

        param1 = net1.parameters()
        param2 = net2.parameters()

        if method == 'Float16':
            rmse, diff_max, diff_min, originalsize, compressedsize = Float16(param2)
            print('Epoch:', i+n, '-', i, '\tOriginal Size:', np.around(originalsize/1024/1024,2), 'MB\tCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

        if method == 'ResidualFloat16':
            rmse, diff_max, diff_min, originalsize, compressedsize, param2_r = ResidualFloat16(param1, param2)
            print('Epoch:', i+n, '-', i, '\tOriginal Size:', np.around(originalsize/1024/1024,2), 'MB\tCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

        if method == 'ResEntropy16bits':
            start = time.time()
            rmse, diff_max, diff_min, originalsize, compressedsize, param2_r = ResEntropy16bits(param1, param2)
            print('Epoch:', i+n, '-', i, '\tCompression Time:', np.around(time.time()-start, 2), 's\tOriginal Size:', np.around(originalsize/1024/1024,2), 'MB\tCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

