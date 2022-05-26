import numpy as np
import torch

import torchvision.models as models
import time
import argparse

from compression import ResEntropy, Float16, ResidualFloat16, ResEntropy16bits

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-epoch', type=str, default='weights/yolov5n/21_0.01.pt', help='path of params of previous epoch')
    parser.add_argument('--cur-epoch', type=str, default='weights/yolov5n/24_0.01.pt', help='path of params of current epoch')
    parser.add_argument('--dnn', type=str, default='yolo', help='dnn model')
    parser.add_argument('--method', type=str, default='ResEntropy16bits', help='compression method')
    
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    
    dnn = opt.dnn        #network
    method = opt.method  #ResEntropy, Float16, ResidualFloat16, ResEntropy16bits

    model_path1 = opt.pre_epoch
    model_path2 = opt.cur_epoch

    map_location = torch.device('cpu')

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

    if method == 'ResEntropy': #Lossless
        start = time.time()
        originalsize, compressedsize = ResEntropy(param1, param2)
        #f.write(str(i+n)+','+str(sourcefile_size)+','+str(compressfile_size)+','+str(compressfile_size/sourcefile_size)+'\n')
        print('Compression Time:', np.around(time.time()-start, 2), 's\nOriginal Size:', np.around(originalsize/1024/1024,2), 'MB\nCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\nBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%')

    if method == 'Float16':
        rmse, diff_max, diff_min, originalsize, compressedsize = Float16(param2)
        print('Original Size:', np.around(originalsize/1024/1024,2), 'MB\nCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\nBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%\nRMSE:', rmse, '\nMax of Diff:', diff_max, '\nMin of Diff:', diff_min)
    if method == 'ResidualFloat16': 
        rmse, diff_max, diff_min, originalsize, compressedsize, _ = ResidualFloat16(param1, param2)
        print('Original Size:', np.around(originalsize/1024/1024,2), 'MB\nCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\nBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%\nRMSE:', rmse, '\nMax of Diff:', diff_max, '\nMin of Diff:', diff_min)

    if method == 'ResEntropy16bits':
        start = time.time()
        rmse, diff_max, diff_min, originalsize, compressedsize, _ = ResEntropy16bits(param1, param2)
        print('Compression Time:', np.around(time.time()-start, 2), 's\nOriginal Size:', np.around(originalsize/1024/1024,2), 'MB\nCompressed Size:', np.around(compressedsize/1024/1024,2), 'MB\nBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%', '\nRMSE:', rmse, '\nMax of Diff:', diff_max, '\nMin of Diff:', diff_min)

