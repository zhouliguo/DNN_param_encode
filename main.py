import numpy as np
import torch
import struct
import os
import torchvision.models as models
import contextlib
import time
from collections import OrderedDict
from codec import arithmeticcoding
from codec.arithmetic_compress import get_frequencies, write_frequencies, compress
from sklearn.metrics import mean_squared_error
from codec.convert import dec2bin, bin2dec


#def float2byte(f):
#    return [hex(i) for i in struct.pack('f', f)]

'''
for i in range(29,48):
    model_path = 'weights/resnet18/'+str(i)+'_0.01.pth'
    map_location = torch.device('cpu')
    model = torch.load(model_path, map_location=map_location)  # load
    
    new_state_dict = OrderedDict()
    for key, value in model['state_dict'].items():
        new_state_dict[key[7:]]=value

    model['state_dict'] = new_state_dict
    torch.save(model, model_path)
quit()
'''

def compress_file(originalfile, compressedfile):
    # Read input file once to compute symbol frequencies
    freqs = get_frequencies(originalfile)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
        
    # Read input file again, compress with arithmetic coding, and write output file
    with contextlib.closing(arithmeticcoding.BitOutputStream(open(compressedfile, "wb"))) as bitout:
        write_frequencies(bitout, freqs)
        compress(freqs, originalfile, bitout)

    originalsize = os.path.getsize(originalfile)
    compressedsize = os.path.getsize(compressedfile)

    return originalsize, compressedsize

def compress_array(originalarray, compressedfile):
    # Read input file once to compute symbol frequencies
    freqs = get_frequencies(originalarray)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
        
    # Read input file again, compress with arithmetic coding, and write output file
    with contextlib.closing(arithmeticcoding.BitOutputStream(open(compressedfile, "wb"))) as bitout:
        write_frequencies(bitout, freqs)
        compress(freqs, originalarray, bitout)

    originalsize = len(originalarray)
    compressedsize = os.path.getsize(compressedfile)

    return originalsize, compressedsize

def ResEntropy(param1, param2):
    originalfile = 'npy/originalfile.npy'
    compressedfile = 'npy/compressedfile.npy'

    md = []

    for j, (p1,p2) in enumerate(zip(param1, param2)):    
        p1_np = p1.detach().numpy()
        p2_np = p2.detach().numpy()

        p1_np = p1_np.view((np.uint8, 4))
        p2_np = p2_np.view((np.uint8, 4))
        
        diff = (p2_np-p1_np).flatten()

        md.append(diff)

    md = np.concatenate(md)

    np.save(originalfile, md)

    return compress_file(originalfile, compressedfile)

def Float16(param2):
    md2 = []
    md3 = []

    for j,p2 in enumerate(param2):    
        p2_np = p2.detach().numpy().flatten()

        md2.append(p2_np)

    md2 = np.concatenate(md2)

    md3 = md2.astype(np.float16)

    diff = md2-md3

    diff_max = np.max(diff)
    diff_min = np.min(diff)

    rmse = np.sqrt(mean_squared_error(md2, md3))

    return rmse, diff_max, diff_min

def ResidualFloat16(param1, param2):
    md1 = []
    md2 = []
    md3 = []

    for j,(p1,p2) in enumerate(zip(param1,param2)):    
        p1_np = p1.detach().numpy().flatten()
        p2_np = p2.detach().numpy().flatten()

        md1.append(p1_np)
        md2.append(p2_np)
        md3.append(p2_np-p1_np)

    md1 = np.concatenate(md1)
    md2 = np.concatenate(md2)
    md3 = np.concatenate(md3).astype(np.float16)

    md3 = md3 + md1
    diff = md2 - md3

    diff_max = np.max(diff)
    diff_min = np.min(diff)
    rmse = np.sqrt(mean_squared_error(md2, md3))
    return rmse, diff_max, diff_min

def ResEntropy16bits(param1, param2):
    originalfile = 'npy/originalfile.npy'
    compressedfile = 'npy/compressedfile.npy'

    md1 = []
    md2 = []
    md3 = []

    for j,(p1,p2) in enumerate(zip(param1,param2)):    
        p1_np = p1.detach().numpy().flatten()
        p2_np = p2.detach().numpy().flatten()

        md1.append(p1_np)
        md2.append(p2_np)
        md3.append(p2_np-p1_np)

    md1 = np.concatenate(md1)
    md2 = np.concatenate(md2)
    md3 = np.concatenate(md3)

    md16bits = []

    start = time.time()
    for j, mdi in enumerate(md3):
        bits16 = dec2bin(mdi)
        md16bits.append(bits16)
        md3[j] = bin2dec(bits16)
    
    print(time.time()-start)

    md16bits = np.concatenate(md16bits)

    np.save(originalfile, md16bits)
    compress_file(originalfile, compressedfile)
    
    originalsize = os.path.getsize(originalfile) * 2
    compressedsize = os.path.getsize(compressedfile)

    md3 = md3+md1

    diff = md2 - md3

    diff_max = np.max(diff)
    diff_min = np.min(diff)
    rmse = np.sqrt(mean_squared_error(md2, md3))

    return rmse, diff_max, diff_min, originalsize, compressedsize

if __name__ == '__main__':
    cnn = 'yolo'        #network
    n = 5               #epoch interval
    method = 'ResEntropy'  #ResEntropy, Float16, ResidualFloat16, ResEntropy16bits

    #f = open('results/yolo_lossless_res-0.001-3.csv', 'w')
    #f.write('epoch,origsize,compsize,ratio\n')

    map_location = torch.device('cpu')

    for i in range(21,31-n):
        model_path1 = 'weights/yolov5n/'+str(i)+'_0.01.pt'
        model_path2 = 'weights/yolov5n/'+str(i+n)+'_0.01.pt'

        model1 = torch.load(model_path1, map_location=map_location)
        model2 = torch.load(model_path2, map_location=map_location)

        if cnn == 'resnet':
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
            print('Epoch:', i+n, '-', i, '\tCompression Time:', np.around(time.time()-start, 2), 's\tOriginal Size:', originalsize, 'MB\tCompressed Size:', compressedsize, 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%')

        if method == 'Float16': #Bit Saving: 50%
            rmse, diff_max, diff_min = Float16(param2)
            print('Epoch:', i+n, '\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

        if method == 'ResidualFloat16': #Bit Saving: 50%
            rmse, diff_max, diff_min = ResidualFloat16(param1, param2)
            print('Epoch:', i+n, '\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

        if method == 'ResEntropy16bits':
            start = time.time()
            rmse, diff_max, diff_min, originalsize, compressedsize = ResEntropy16bits(param1, param2)
            print('Epoch:', i+n, '-', i, '\tCompression Time:', np.around(time.time()-start, 2), 's\tOriginal Size:', originalsize, 'MB\tCompressed Size:', compressedsize, 'MB\tBit Saving:', np.around(100-100*compressedsize/originalsize, 2), '%', '\tRMSE:', rmse, '\tMax of Diff:', diff_max, '\tMin of Diff:', diff_min)

