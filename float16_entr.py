import numpy as np
import torch
import collections
import codec.ShannonCoding as ShannonCoding
import struct
import os
import torchvision.models as models
from arithmetic_compress import get_frequencies, write_frequencies, compress
#from adaptive_arithmetic_compress import compress
import arithmeticcoding
import contextlib
import sys
import time
from collections import OrderedDict

#Exponential-Golomb coding
def exp_golomb_code(x, sign=True):
    if sign:
        if x == 0:
            return '1'
        if x > 0:
            x = 2*x-1
        else:
            x = -2*x
            
        x_a = x+1
        x_a_bin = bin(x_a).replace('0b','')
        z = x_a_bin.replace('1','0')[:-1]
        g = z+x_a_bin
    else:
        if x == 0:
            return '1'     
        x_a = x+1
        x_a_bin = bin(x_a).replace('0b','')
        z = x_a_bin.replace('1','0')[:-1]
        g = z+x_a_bin

    return g

def float2byte(f):
    return [hex(i) for i in struct.pack('f', f)]

def dec2bin(f):
    #f = f*4
    if f>=0:
        b0=0
    else:
        b0=1
        f=-f

    for i in range(7):
        m = f*2
        f = m-int(m)
        b0 = b0*2+int(m)
    
    b1=0
    for i in range(8):
        m = f*2
        f = m-int(m)
        b1 = b1*2+int(m)

    return np.array([b0, b1], np.uint8)

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

resnet = False
n = 5

f = open('results/yolo_float16_xxx-0.01-5.csv', 'w')
f.write('epoch,origsize,compsize,ratio\n')

map_location = torch.device('cpu')

for i in range(2,32-n):
    model_path1 = 'weights/yolov5n/epoch'+str(i)+'_lr_0.01.pt'
    model_path2 = 'weights/yolov5n/epoch'+str(i+n)+'_lr_0.01.pt'

    model1 = torch.load(model_path1, map_location=map_location)  # load
    model2 = torch.load(model_path2, map_location=map_location)  # load

    if resnet:
        net1 = models.__dict__['resnet18']()
        net1.load_state_dict(model1['state_dict'])
        net2 = models.__dict__['resnet18']()
        net2.load_state_dict(model2['state_dict'])
    else:
        net1 = model1['model']
        net2 = model2['model']

    param1 = net1.parameters()
    param2 = net2.parameters()

    sourcefile = 'npy/yolo-0.01-5-'+str(i+n)+'.npy'
    sourcefile1 = 'npy/yolo-0.01-5-'+str(i+n)+'_1.npy'
    compressfile = 'npy/yolocode-0.01-5-'+str(i+n)+'.npy'

    md = []

    for j,(p1,p2) in enumerate(zip(param1,param2)):    
        p1_np = p1.detach().numpy()
        p2_np = p2.detach().numpy()
        
        diff = (p2_np-p1_np).flatten()

        md.append(diff)

    md = np.concatenate(md)

    np.save(sourcefile, md)

    mdbin = []
    for mdi in md:
        mdbin.append(dec2bin(mdi))

    mdbin = np.concatenate(mdbin)

    np.save(sourcefile1, mdbin)

    start = time.time()

    # Read input file once to compute symbol frequencies
    freqs = get_frequencies(sourcefile1)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
        
    # Read input file again, compress with arithmetic coding, and write output file
    with contextlib.closing(arithmeticcoding.BitOutputStream(open(compressfile, "wb"))) as bitout:
        write_frequencies(bitout, freqs)
        compress(freqs, sourcefile1, bitout)

    sourcefile_size = os.path.getsize(sourcefile)
    sourcefile_size = sourcefile_size
    compressfile_size = os.path.getsize(compressfile)#+sourcefile_size/4/8+4
    
    print(compressfile_size/sourcefile_size)

    f.write(str(i+n)+','+str(sourcefile_size)+','+str(compressfile_size)+','+str(compressfile_size/sourcefile_size)+'\n')

    print(i, time.time()-start)
