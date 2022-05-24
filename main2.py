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

resnet = True
n = 5

map_location = torch.device('cpu')

for i in range(59,70):
    model_path1 = 'weights/resnet18/'+str(i)+'_0.001.pth'

    model1 = torch.load(model_path1, map_location=map_location)  # load

    if resnet:
        net1 = models.__dict__['resnet18']()
        net1.load_state_dict(model1['state_dict'])
    else:
        net1 = model1['model']

    param1 = net1.parameters()

    md = []

    for j,p1 in enumerate(param1):    
        p1_np = p1.detach().numpy().flatten()
        
        md.append(p1_np)

    md = np.concatenate(md)

    np.save('npy/resnet18/'+str(i)+'_0.001.npy', md)
    continue
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
