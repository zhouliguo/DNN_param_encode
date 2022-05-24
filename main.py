import numpy as np
import torch
import collections
import codec.ShannonCoding as ShannonCoding
import struct
import os
import torchvision.models as models
from codec.arithmetic_compress import get_frequencies, write_frequencies, compress
#from adaptive_arithmetic_compress import compress
from codec import arithmeticcoding
import contextlib
import sys
import time
from collections import OrderedDict

def float2byte(f):
    return [hex(i) for i in struct.pack('f', f)]

def dec2bin(f):
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

cnn = 'resnet'
n = 3

f = open('results/yolo_lossless_res-0.001-3.csv', 'w')
f.write('epoch,origsize,compsize,ratio\n')

map_location = torch.device('cpu')

for i in range(268,277-n):
    model_path1 = 'weights/yolov5n/'+str(i)+'_0.001.pt'
    model_path2 = 'weights/yolov5n/'+str(i+n)+'_0.001.pt'

    model1 = torch.load(model_path1, map_location=map_location)  # load
    model2 = torch.load(model_path2, map_location=map_location)  # load

    if cnn == 'resnet':
        net1 = models.__dict__['resnet18']()
        net1.load_state_dict(model1['state_dict'])
        net2 = models.__dict__['resnet18']()
        net2.load_state_dict(model2['state_dict'])
    else:
        net1 = model1['model']
        net2 = model2['model']

    param1 = net1.parameters()
    param2 = net2.parameters()

    sourcefile = 'npy/yolo-0.001-3-'+str(i+n)+'.npy'
    compressfile = 'npy/yolocode-0.001-3-'+str(i+n)+'.npy'

    md = []

    for j,(p1,p2) in enumerate(zip(param1,param2)):    
        p1_np = p1.detach().numpy()
        p2_np = p2.detach().numpy()

        p1_np = p1_np.view((np.uint8, 4))
        p2_np = p2_np.view((np.uint8, 4))
        
        diff = (p2_np-p1_np).flatten()
        #diff = p1_np.flatten()

        md.append(diff)

    md = np.concatenate(md)

    np.save(sourcefile, md)

    # Calculate symbol frequency
    #count = collections.Counter(list(md))

    # symbol list
    #color = list(count.keys())

    # frequency list
    #number = list(count.values())
    #number = np.array(number)

    # probabilities list
    #p = number / np.sum(number)

    #shannon = ShannonCoding.ShannonCoding(color, p)

    # encode
    #total_code = shannon.encode(md)

    # decode
    #a = shannon.decode(total_code)

    #shannon.print_format('Gray')

    #print('Compression ratio:', len(total_code) / (len(md) * 8))

    start = time.time()

    # Read input file once to compute symbol frequencies
    freqs = get_frequencies(sourcefile)
    freqs.increment(256)  # EOF symbol gets a frequency of 1
        
    # Read input file again, compress with arithmetic coding, and write output file
    with contextlib.closing(arithmeticcoding.BitOutputStream(open(compressfile, "wb"))) as bitout:
        write_frequencies(bitout, freqs)
        compress(freqs, sourcefile, bitout)

    sourcefile_size = os.path.getsize(sourcefile)
    compressfile_size = os.path.getsize(compressfile)

    f.write(str(i+n)+','+str(sourcefile_size)+','+str(compressfile_size)+','+str(compressfile_size/sourcefile_size)+'\n')

    print(i, time.time()-start)
