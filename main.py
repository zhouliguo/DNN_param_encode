import numpy as np
import torch
import collections
import ShannonCoding
import struct
import cv2
import torchvision.models as models
from arithmetic_compress import get_frequencies, write_frequencies, compress
#from adaptive_arithmetic_compress import compress
import arithmeticcoding
import contextlib
import sys
import time

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


model_path1 = 'D:/pytorch/DifferentialEncoding/weights/resnet18/51_0.01.pth'
model_path2 = 'D:/pytorch/DifferentialEncoding/weights/resnet18/58_0.01.pth'

map_location = torch.device('cpu')

model1 = torch.load(model_path1, map_location=map_location)  # load
model2 = torch.load(model_path2, map_location=map_location)  # load

models1 = models.__dict__['resnet18']().half()
models1.load_state_dict(model1['state_dict'])
models2 = models.__dict__['resnet18']().half()
models2.load_state_dict(model2['state_dict'])

param1 = models1.parameters()
param2 = models2.parameters()


md = []

for i,(p1,p2) in enumerate(zip(param1,param2)):    
    p1_np = p1.detach().numpy()
    p2_np = p2.detach().numpy()

    p1_np = p1_np.view((np.uint8, 4))
    p2_np = p2_np.view((np.uint8, 4))
    
    diff = (p2_np-p1_np).flatten()
    #diff = p1_np.flatten()

    md.append(diff)

md = np.concatenate(md)

np.save('res.npy', md)

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


inputfile = 'res.npy'
outputfile = 'rescodea.npy'

start = time.time()

# Read input file once to compute symbol frequencies
freqs = get_frequencies(inputfile)
freqs.increment(256)  # EOF symbol gets a frequency of 1
	
# Read input file again, compress with arithmetic coding, and write output file
with contextlib.closing(arithmeticcoding.BitOutputStream(open(outputfile, "wb"))) as bitout:
	write_frequencies(bitout, freqs)
	compress(freqs, inputfile, bitout)

print(time.time()-start)
