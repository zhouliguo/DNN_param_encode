import numpy as np
import os
from collections import OrderedDict
import contextlib
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
    originalsize, compressedsize = compress_file(originalfile, compressedfile)

    return originalsize, compressedsize

def Float16(param2):
    md2 = []

    for j,p2 in enumerate(param2):    
        p2_np = p2.detach().numpy().flatten()

        md2.append(p2_np)

    md2 = np.concatenate(md2)

    md2_ = md2.astype(np.float16)

    originalsize = len(md2)*4
    compressedsize = len(md2_)*2

    diff = md2-md2_

    diff_max = np.max(diff)
    diff_min = np.min(diff)

    rmse = np.sqrt(mean_squared_error(md2, md2_))

    return rmse, diff_max, diff_min, originalsize, compressedsize

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

    originalsize = len(md2)*4
    compressedsize = len(md3)*2

    md2_r = md3 + md1
    diff = md2 - md2_r

    diff_max = np.max(diff)
    diff_min = np.min(diff)
    rmse = np.sqrt(mean_squared_error(md2, md2_r))

    return rmse, diff_max, diff_min, originalsize, compressedsize, md2_r

def ResEntropy16bits(param1, param2):
    bits16file = 'npy/bits16file.npy'
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

    for j, mdi in enumerate(md3):
        bits16 = dec2bin(mdi)
        md16bits.append(bits16)
        md3[j] = bin2dec(bits16)
    
    md16bits = np.concatenate(md16bits)

    np.save(bits16file, md16bits)
    compress_file(bits16file, compressedfile)
    
    originalsize = os.path.getsize(bits16file) * 2
    compressedsize = os.path.getsize(compressedfile)

    md2_r = md3+md1
    diff = md2 - md2_r

    diff_max = np.max(diff)
    diff_min = np.min(diff)
    rmse = np.sqrt(mean_squared_error(md2, md2_r))

    return rmse, diff_max, diff_min, originalsize, compressedsize, md2_r