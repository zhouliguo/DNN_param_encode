import numpy as np

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

def bin2dec(f):
    b0 = bin(f[0])[2:].zfill(8)
    b1 = bin(f[1])[2:].zfill(8)

    b0 = b0+b1

    d = 0
    for i, b in enumerate(b0):
        if i==0:
            s = b
            continue
        d = d+int(b)/(2**i)

    if s == '1':
        d = -d
    #d = d/4
    return d