import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import mean_squared_error

def dec2bin1(f):
    #f  = f*4
    if f>=0:
        s=0
    else:
        s=1
        f=-f
    b0 = 0
    for i in range(8):
        m = f*2
        f = m-int(m)
        b0 = b0*2+int(m)
    
    b1=0
    for i in range(8):
        m = f*2
        f = m-int(m)
        b1 = b1*2+int(m)

    b2=0
    for i in range(8):
        m = f*2
        f = m-int(m)
        b2 = b2*2+int(m)

    b3=0
    for i in range(8):
        m = f*2
        f = m-int(m)
        b3 = b3*2+int(m)

    return np.array([b0, b1], np.uint8)

def dec2bin(f):
    #f  = f*4
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

#net1 = models.__dict__['resnet18']()
#net2 = models.__dict__['resnet18']()
map_location = torch.device('cpu')

n = 5
f = open('results/yolo_float16_xxx-0.001-5.csv', 'w')
f.write('epoch,\trmse,\t abs_mean,\t max,\t min\n')

resnet = False
for i in range(268,277-n):
    model_path1 = 'weights/yolov5n/epoch'+str(i)+'_lr_0.001.pt'
    model_path2 = 'weights/yolov5n/epoch'+str(i+n)+'_lr_0.001.pt'


    model1 = torch.load(model_path1, map_location=map_location)  # load
    model2 = torch.load(model_path2, map_location=map_location)  # load

    #net1.load_state_dict(model1['state_dict'])
    #net2.load_state_dict(model2['state_dict'])

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

    #print(np.max(md3))
    #print(np.min(md3))
    for j, mdi in enumerate(md3):
        md3[j] = bin2dec(dec2bin1(mdi))
    md3 = md3+md1
    #md3 = md3.astype(np.float16)+md1
    #md3 = md2.astype(np.float16)

    diff = md2-md3

    diff_max = np.max(diff)
    diff_min = np.min(diff)

    abs_mean = np.mean(np.abs(diff))

    rmse = np.sqrt(mean_squared_error(md2, md3))
    print(i+n, rmse, diff_max, diff_min)
    f.write(str(i+n)+','+str(rmse)+','+str(abs_mean)+','+str(diff_max)+','+str(diff_min)+'\n')

f.close()
