import numpy as np
import torch
import torchvision.models as models
from sklearn.metrics import mean_squared_error

model_path1 = 'D:/pytorch/DifferentialEncoding/weights/resnet18/51_0.01.pth'
model_path2 = 'D:/pytorch/DifferentialEncoding/weights/resnet18/52_0.01.pth'

map_location = torch.device('cpu')

model1 = torch.load(model_path1, map_location=map_location)  # load
model2 = torch.load(model_path2, map_location=map_location)  # load

models1 = models.__dict__['resnet18']()
models1.load_state_dict(model1['state_dict'])

models2 = models.__dict__['resnet18']()
models2.load_state_dict(model2['state_dict'])

param1 = models1.parameters()
param2 = models2.parameters()

md1 = []
md2 = []
md3 = []

for i,(p1,p2) in enumerate(zip(param1,param2)):    
    p1_np = p1.detach().numpy().flatten()
    p2_np = p2.detach().numpy().flatten()

    md1.append(p1_np)
    md2.append(p2_np)
    md3.append(p2_np-p1_np)

md1 = np.concatenate(md1)
md2 = np.concatenate(md2)
md3 = np.concatenate(md3).astype(np.float16)+md1

diff = md2-md2.astype(np.float16) 
#diff = md2-md3
diff_max = np.max(diff)
diff_min = np.min(diff)
mse = np.square(diff)
mse = np.mean(mse)
rmse = np.sqrt(mse)

#mse = mean_squared_error(md2, md3)
print(mse, rmse, diff_max, diff_min)


