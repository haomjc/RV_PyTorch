from torch.nn import functional as F
#import numpy as np
#from torch.autograd import Variable
import torch.nn as nn
#import torch.optim as optim
import torch

from Net import poly_model
 
model = torch.load('model.pkl')

x = [5.0000e+01, 1.1500e+02, 8.1000e-01, 2.0000e+00]

x=torch.tensor(x, dtype=torch.float).unsqueeze(0) 

output = model(x.repeat(8, 1)) 

print(output[0])


#Net:  104.3223, 123.9925, 145.3738, 123.4775,  92.2881,  23.8165,  20.8271, 5.2340,   6.2546,  30.0480,  52.7297,  95.3595,  64.5688,  50.3283,  20.6697
