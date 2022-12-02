# -*- coding: utf-8 -*-

#把ResNet18加入程序

from    torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from RV_data import RV_data

from visdom import Visdom
 
from BPNet import BP18    #导入神经网络模型
from Resnet18 import ResNet18
from Resnet34 import ResNet34

import random

def get_batch(epoch): 
    #i = epoch%8
    i = random.randint(0, 8)
    #print(i)
    x_6E, y_6E, x_20E, y_20E, x_40E, y_40E,  x_80E, y_80E, x_110E, y_110E, x_160E, y_160E, x_320E, y_320E, x_450E, y_450E = RV_data()
    x1 = [x_6E[i],x_40E[i],x_110E[i],x_320E[i] ]*(1+torch.randn(4,4).cpu().numpy()*0.01)    #采用高斯随机分布增加样本数量
    #i = random.randint(0, 8)
    #print(x)
    #x2 = [x_6E[i],x_20E[i],x_40E[i],x_80E[i],x_110E[i],x_160E[i],x_320E[i],x_450E[i] ]*(1+torch.randn(8,4).cpu().numpy()*0.01)    

    #每一批是8种型号减速器的1个工况
    #x=torch.tensor(x, dtype=torch.float).unsqueeze(1) 
    x = np.vstack([x1,])
    x=torch.tensor(x, dtype=torch.float)
    #x = make_features(x)  
    test_x1=torch.tensor([x_20E[i],x_80E[i],x_160E[i],x_450E[i] ],  dtype=torch.float)    
    
    y = [y_6E,y_40E, y_110E,y_320E ]   #变量y样本            
    y = np.vstack([y, ])
    #y=torch.tensor(y, dtype=torch.float).unsqueeze(1) 
    y=torch.tensor(y, dtype=torch.float)
    #y = make_features(y)  
    #print(y)
    test_y = torch.tensor([y_20E,y_80E,y_160E, y_450E],  dtype=torch.float)
    return Variable(x), Variable(y), Variable(test_x1), Variable(test_y)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss')) 
viz.line([0.], [0.], win='test_loss', opts=dict(title='test loss')) 

#model = poly_model()          #选择神经网络模型

model = ResNet18()

criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr = 1e-6, momentum=0.5)   #加动量后收敛反而慢？
#optimizer = optim.SGD(model.parameters(), lr = 1e-5)

epoch = 1
print_loss = 0
print_test_loss=0



pd_epoch = []
pd_print_loss = []

pd_print_test_loss = []


while True:   #定义loss到多少停止运算

    batch_x,batch_y, test_x, test_y = get_batch(epoch) 
    #print(batch_x)
    output = model(batch_x) 
    #print(output)
    loss = criterion(output,batch_y)
    
    test_output = model(test_x) 
    test_loss = criterion(test_output,test_y)
    
    #loss = criterion(output,batch_y) 
    print_loss = print_loss+loss.item()
    print_test_loss = print_test_loss+test_loss.item()

    
    optimizer.zero_grad() 

    loss.backward()     
    optimizer.step() 
    epoch+=1

    draw_acc = 100
    if epoch%draw_acc == 0:
        viz.line([print_loss/draw_acc], [epoch/9*1], win='train_loss', update='replace' if epoch == draw_acc else 'append')
        viz.line([print_test_loss/draw_acc], [epoch/9*1], win='test_loss', update='replace' if epoch == draw_acc else 'append')        
        
        
        print("epoch:", epoch)
        print("train_loss:%.10f\n" % (print_loss/draw_acc))
        print("test_loss:%.10f\n" % (print_test_loss/draw_acc))
        
        pd_epoch.append(epoch/9*1)
        pd_print_loss.append(print_loss/draw_acc)        
        pd_print_test_loss.append(print_test_loss/draw_acc)   
        
        
        if print_test_loss/draw_acc<100:          #允许的MSELoss平均误差
            break        
            
        if epoch/9*1>7000:
            break
            
        print_loss = 0 
        print_test_loss = 0
#print(pd_print_loss)
import pandas as pd
#写入文件
pd_data = pd.DataFrame(np.vstack([np.array(pd_epoch), np.array(pd_print_loss), np.array(pd_print_test_loss)]).T,columns=['epoch','train_loss', 'test_loss'])
pd_data.to_csv('D:\\print_loss,epoch8-18layer.csv')        
        
torch.save(model, '18layer-model.pkl')
