import torch
import torch.nn as nn
import torch.nn.functional as F

class BP18(nn.Module): 
    def __init__(self): 
        super(BP18, self).__init__() 
        self.poly1 = nn.Linear(4,16)
        self.poly2 = nn.Linear(16,16)  
        self.poly3 = nn.Linear(16,16)  
        self.poly4 = nn.Linear(16,16)  
        self.poly5 = nn.Linear(16,32)  
        self.poly6 = nn.Linear(32,32)  
        self.poly7 = nn.Linear(32,32)  
        self.poly8 = nn.Linear(32,32)  
        self.poly9 = nn.Linear(32,64)  
        self.poly10 = nn.Linear(64,64)
        self.poly11 = nn.Linear(64,64)  
        self.poly12 = nn.Linear(64,64)  
        self.poly13 = nn.Linear(64,128)  
        self.poly14 = nn.Linear(128, 128)  
        self.poly15 = nn.Linear(128, 128)  
        self.poly16 = nn.Linear(128, 128)  
        self.poly17 = nn.Linear(128, 128)  
        self.poly18 = nn.Linear(128,15)  

        self.dropout = nn.Dropout(p=0.5)   #加一个dropout,防止过拟合
  
    def forward(self, x): 
        out = self.poly1(x)
        out = F.leaky_relu(out)

        out = self.poly2(out)   
        out = F.leaky_relu(out)     
        
        out = self.poly3(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly4(out)   
        out = F.leaky_relu(out)   
        
        out = self.poly5(out)   
        out = F.leaky_relu(out)   
        
        out = self.poly6(out)   
        out = F.leaky_relu(out)       
        
        out = self.poly7(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly8(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly9(out)   
        out = F.leaky_relu(out)   
        
        out = self.poly10(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly11(out)   
        out = F.leaky_relu(out)     
        
        out = self.poly12(out)   
        out = F.leaky_relu(out)     
        
        out = self.poly13(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly14(out)   
        out = F.leaky_relu(out)   
        
        out = self.poly15(out)   
        out = F.leaky_relu(out)   
        
        out = self.poly16(out)   
        out = F.leaky_relu(out)       
        
        out = self.poly17(out)   
        out = F.leaky_relu(out)    
        
        out = self.poly18(out)   
        
        return out 
