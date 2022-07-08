import torch.nn as nn
import torch.nn.functional as F
import torch as th

class DNN(nn.Module):
    def __init__(self,input_size,out_size=1):
        super(DNN, self).__init__()
        ##DNN部分
        self.Dense1 = nn.Linear(in_features = input_size, out_features=8, bias = False)
        self.Dense2 = nn.Linear(in_features = 8, out_features = out_size, bias = False )
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x = self.Dense1(x)
        x = self.relu(x)
        x = self.Dense2(x)
        y = self.sigmoid(x)
        y = y.squeeze(-1)
        return y
