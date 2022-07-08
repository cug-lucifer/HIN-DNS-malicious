import torch as th
import torch
import pandas as pd
import numpy as np
import torchvision.models as models
model = th.load('./model/save_model/DNSGNN_all_0_1000_3p.pth')



parm={}
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
    parm[name]=parameters.detach().numpy()

print(parm['model.SemanticConv.project.2.weight'][0,:])