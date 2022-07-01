import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

################################
########## bilinear ############
###############################

class TailNorm(torch.nn.Module):
    # input: complex tensor (...,2)
    # output: real possibility (...,1)
    def forward(self,x):
        x = 1-x.norm(dim=-1)  # so
        x = torch.nn.Sigmoid()(x)
        return x

class TailFermi(torch.nn.Module):
    # input: complex tensor (...,2)
    # output: real possibility (...,1)
    def forward(self,x,target=None):
        x = 10*(x.norm(dim=-1)-1)
        x = torch.nn.Sigmoid()(x)
        return x
