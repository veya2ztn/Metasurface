import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import BaseModel,Inverse_Model,conv_shrink
import numpy as np

class U1Tanh(nn.Module):
    def __init__(self,sloop=1):
        super().__init__()
        self.sloop=1
    def forward(self,x):
        x = nn.Tanh()(self.sloop*x)
        x = (x+1)/2
        return x
class DenseLayer_A(nn.Module):
    def __init__(self,channel_list):
      super().__init__()
      self.layers = nn.ModuleList()
      for i in range(len(channel_list)-1):
            self.layers.append(nn.Sequential(
              nn.Linear(channel_list[i],channel_list[i+1]),
              nn.Dropout(),
              U1Tanh()
              ))

    def test(self,x):
      with torch.no_grad(): #<--very important
        x = self(x)
        return x
    def forward(self,x):
        for layer in self.layers:
          x = layer(x)
        return x



class HyperDense(Inverse_Model):
    def __init__(self,image_type,curve_type,dense_type,channel_list,**kargs):
        assert curve_type.data_shape[0]==1
        super().__init__(image_type,curve_type,**kargs)
        self.layers = nn.ModuleList([dense_type(channel_list) for i in range(np.prod(image_type.data_shape))])
    def forward(self, x,target=None):
        batch_size, branch, HW = x.size()
        output=[layer(x) for layer in self.layers]
        x     = torch.cat(output)
        x     = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class HyperDense_A1(HyperDense):
    def __init__(self,image_type,curve_type,**kargs):
        channel_list = [curve_type.data_shape[-1] ,32,16,8,4,1]
        dense_type   = DenseLayer_A
        super().__init__(image_type,curve_type,dense_type,channel_list,**kargs)
class HyperDense_A9(HyperDense):
    def __init__(self,image_type,curve_type,**kargs):
        channel_list = [curve_type.data_shape[-1] ,32,64,128,64,32,16,32,64,128,64,32,16,8,4,1]
        dense_type   = DenseLayer_A
        super().__init__(image_type,curve_type,dense_type,channel_list,**kargs)

class HyperDense_A999(HyperDense):
    def __init__(self,image_type,curve_type,**kargs):
        channel_list = [curve_type.data_shape[-1]]+[32,64,128,64,32,16]*10+[16,8,4,1]
        dense_type   = DenseLayer_A
        super().__init__(image_type,curve_type,dense_type,channel_list,**kargs)
