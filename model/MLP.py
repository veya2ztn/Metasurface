import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import sys
from .model import BaseModel,Forward_Model,Inverse_Model
################################
############ MLP ###############
################################


from mltool.ModelArchi.BackboneUnit import MLPlayer
class MLP_Forward(Forward_Model):
    def __init__(self,image_type,curve_type,mid_channel_list,activater=nn.ReLU,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.in_put_fea   = self.input_dim
        channel_list      = [self.in_put_fea]+mid_channel_list
        self.layer        = MLPlayer(channel_list,activater=activater)
        self.final        = torch.nn.Linear(channel_list[-1],self.output_dim)

    def forward(self,x,target=None):
        x = x.reshape(x.size(0),-1)
        x = self.layer(x)
        x = self.final(x)
        x = x.reshape(self.final_shape)
        if target is None:
            return x
        else:
            loss = self._loss(x,target)
            return loss,x

class MLP_Inverse(Inverse_Model):
    def __init__(self,image_type,curve_type,mid_channel_list,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.in_put_fea   = self.input_dim
        channel_list      = [self.in_put_fea]+mid_channel_list
        self.layer        = MLPlayer(channel_list)
        self.final        = torch.nn.Linear(channel_list[-1],self.output_dim)

    def forward(self,x,target=None):
        x = x.reshape(x.size(0),-1)
        x = self.layer(x)
        x = self.final(x)
        x = nn.Sigmoid()(x)
        x = x.reshape(self.final_shape)
        if target is None:
            return x
        else:
            loss = self._loss(x,target)
            return loss,x

class MLPConfig:
    def __init__(self,config):
        self.channel_list  = config['channel_list']

    def __name__(self):

        return "MLP:"+",".join([str(c) for c in self.channel_list])
    def __call__(self,image_type,curve_type,**kargs):
        image_type.force_field('real')
        curve_type.force_field('real')
        model = MLP_Forward(image_type,curve_type,self.channel_list,**kargs)
        return model

mlp_test = MLPConfig({'channel_list':[256,512,1024,512,256]})
mlp_deep = MLPConfig({'channel_list':[256,512,1024,2048,1024,512,256,128,64,128,64,128,64,128]})

if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
