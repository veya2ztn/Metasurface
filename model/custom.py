import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from layers import *
from .model import Inverse_Model,_Model
from .tail_layer import *
from . import INV_RS1D as Inverse_Resnet
from . import INV_SQ1D as Inverse_Squeeze
from . import FWD_SQ2D as squeezenet2D

################################
########## bilinear ############
################################

class BLModel(_Model):
    def __init__(self,branch_left_down,branch_left_up,branch_right):
        '''
            branch_left_up   ---|
                                |--> branch_right
            branch_left_down ---|
        '''
        super().__init__()
        self.bld      = branch_left_down
        self.blu      = branch_left_up
        self.brt      = branch_right
        self.bilinear = torch.nn.Bilinear(self.bld.out_put_fea,self.blu.out_put_fea,
                                          self.brt.in_put_fea)
    def forward(self,x,target=None):
        '''
          input a complex curve (...,2)
        '''
        x  = x.reshape(x.size(0),-1,2)
        re = x[...,0]
        im = x[...,1]
        re = self.branch1(re)
        im = self.branch2(im)
        x  = self.bilinear(re,im)
        x  = self.branch3(x)
        x  = nn.Sigmoid()(x)
        if target is not None:
            loss = self._loss(x,target)
            return loss,x
        return x

from .MLP import MLPModel
class MLPBLinear1(BLModel):
    def __init__(self,ind,cend,outd,cl1,cl2):
        if len(cl1) ==0:
            branch1 = nn.Identity()
            branch2 = nn.Identity()
        else:
            branch1 = MLPModel(ind,cend,cl1)
            branch2 = MLPModel(ind,cend,cl1)
        branch3 = MLPModel(cend,outd,cl2)
        super().__init__(branch1,branch2,branch3)
class MLPBLinear2(BLModel):
    '''
    share weight
    '''
    def __init__(self,ind,cend,outd,cl1,cl2):
        branch1 = MLPModel( ind,cend,cl1)
        branch3 = MLPModel(cend,outd,cl2)
        super().__init__(branch1,branch1,branch3)
class MLPBLConfig:
    def __init__(self,config,**kwargs):
        self.cend   = config['cend']
        self.outd   = config['outd']
        self.cl1    = config['cl1']
        self.cl2    = config['cl2']
        self.ShareWQ= config['isshareweight']

    def __call__(self,out_put_field,output_field='real'):
        ind =out_put_field
        cend=self.cend
        outd=self.outd
        cl1 =self.cl1
        cl2 =self.cl2
        if not self.ShareWQ==1:
            model = MLPBLinear1(ind,cend,outd,cl1,cl2)
        else:
            model = MLPBLinear2(ind,cend,outd,cl1,cl2)
        return model

if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
