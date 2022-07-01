import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import BaseModel,Forward_Model
import numpy as np
from mltool.ModelArchi.NLP import BiLSTM,BiLSTM2

class conv_module_type1(nn.Module):
    def __init__(self,inplane,ouplane):
        super().__init__()
        self.layer= nn.Sequential(
            nn.Conv2d(inplane, ouplane, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(ouplane)
            )
    def forward(self,x):
        return self.layer(x)
class conv_module_type2(nn.Module):
    def __init__(self,inplane,midplane,relu=nn.LeakyReLU):
        super().__init__()
        self.relu = relu()
        self.layer= nn.Sequential(
            nn.Conv2d(inplane, midplane, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(midplane),
            self.relu,
            nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(midplane),
            self.relu,
            nn.Conv2d(midplane, inplane, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(inplane),
            )
    def forward(self,x):
        y = self.layer(x)
        x = x+y
        return x




class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps):
        super(TimeDistributed, self).__init__()
        self.layers = nn.ModuleList([layer for i in range(time_steps)])
    def forward(self, x):
        batch_size, time_steps, H, W = x.size()
        x = x.reshape(batch_size, time_steps, H*W)
        output = []
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :])
            output.append(output_t)
        output   = torch.stack(output, 1)
        return output




class TimeDistributed1(nn.Module):
    def __init__(self, _in, _out, time_steps,drop_out_rate=0.5):
        super(TimeDistributed1, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(
                                         nn.Linear(_in,_out),
                                         torch.nn.Dropout(drop_out_rate),
                                     )
                                    for i in range(time_steps)])
    def forward(self, x):
        batch_size, time_steps, H, W = x.size()
        x = x.reshape(batch_size, time_steps, H*W)
        output = []
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :])
            output.append(output_t)
        output   = torch.stack(output, 1)
        return output
class MGRU(nn.Module):
    def __init__(self,_int, _out):
        super().__init__()
        self.GRU=nn.GRU(_int, _out)
    def forward(self,x):
        x = self.GRU(x)[0]
        x = x.reshape(x.size(0),-1)
        return x
class GRUTD(nn.Module):
    def __init__(self,_in,_int,_out):
        super().__init__()
        self.timedis=TimeDistributed(nn.Linear(_in,_int), time_steps = _in)
        self.GRU    =nn.GRU(_int, _out)
    def forward(self, x):
        x = self.timedis(x)
        x = self.GRU(x)[0]

        return x
class U1Tanh(nn.Module):
    def forward(self,x):
        x = nn.Tanh()(x)
        x = (x+1)/2
        return x

class ResnetNature(Forward_Model):
    def __init__(self,image_type,curve_type,model_field='real',final_pool=False,first_pool=False,**kargs):
        super(ResnetNature, self).__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,128)
        self.layer3 =conv_module_type2(256,128)
        self.timedis=TimeDistributed(nn.Linear(256,128), time_steps = 256)
        self.GRU    =nn.GRU(128, 10)
        self.fc1    =nn.Sequential(
                        nn.Linear(2560,200),
                        nn.Sigmoid()
                    )
        self.fc2    =nn.Sequential(
                        nn.Linear(200,128),
                        nn.Sigmoid()
                    )

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = x+y
        x = self.relu(x)
        y = self.layer3(x)
        x = x+y
        x = self.relu(x)
        x = self.timedis(x)
        x = self.GRU(x)[0]
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(self.final_shape)
        x = 1-x
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class ResnetNatureModule_a(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU(inplace=True)
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,128)
        self.layer3 =conv_module_type2(256,128)
        self.NLPtail= None
    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = x+y
        x = self.relu(x)
        y = self.layer3(x)
        x = x+y
        x = self.relu(x)

        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        x = 1-x
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_a0(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,300,128),
            nn.Linear(128,400),
            nn.Sigmoid(),
            nn.Linear(400,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_a1(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_a2(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,300,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_a3(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,600,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_a4(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,drop_out_rate=0,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            conv_module_type1(256,16),
            nn.AdaptiveAvgPool2d(4),
            TimeDistributed1(16,32, time_steps = 16),
            BiLSTM(32,64,32),
            nn.Linear(32,64),
            torch.nn.Dropout(drop_out_rate),
            nn.Tanh(),
            nn.Linear(64,self.output_dim),
            torch.nn.Dropout(drop_out_rate),
            U1Tanh(),
        )
class ResnetNature_a5(ResnetNatureModule_a):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM2(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,self.output_dim),
            nn.Sigmoid(),
        )


class ResnetNatureModule_b(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,128)
        self.layer3 =conv_module_type2(256,128)
        self.NLPtail= None

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = self.relu(x)
        y = self.layer3(x)
        x = self.relu(x)
        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_b0(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,300,128),
            nn.Linear(128,400),
            nn.Sigmoid(),
            nn.Linear(400,128),
            nn.Sigmoid(),
        )
class ResnetNature_b1(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_b2(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,300,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_b3(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,600,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )


class ResnetNatureModule_c(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,256)
        self.layer3 =conv_module_type2(256,256)
        self.layer4 =conv_module_type2(256,128)
        self.layer5 =conv_module_type2(256,128)
        self.NLPtail= None

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = self.relu(x)
        y = self.layer3(x)
        x = self.relu(x)
        y = self.layer4(x)
        x = self.relu(x)
        y = self.layer5(x)
        x = self.relu(x)
        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_c0(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,300,128),
            nn.Linear(128,400),
            nn.Sigmoid(),
            nn.Linear(400,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_c1(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_c2(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,300,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_c3(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(BiLSTM(256,600,128), time_steps = 256),
            MGRU(128,10),
            nn.Linear(2560,200),
            nn.Sigmoid(),
            nn.Linear(200,self.output_dim),
            nn.Sigmoid(),
        )


class ResnetNature_b10(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Tanh(),
        )
class ResnetNature_b11(ResnetNatureModule_b):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_c10(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Tanh(),
        )
class ResnetNature_c11(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Sigmoid(),
        )

class ResnetNature_c10(ResnetNatureModule_c):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Tanh(),
        )

class ResnetNatureModule_d(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,256)
        self.layer3 =conv_module_type2(256,256)
        self.layer4 =conv_module_type2(256,256)
        self.layer5 =conv_module_type2(256,256)
        self.NLPtail= None

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = self.relu(x)
        y = self.layer3(x)
        x = self.relu(x)
        y = self.layer4(x)
        x = self.relu(x)
        y = self.layer5(x)
        x = self.relu(x)
        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_d1(ResnetNatureModule_d):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,128),
            nn.Sigmoid(),
        )

class ResnetNatureModule_e(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11=conv_module_type1(128,256)
        self.layer12=conv_module_type1(128,256)
        self.layer2 =conv_module_type2(256,512)
        self.layer3 =conv_module_type2(256,512)
        self.layer4 =conv_module_type2(256,128)
        self.layer5 =conv_module_type2(256,128)
        self.NLPtail= None

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = self.relu(x)
        y = self.layer3(x)
        x = self.relu(x)
        y = self.layer4(x)
        x = self.relu(x)
        y = self.layer5(x)
        x = self.relu(x)
        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_e1(ResnetNatureModule_e):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,128), time_steps = 256),
            BiLSTM(128,1000,128),
            nn.Linear(128,1000),
            nn.Sigmoid(),
            nn.Linear(1000,128),
            nn.Sigmoid(),
        )
    #
    # class ResnetNature_a0EL1(EnhanceLoss1,ResnetNature_a0):pass
    # class ResnetNature_a1EL1(EnhanceLoss1,ResnetNature_a1):pass
    # class ResnetNature_a2EL1(EnhanceLoss1,ResnetNature_a2):pass
    # class ResnetNature_a3EL1(EnhanceLoss1,ResnetNature_a3):pass
    # class ResnetNature_a0EL2(EnhanceLoss2,ResnetNature_a0):pass
    # class ResnetNature_a1EL2(EnhanceLoss2,ResnetNature_a1):pass
    # class ResnetNature_a2EL2(EnhanceLoss2,ResnetNature_a2):pass
    # class ResnetNature_a3EL2(EnhanceLoss2,ResnetNature_a3):pass
