import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Tandem_Model
from .INV_RS1D import Resnet18M_a1,TransResUp_d1,U1Tanh,TransResUp_a0N50,TransResUp_a1N50,TransResUp_a0,TransResUp_a1
from .INV_RS2D import MLUPInverse
from .FWD_RESNAT import ResnetNature_a1,ResnetNature_a4
from .FWD_SQ2D import SqueezeNet1S
from .FWD_RS2D import Resnet18S,Resnet18SFN
#ResnetNature_a1_pre_train = "/media/tianning/DATA/metasurface/checkpoints_old_name_rule/ResnetNature_a1.lr=0.0010000.(128).(Sample.unisample128.norm).none_norm.on97000/trail00/best/epoch-0140|fl-0.027664|rl-0.027675"

class TandemModel_1_old(Tandem_Model):
    def __init__(self,image_type,curve_type,pre_train_forward='auto',pre_train_inverse=False,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.Curve2ImageNetwork=Resnet18M_a1(image_type,curve_type)
        self.Image2CurveNetwork=ResnetNature_a1(image_type,curve_type)
        if pre_train_forward:
            if pre_train_forward == 'auto':pre_train_forward=ResnetNature_a1_pre_train
            print(f'use pre_train_weight:{pre_train_forward}')
            self.Image2CurveNetwork.load_state_dict(torch.load(pre_train_forward)['state_dict'])
    def forward(self,curve,target=None):
        image = self.Curve2ImageNetwork(curve)
        curve = self.Image2CurveNetwork(image)
        curve = 1-curve # for ResnetNature_a1 on our dataset
        return curve,image

import os
class TandemWrapper(Tandem_Model):
    def __init__(self,image_type,curve_type,config,for_PTNQ=False,filtePara='sigmoid',**kargs):
        super().__init__(image_type,curve_type,**kargs)
        name,inverse_model_type,forward_model_type,forward_model_pre_train,forward_model_pre_train_PTN=config
        self.Curve2ImageNetwork=inverse_model_type(image_type,curve_type)
        self.Curve2ImageNetwork.tail_layer = torch.nn.Identity()
        if filtePara is None:
            self.filter_layer = torch.nn.Identity()
        elif filtePara=='sigmoid':
            self.filter_layer = torch.nn.Sigmoid()
        else:
            self.filter_layer = U1Tanh(filtePara)
        self.Image2CurveNetwork=forward_model_type(image_type,curve_type)
        if for_PTNQ:
           self.Image2CurveNetwork.load_state_dict(torch.load(forward_model_pre_train_PTN)['state_dict'])
        else:
           self.Image2CurveNetwork.load_state_dict(torch.load(forward_model_pre_train)['state_dict'])
    def forward(self,curve,target=None):
        image = self.Curve2ImageNetwork(curve)
        image = self.filter_layer(image)
        curve = self.Image2CurveNetwork(image)
        curve = 1-curve # for ResnetNature_a1 on our dataset
        return curve,image


class TandemConfig:
    def __init__(self,config):
        self.config     = config
        self.__name__   = config[0]
    def __call__(self,image_type,curve_type,for_PTNQ=False,filtePara=10,**kargs):
        model = TandemWrapper(image_type,curve_type,self.config,for_PTNQ=for_PTNQ,filtePara=filtePara)
        return model

SMSDatasetB1NES32_BASE = "/data/Metasurface/checkpoints/SMSDatasetB1NES32,curve,simple.on97000"
ResnetNature_a4_pre_train = os.path.join(SMSDatasetB1NES32_BASE,"ResnetNature_a4/22_46_03-seed-84771/best/best_MAError_0.0358")
Resnet18SFN_pre_train     = os.path.join(SMSDatasetB1NES32_BASE,"Resnet18SFN/09_05_27-seed-16843/best/best_MsaError_0.0045")
SqueezeNet1S_pre_train     = os.path.join(SMSDatasetB1NES32_BASE,"SqueezeNet1S/21_46_00-seed-57457/best/best_MAError_0.0498")
PTNSMSDatasetB1NES32_BASE = "/data/Metasurface/checkpoints/PTNSMSDatasetB1NES32,curve,simple.on9000"
PTNResnetNature_a4_pre_train = os.path.join(PTNSMSDatasetB1NES32_BASE,"ResnetNature_a4/23_35_53-seed-13831/best/best_MAError_0.0168")
PTNResnet18SFN_pre_train     = os.path.join(PTNSMSDatasetB1NES32_BASE,"Resnet18SFN/08_46_39-seed-90807/best/best_MSError_0.0051")
PTNSqueezeNet1S_pre_train     = os.path.join(PTNSMSDatasetB1NES32_BASE,"SqueezeNet1S/21_24_46-seed-63948/best/best_MSError_0.0051")

ResnetNature_a1_pre_train = "/data/Metasurface/warmup/ResnetNature_a1/fl-0.027664"
PTNSMSDatasetB1NES128_BASE= "/data/Metasurface/checkpoints/PTNSMSDatasetB1NES128,curve,simple.on9000"
PTNResnetNature_a1_pre_train=os.path.join(PTNSMSDatasetB1NES128_BASE,"ResnetNature_a1/08_22_23-seed-22288/best/best_MSError_0.0055")

TandemModel_a1 = TandemConfig(["TandemModel_a1",Resnet18M_a1,ResnetNature_a1,ResnetNature_a1_pre_train,PTNResnetNature_a1_pre_train])
TandemModel_a2 = TandemConfig(["TandemModel_a2",Resnet18M_a1,ResnetNature_a4,ResnetNature_a4_pre_train,PTNResnetNature_a4_pre_train])
TandemModel_a3 = TandemConfig(["TandemModel_a3",Resnet18M_a1,Resnet18SFN,Resnet18SFN_pre_train,PTNResnet18SFN_pre_train])
TandemModel_a4 = TandemConfig(["TandemModel_a4",Resnet18M_a1,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])

TandemModel_b1 = TandemConfig(["TandemModel_b1",MLUPInverse,ResnetNature_a1,ResnetNature_a1_pre_train,PTNResnetNature_a1_pre_train])
TandemModel_b2 = TandemConfig(["TandemModel_b2",MLUPInverse,ResnetNature_a4,ResnetNature_a4_pre_train,PTNResnetNature_a4_pre_train])
TandemModel_b3 = TandemConfig(["TandemModel_b3",MLUPInverse,Resnet18SFN,Resnet18SFN_pre_train,PTNResnet18SFN_pre_train])
TandemModel_b4 = TandemConfig(["TandemModel_b4",MLUPInverse,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])

#TandemModel_c1 = TandemConfig(["TandemModel_c1",TransResUp_d1,ResnetNature_a1,ResnetNature_a1_pre_train,PTNResnetNature_a1_pre_train])
TandemModel_c2 = TandemConfig(["TandemModel_c2",TransResUp_d1,ResnetNature_a4,ResnetNature_a4_pre_train,PTNResnetNature_a4_pre_train])
TandemModel_c3 = TandemConfig(["TandemModel_c3",TransResUp_d1,Resnet18SFN,Resnet18SFN_pre_train,PTNResnet18SFN_pre_train])
TandemModel_c4 = TandemConfig(["TandemModel_c4",TransResUp_d1,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])

TandemModel_c0N50 = TandemConfig(["TandemModel_c0N50",TransResUp_a0N50,Resnet18SFN,Resnet18SFN_pre_train,PTNResnet18SFN_pre_train])
TandemModel_c1N50 = TandemConfig(["TandemModel_c1N50",TransResUp_a1N50,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])
TandemModel_c0    = TandemConfig(["TandemModel_c0",TransResUp_a0,Resnet18SFN,Resnet18SFN_pre_train,PTNResnet18SFN_pre_train])
TandemModel_c5    = TandemConfig(["TandemModel_c5",TransResUp_a1,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])
