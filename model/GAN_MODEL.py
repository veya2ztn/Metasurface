import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Generative_Model,Discriminator_Model
from mltool.ModelArchi.MyConv2d import ResNetConfig,BottleneckV0
#ResnetNature_a1_pre_train = "/media/tianning/DATA/metasurface/checkpoints_old_name_rule/ResnetNature_a1.lr=0.0010000.(128).(Sample.unisample128.norm).none_norm.on97000/trail00/best/epoch-0140|fl-0.027664|rl-0.027675"
from .INV_RS1D import Resnet18M_a1,TransResUp_d1,U1Tanh
from .INV_RS2D import MLUPInverse
from .FWD_RESNAT import ResnetNature_a1,ResnetNature_a4
from .FWD_SQ2D import SqueezeNet1S
from .FWD_RS2D import Resnet18S,Resnet18SFN

import os
class DiscrimResnet_a1(Discriminator_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        #image_type and curve_type doesnt work here
        #image_type = (1,16,16)
        layerconfig = [[1,None,None]]+[[32,2,2],[64,2,1],[128,2,2],[64,2,1],[32,2,2],[16,2,1]]
        self.backbone = ResNetConfig(BottleneckV0,layerconfig)
        self.backbone.finalpool= nn.AdaptiveAvgPool2d(2)
        self.fc       = nn.Sequential(nn.Linear(layerconfig[-1][0]*4,1))
    def forward(self,x):

        x = self.backbone(x)# (batch,channel,2,2)
        x = x.reshape(x.size(0),-1)#
        x = self.fc(x)
        return x



class GAN_MODEL(Generative_Model):
    def __init__(self,image_type,curve_type,config,for_PTNQ=False,filtePara='sigmoid',**kargs):
        super().__init__(image_type,curve_type,**kargs)
        name,inverse_model_type,discrim_model_type,forward_model_type,forward_model_pre_train,forward_model_pre_train_PTN=config
        ###############
        self.Image2CurveNetwork=forward_model_type(image_type,curve_type)
        # if for_PTNQ:
        #     self.Image2CurveNetwork.load_state_dict(torch.load(forward_model_pre_train_PTN)['state_dict'])
        #     print(f'load pretrain model from {forward_model_pre_train_PTN}')
        # else:
        #     self.Image2CurveNetwork.load_state_dict(torch.load(forward_model_pre_train)['state_dict'])
        #     print(f'load pretrain model from {forward_model_pre_train}')
        self.Image2CurveNetwork.eval()

        ###############
        data_shape    = list(curve_type.data_shape)
        data_shape[-1]= data_shape[-1]+data_shape[-1]//2
        data_shape    = tuple(data_shape)
        curve_type.reset(curve_type.field,data_shape)
        self.Curve2ImageNetwork=inverse_model_type(image_type,curve_type)
        if filtePara is None:
            self.Curve2ImageNetwork.tail_layer = torch.nn.Identity()
        elif filtePara=='sigmoid':
            self.Curve2ImageNetwork.tail_layer = torch.nn.Sigmoid()
        else:
            self.Curve2ImageNetwork.tail_layer = U1Tanh(filtePara)
        ################

        self.ImageDiscriminator = discrim_model_type(image_type,curve_type)
    def forward(self,curve,target=None):
        curve_random = torch.rand_like(curve)[...,:curve.shape[-1]//2]
        curve = torch.cat([curve,curve_random],-1)
        image = self.Curve2ImageNetwork(curve)
        curve = self.Image2CurveNetwork(image)
        return curve,image

class GAN_Config:
    def __init__(self,config):
        self.config     = config
        self.__name__   = config[0]
    def __call__(self,image_type,curve_type,for_PTNQ=False,filtePara=10,**kargs):
        model = GAN_MODEL(image_type,curve_type,self.config,for_PTNQ=for_PTNQ,filtePara=filtePara)
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

GANModel_aS = GAN_Config(["GANModel_aS",TransResUp_d1,DiscrimResnet_a1,SqueezeNet1S,SqueezeNet1S_pre_train,PTNSqueezeNet1S_pre_train])
