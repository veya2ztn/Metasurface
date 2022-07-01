import torch
import torch.nn as nn
import torch.nn.functional as F
import mltool.ModelArchi.resnet1D as real_models
import mltool.torch_complex.resnet1D as cplx_models
from .model import BaseModel,Inverse_Model,conv_shrink
from .tail_layer import *
#

class U1Tanh(nn.Module):
    def __init__(self,sloop=1):
        super().__init__()
        self.sloop=1
    def forward(self,x):
        x = nn.Tanh()(self.sloop*x)
        x = (x+1)/2
        return x
####### inverse base classes ###############
class ResnetS1D(Inverse_Model):
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super(ResnetS1D, self).__init__(image_type,curve_type,**kargs)
        self.backbone        = backbone
        self.backbone.conv1  = self.backbone.conv1.__class__(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.maxpool= self.backbone.maxpool.__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.backbone.avgpool= torch.nn.AdaptiveMaxPool1d(4)
        self.s_after_conv    = self.outchannel*4
        self.backbone.fc     = self.backbone.fc.__class__(in_features=self.s_after_conv, out_features=self.output_dim)
    def forward(self, x,target=None):
        x=self.backbone(x)  ;#print(x.shape)
        x=x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class ResnetM1D(Inverse_Model):
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.backbone        = backbone
        self.backbone.conv1  = self.backbone.conv1.__class__(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.maxpool= self.backbone.maxpool.__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.backbone.avgpool= torch.nn.AdaptiveMaxPool1d(1)
        self.tail_layer = torch.nn.Identity()

    def forward(self, x,target=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.avgpool(x)
        x = x.reshape(self.final_shape)
        x = self.tail_layer(x)
        #x = torch.nn.Sigmoid()(x)## old version
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class ResnetM1D_old(Inverse_Model):
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.backbone        = backbone
        self.backbone.conv1  = self.backbone.conv1.__class__(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.maxpool= self.backbone.maxpool.__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.backbone.avgpool= torch.nn.AdaptiveMaxPool1d(1)
        self.backbone.fc     = torch.nn.Identity()
    def forward(self, x,target=None):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.avgpool(x)
        x = x.reshape(self.final_shape)
        x = torch.nn.Sigmoid()(x)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x


class Resnet18M_a1(ResnetM1D):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models  = cplx_models if model_field == 'complex' else real_models
        backbone= models.resnet18()
        self.outchannel  = 256
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)

class Resnet18_a1MConfig:
    def __init__(self,U1sloop,**kwargs):
        self.U1sloop     = U1sloop
        self.__name__    = f"Resnet18M_a1s{U1sloop}"
    def __call__(self,image_type,curve_type):
        model = Resnet18M_a1(image_type,curve_type)
        model.final_layer = U1Tanh(self.U1sloop)
        return model

Resnet18M_a1s10=Resnet18_a1MConfig(10)
Resnet18M_a1s20=Resnet18_a1MConfig(20)
Resnet18M_a1s30=Resnet18_a1MConfig(30)
Resnet18M_a1s40=Resnet18_a1MConfig(40)
Resnet18M_a1s50=Resnet18_a1MConfig(50)

####### upsample - transposeconv method

from mltool.ModelArchi.TransConv2d import TransposeBottleneckV1,UpSampleResNet,FPNUpSample

class TransResUp(Inverse_Model):
    def __init__(self,image_type,curve_type,layerconfig,filtePara='sigmoid',noise_dim=0,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.noise_dim   = noise_dim
        if self.noise_dim:layerconfig[0][0]+=self.noise_dim
        self.backbone        = UpSampleResNet(TransposeBottleneckV1,layerconfig)
        if filtePara is None:
            self.tail_layer = torch.nn.Identity()
        elif filtePara=='sigmoid':
            self.tail_layer = torch.nn.Sigmoid()
        else:
            self.tail_layer = U1Tanh(filtePara)
        self.final_batchnorm= torch.nn.BatchNorm2d(1)
    def forward(self, x,target=None):
        x = x.permute(0,2,1).unsqueeze(-1)

        if self.noise_dim:
            batch,c,w,h = x.shape
            noise = torch.randn((batch,self.noise_dim,w,h)).to(x.device)
            x = torch.cat([x,noise],1)

        x = self.backbone(x)  ;#print(x.shape)
        x = self.final_batchnorm(x)
        x = x.reshape(self.final_shape)
        x = self.tail_layer(x)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class TransResUp_config:
    def __init__(self,name,layerconfig,noise=0,**kwargs):
        self.layerconfig     = layerconfig
        self.__name__        = name
        self.noise=noise
    def __call__(self,image_type,curve_type,filtePara='sigmoid',**kwargs):
        layerconfig = [[curve_type.data_shape[-1],None,None]]+self.layerconfig
        model = TransResUp(image_type,curve_type,layerconfig,filtePara=filtePara,noise_dim=self.noise)
        return model

TransResUp_a0 = TransResUp_config('TransResUp_a0',[[128,1,2],[64,2,1],[32,2,2],[16,2,1],[4,2,2],[1,2,1]])
TransResUp_a1 = TransResUp_config('TransResUp_a1',[[64,2,2],[32,2,1],[16,2,2],[8,2,1],[4,2,2],[1,2,1]])
TransResUp_b1 = TransResUp_config('TransResUp_b1',[[64,2,2],[32,3,1],[16,4,2],[8,5,1],[4,3,2],[1,2,1]])
TransResUp_c1 = TransResUp_config('TransResUp_c1',[[64,1,2],[32,1,1],[16,1,2],[8,1,1],[4,1,2],[2,1,1],[1,1,2]])
TransResUp_d1 = TransResUp_config('TransResUp_d1',[[64,2,2],[96,2,1],[128,2,2],[64,2,1],[32,2,2],[16,2,1],[8,2,1],[4,2,1],[1,1,1]])

TransResUp_a0N50 = TransResUp_config('TransResUp_a0N50',[[128,1,2],[64,2,1],[32,2,2],[16,2,1],[4,2,2],[1,2,1]],noise=50)
TransResUp_a1N50 = TransResUp_config('TransResUp_a1N50',[[64,2,2],[32,2,1],[16,2,2],[8,2,1],[4,2,2],[1,2,1]],noise=50)

###### FPN transpose ############

class FPNTransResUp(Inverse_Model):
    def __init__(self,image_type,curve_type,layerconfig,filtePara='sigmoid',**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.backbone        = FPNUpSample(TransposeBottleneckV1,layerconfig)
        if filtePara=='sigmoid':
            self.tail_layer = torch.nn.Sigmoid()
        else:
            self.tail_layer = U1Tanh(filtePara)
        self.finalpool = nn.AdaptiveAvgPool2d(16)
        self.final_batchnorm= torch.nn.BatchNorm2d(1)
    def forward(self, x,target=None):
        x = x.permute(0,2,1).unsqueeze(-1)
        p_list=self.backbone(x)  ;#print(x.shape)
        p_list=[self.finalpool(self.tail_layer(p)) for p in p_list]
        x = torch.cat(p_list,1).mean(1,keepdim=True)
        x = self.final_batchnorm(x)
        x=x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class FPNTransResUp_config:
    def __init__(self,name,layerconfig,**kwargs):
        self.layerconfig     = layerconfig
        self.__name__        = name
    def __call__(self,image_type,curve_type,filtePara='sigmoid',**kwargs):
        layerconfig = [[curve_type.data_shape[-1],None,None]]+self.layerconfig
        model = FPNTransResUp(image_type,curve_type,layerconfig,filtePara=filtePara)
        return model

FPNTransResUp_a1 = FPNTransResUp_config('FPNTransResUp_a1',[[64,2,2],[32,2,1],[16,2,2],[8,2,2]])
FPNTransResUp_b1 = FPNTransResUp_config('FPNTransResUp_a1',[[32,2,2],[64,2,1],[32,2,2],[16,2,2]])





# class Resnet18S1D(ResnetS1D):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#         models  = cplx_models if model_field == 'complex' else real_models
#         backbone= models.resnet18()
#         self.outchannel  = 512
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#
# class Resnet34S1D(ResnetS1D):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#         models  = cplx_models if model_field == 'complex' else real_models
#         backbone=models.resnet34()
#         self.outchannel  = 512
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#
# class Resnet50S1D(ResnetS1D):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#
#         models  = cplx_models if model_field == 'complex' else real_models
#         backbone=models.resnet50()
#         self.outchannel  = 2048
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#
# class Resnet101S1D(ResnetS1D):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#
#         models  = cplx_models if model_field == 'complex' else real_models
#         backbone=models.resnet101()
#         self.outchannel  = 2048
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#
# class ResnetWrapper(ResnetS1D):
#     def __init__(self,out_put_fea,output_field,backbone,outchannel,**kargs):
#         backbone        = backbone
#         self.outchannel = outchannel
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#
# class ResnetConfig1D:
#     def __init__(self,config,**kwargs):
#
#         block_type_n= config['resnet_block']
#         block_list  = config['resnet_block_list']
#         if block_type_n == 'BasicBlock':
#             block_type = models.BasicBlock
#             self.outchannel  = 512
#         elif block_type_n == 'Bottleneck':
#             block_type     = models.Bottleneck
#             self.outchannel  = 512*4
#         else:
#             raise NotImplementedError
#         self.backbone=models.ResNet(block_type,block_list, **kwargs)
#
#     def __call__(self,out_put_fea,output_field):
#         backbone  = self.backbone
#         outchannel= self.outchannel
#         model = ResnetWrapper(out_put_fea,output_field,backbone,outchannel)
#         return model
#
# class Forward_Tail:
#     def forward(self,x,target=None):
#         x = self.backbone(x)#(...,256,x)
#         x = self.tail_layer(x)#(...,256)
#         x = x.view(self.final_shape)#(...,1,16,16)
#         if target is not None:
#             loss = self._loss(x,target)
#             return loss,x
#         return x
#
# class INV_CPLX_ResNet18(Forward_Tail,Resnet18S):
#     def __init__(self,image_type,curve_type,tail_layer=TailNorm,**kargs):
#         super().__init__(image_type,curve_type,model_field='complex',first_pool=True,**kargs)
#         self.tail_layer = tail_layer()
#
# class INV_CPLX_ResNet101(Forward_Tail,Resnet101S):
#     def __init__(self,image_type,curve_type,tail_layer=TailNorm,**kargs):
#         super().__init__(image_type,curve_type,model_field='complex',first_pool=True,**kargs)
#         self.tail_layer = tail_layer()


if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
