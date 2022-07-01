import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as real_models
from mltool.universal_model_util import relu2leackrelu
import numpy as np

from .model import BaseModel,Forward_Model,conv_shrink

#### ordinary Resnet Class
##### modified pooling layer
##### origin Resnet return a logits directly after Linear Layer! It need post sigmoid()
class ResnetS(Forward_Model):
    def __init__(self,image_type,curve_type,backbone,model_field='real',outchannel=512,
                      input_to_complex="padzero",final_pool=False,first_pool=False,final_double=True, **kargs):
        super(ResnetS, self).__init__(image_type,curve_type,**kargs)
        self.backbone        = backbone
        self.outchannel      = outchannel
        self.model_field     = model_field
        self.input_to_complex= input_to_complex
        self.backbone.conv1  = self.backbone.conv1.__class__(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.maxpool= self.backbone.maxpool.__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.backbone.avgpool= self.backbone.avgpool.__class__(final_pool) if final_pool else nn.Identity()
        if 'Adaptive' in self.backbone.avgpool.__class__.__name__:
            self.c_after_conv    = self.outchannel*final_pool*final_pool if final_pool else self.outchannel
        else:
            self.c_after_conv    = self.outchannel*self.input_dim //64 //(4 if final_pool else 1) //(4 if first_pool else 1)
        self.output_dim      = self.output_dim//2 if (model_field == 'complex' and final_double) else self.output_dim
        self.backbone.fc     = self.backbone.fc.__class__(in_features=self.c_after_conv, out_features=self.output_dim)
        self.tail_layer      = nn.Identity()
        self.merge           =  (model_field == 'complex') and (curve_type.data_shape[-1]!=2)
        print(f"your model field is {model_field}")
        print(f"your curve shape is {curve_type.data_shape}")
        self.debugQ = False
        self.flatten_before_fc = True
    def debug(self,info):
        if not self.debugQ:return
        print(info)
    def forward(self, x,target=None):
        self.debug(x.shape);
        if self.input_field == 'real' and self.model_field == 'complex':
        	if self.input_to_complex== "padzero":
        		x = torch.stack([x,torch.zeros_like(x)],dim=-1)

        #x=self.backbone(x)  ;#print(x.shape)
        self.debug(x.shape);x = self.backbone.conv1(x)
        self.debug(x.shape);x = self.backbone.bn1(x)
        self.debug(x.shape);x = self.backbone.relu(x)
        self.debug(x.shape);x = self.backbone.maxpool(x)
        self.debug(x.shape);x = self.backbone.layer1(x)
        self.debug(x.shape);x = self.backbone.layer2(x)
        self.debug(x.shape);x = self.backbone.layer3(x)
        self.debug(x.shape);x = self.backbone.layer4(x)
        self.debug(x.shape);x = self.backbone.avgpool(x)
        if self.flatten_before_fc:self.debug(x.shape);x = x.reshape(x.size(0), -1,2) if (x.shape[-1]==2) and (self.model_field=='complex') else x.reshape(x.size(0), -1)
        self.debug(x.shape);x = self.backbone.fc(x)
        if self.merge:x = x.norm(dim=-1)
        self.debug(x.shape);x=self.tail_layer(x)
        self.debug(x.shape);x=x.reshape(self.final_shape)
        self.debug(x.shape);
        #
        if target is None:return x
        else:loss = self._loss(x,target)
        return loss,x
class ResnetSWrapper:
    def __init__(self,name,model_field,backbone_str,outchannel,tail_type,post_process_type=None,**kargs):
        self.__name__ = name
        self.backbone_str = backbone_str
        self.model_field = model_field
        self.tail_type= tail_type
        self.post_process= post_process_type
        self.kargs    = kargs
        self.outchannel = outchannel

    def compile_backbone(self):
        backbone_str = self.backbone_str
        model_field  = self.model_field
        models  = cplx_models if model_field == 'complex' else real_models
        if   backbone_str == "resnet18":
            return models.resnet18()
        elif backbone_str == "resnet34":
            return models.resnet34()
        elif backbone_str == "resnet50":
            return models.resnet50()
        elif backbone_str== "resnet101":
            return models.resnet101()
        else:
            raise NotImplementedError

    def modify_tail(self,model):
        type = self.tail_type
        if type is None:return
        type_list = type.split('.')
        s_c = model.c_after_conv
        e_c   = model.output_dim
        c_list= [s_c]
        for i,code in enumerate(type_list):
            if ":" not in code:continue
            c_now = code.split(':')[1]
            if c_now=="half":
                c_list.append(c_list[i]//2)
            else:
                c_list.append(int(c_now))
        c_list.append(e_c)
        modules = []
        for i,code in enumerate(type_list):
            modules.append(model.backbone.fc.__class__(in_features=c_list[i], out_features=c_list[i+1]))
            if   code[1]=='S':
                modules.append(nn.Sigmoid())
            elif code[1]=='T':
                modules.append(nn.Tanh())
        model.tail_layer  = nn.Sequential(*modules)
        model.backbone.fc = nn.Identity()

    def modify_post_process(self,model):
        if self.post_process is None:return
        for post_process in self.post_process.split(':'):
            if post_process == "leakyrelu":
                model.backbone=relu2leackrelu(model.backbone)
            elif post_process == "complex_realize":
                assert self.model_field == 'complex'
                model = nlt.SemiToReal(model)
                model.backbone.conv1 = nlt.SemiToReal_Conv2d_first_layer(model.backbone.conv1)
            elif post_process == "fullcomplex":
                assert self.model_field == 'complex'
                model = nlt.SemiToFull_Conv2d(model)
                model = nlt.SemiToFull_Linear(model)
                model = nlt.SemiToFull_FunctionLayer(model)


    def __call__(self,image_type,curve_type):
        backbone = self.compile_backbone()
        model    = ResnetS(image_type,curve_type,backbone,model_field=self.model_field,outchannel=self.outchannel,**self.kargs)
        _        = self.modify_tail(model)
        _        = self.modify_post_process(model)
        return model

Resnet18S  = ResnetSWrapper("Resnet18S",'real' ,'resnet18', 512,None)
Resnet34S  = ResnetSWrapper("Resnet34S",'real' ,'resnet34', 512,None)
Resnet50S  = ResnetSWrapper("Resnet50S",'real' ,'resnet50',2048,None)
Resnet101S = ResnetSWrapper("Resnet101S",'real','resnet101',2048,None)

##### FN flag will use deep dense fc layer at the end
Resnet18SFN = ResnetSWrapper("Resnet18SFN",'real' ,'resnet18', 512,'LS:2000.LS')
Resnet34SFN = ResnetSWrapper("Resnet34SFN",'real' ,'resnet34', 512,'LS:2000.LS')
Resnet50SFN = ResnetSWrapper("Resnet50SFN",'real' ,'resnet50',2048,'LS:2000.LS')


##### splite the fc out of backbone, and become  tail_layer
Resnet18KSFN     =      ResnetSWrapper("Resnet18KSFN",'real' ,'resnet18', 512,'LS')
Resnet18KSFNT    =     ResnetSWrapper("Resnet18KSFNT",'real' ,'resnet18', 512,'LT:half.LS')
Resnet18KSFNTTend= ResnetSWrapper("Resnet18KSFNTTend",'real' ,'resnet18', 512,'LT:half.LT')
Resnet18KSFNLeakReLU = ResnetSWrapper("Resnet18KSFNLeakReLU",'real' ,'resnet18', 512,'LT:half.LS',post_process_type='leakyrelu')
Resnet18KSFNLeakReLUTend= ResnetSWrapper("Resnet18KSFNLeakReLUTend",'real' ,'resnet18', 512,'LT:half.LT',post_process_type='leakyrelu')
from mltool.ModelArchi.SymmetryCNN import cnn2symmetrycnn
P4Z2_Resnet18KSFNLeakReLUTend = lambda *arg,**kargs: cnn2symmetrycnn(Resnet18KSFNLeakReLUTend(*arg,**kargs),type='P4Z2')
Z2_Resnet18KSFNLeakReLUTend = lambda *arg,**kargs: cnn2symmetrycnn(Resnet18KSFNLeakReLUTend(*arg,**kargs),type='Z2')

P4Z2auto_Resnet18KSFNLeakReLUTend = lambda *arg,**kargs: cnn2symmetrycnn(Resnet18KSFNLeakReLUTend(*arg,**kargs),type='P4Z2',active_symmetry_fix='even')
Z2auto_Resnet18KSFNLeakReLUTend   = lambda *arg,**kargs: cnn2symmetrycnn(Resnet18KSFNLeakReLUTend(*arg,**kargs),type='Z2',active_symmetry_fix='even')

def Abs_Z2_Resnet18KSFNLeakReLUTend(*arg,**kargs):
    model                       = Z2_Resnet18KSFNLeakReLUTend(*arg,**kargs)
    model.backbone.avgpool      = nn.AdaptiveAvgPool2d(1)
    model.tail_layer[0]         = nn.Linear(in_features=512, out_features=1024, bias=True)
    return model
def Abs_Z2auto_Resnet18KSFNLeakReLUTend(*arg,**kargs):
    model                   = Z2auto_Resnet18KSFNLeakReLUTend(*arg,**kargs)
    model.backbone.avgpool      = nn.AdaptiveAvgPool2d(1)
    model.tail_layer[0]         = nn.Linear(in_features=512, out_features=1024, bias=True)
    return model

### notice the up-low class relation
Resnet34KSFN     =      ResnetSWrapper("Resnet34KSFN",'real' ,'resnet34', 512,'LS')
Resnet34KSFNT    =     ResnetSWrapper("Resnet34KSFNT",'real' ,'resnet34', 512,'LT:half.LS')
Resnet34KSFNTTend= ResnetSWrapper("Resnet34KSFNTTend",'real' ,'resnet34', 512,'LT:half.LT')
Resnet34KSFNLeakReLU = ResnetSWrapper("Resnet34KSFNLeakReLU",'real' ,'resnet34', 512,'LT:half.LS',
                                        post_process_type='leakyrelu')
Resnet34KSFNLeakReLUTend= ResnetSWrapper("Resnet34KSFNLeakReLUTend",'real' ,'resnet34', 512,'LT:half.LT',
                                        post_process_type='leakyrelu')

from mltool.ModelArchi.MyConv2d import SimpleResNetConfig,BottleneckV0
class MyResnet_V1(Forward_Model):
    def __init__(self,image_type,curve_type,layerconfig,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        #layerconfig = [[1,None,None]]+[[32,2,2],[64,2,2],[128,1,2],[256,1,2]]
        self.backbone   = SimpleResNetConfig(BottleneckV0,layerconfig,relu=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layerconfig=layerconfig
        self.fc         = nn.Sequential(nn.Linear(self.layerconfig[-1][0],self.layerconfig[-1][0]//2),
                                  nn.Tanh(),
                                  nn.Linear(self.layerconfig[-1][0]//2,self.output_dim),
                                  nn.Sigmoid()
                                 )
    def forward(self,x):
        x = self.backbone(x)
        x = x.reshape(x.size(0),self.layerconfig[-1][0])
        x = self.fc(x)
        x=x.reshape(self.final_shape)
        return x
class MyResnet_V2(Forward_Model):
    def __init__(self,image_type,curve_type,layerconfig,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        #layerconfig = [[1,None,None]]+[[32,2,2],[64,2,2],[128,1,2],[256,1,2]]
        self.backbone   = SimpleResNetConfig(BottleneckV0,layerconfig,relu=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layerconfig=layerconfig
        self.fc         = nn.Sequential(nn.Linear(self.layerconfig[-1][0],self.layerconfig[-1][0]//2),
                                  nn.Tanh(),
                                  nn.Linear(self.layerconfig[-1][0]//2,self.output_dim),
                                  nn.Sigmoid()
                                 )
        self.pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.reshape(x.size(0),self.layerconfig[-1][0])
        x = self.fc(x)
        x=x.reshape(self.final_shape)
        return x

class MyResnet_V1_Wrapper(object):
    __name__ = "ResnetMya1"
    def __init__(self,layerconfig,name=None):
        self.layerconfig = layerconfig
        if name is not None:self.__name__   = name
    def __call__(self,image_type,curve_type):
        return MyResnet_V1(image_type,curve_type,self.layerconfig)
class MyResnetTend_V1_Wrapper(MyResnet_V1_Wrapper):
    __name__ = "ResnetMya1Tend"
    def __call__(self,image_type,curve_type):
        model   =  MyResnet_V1(image_type,curve_type,self.layerconfig)
        model.fc= nn.Sequential(nn.Linear(model.layerconfig[-1][0],model.layerconfig[-1][0]//2),
                                  nn.Tanh(),
                                  nn.Linear(model.layerconfig[-1][0]//2,model.output_dim),
                                  nn.Tanh()
                                 )
        return model
class MyResnet_V2_Wrapper(object):
    __name__ = "ResnetMya1"
    def __init__(self,layerconfig,name=None):
        self.layerconfig = layerconfig
        if name is not None:self.__name__   = name
    def __call__(self,image_type,curve_type):
        return MyResnet_V2(image_type,curve_type,self.layerconfig)

ResnetMya1     =  MyResnet_V1_Wrapper([[1,None,None]]+[[32,2,2],[64,2,2],[128,1,2],[256,1,2]],name='ResnetMya1')
ResnetMya1Tend =  MyResnetTend_V1_Wrapper([[1,None,None]]+[[32,2,2],[64,2,2],[128,1,2],[256,1,2]],name='ResnetMya1Tend')
ResnetMya_Simple= MyResnet_V2_Wrapper([[1,None,None]]+[[32,2,2],[64,2,2],[128,1,2]],name='ResnetMya_Simple')
if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
