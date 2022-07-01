import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .model import BaseModel,Forward_Model,conv_shrink

################################
########## Densenet ##############
################################


class DensenetS(Forward_Model):
    #def __init__(self,out_put_fea,output_field,backbone,final_pool=False,**kargs):
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super(DensenetS, self).__init__(image_type,curve_type,**kargs)
        assert model_field == 'real'  # !!! no cplx version so far
        self.backbone                 = backbone
        self.backbone.features.conv0  = self.backbone.features.conv0.__class__(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.backbone.features.pool0  = self.backbone.features.pool0.__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.relu                     = self.backbone.features.relu0.__class__(inplace=True)
        self.avgpool                  = self.backbone.features.pool0.__class__(kernel_size=2,stride=1) if final_pool else nn.Identity()
        shrink_coef                   = conv_shrink(self.backbone.features)*conv_shrink(self.avgpool)
        self.s_after_conv             = self.outchannel*self.input_dim //(shrink_coef**2)
        #self.s_after_conv             = self.outchannel*self.in_size //64 //(4 if final_pool else 1)
        self.backbone.classifier      = self.backbone.classifier.__class__(in_features=self.s_after_conv, out_features=self.output_dim)

    def forward(self, x,target=None):
        x = self.backbone.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, self.s_after_conv)
        x = self.backbone.classifier(x)
        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class Densenet121S(DensenetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        backbone=models.densenet121()
        self.outchannel  = 1024
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)

class Densenet169S(DensenetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        backbone=models.densenet169()
        self.outchannel  = 1664
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)

class Densenet201S(DensenetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        backbone=models.densenet201()
        self.outchannel  = 1920
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)


if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
