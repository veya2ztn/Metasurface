import torch
import torch.nn as nn
import torch.nn.functional as F
import mltool.ModelArchi.squeezenet1D as real_models
from .model import BaseModel,Inverse_Model,conv_shrink
from .tail_layer import *



################################
######## SqueezeNet ############
################################

class SqueezeNetS1D(Inverse_Model):
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super(SqueezeNetS1D, self).__init__(image_type,curve_type,**kargs)
        self.backbone            = backbone
        self.backbone.features[0]= self.backbone.features[0].__class__(1, 96, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.final_pool = self.backbone.features[2].__class__(kernel_size=3, stride=2, padding=1) if final_pool else nn.Identity()
        self.backbone.features[2]= self.backbone.features[2].__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()

        shrink_coef              = conv_shrink(self.backbone.features)*conv_shrink(self.backbone.final_pool)
        self.c_after_conv        = self.outchannel*self.input_dim //(shrink_coef)
        #self.c_after_conv        = self.c_after_conv*2 if model_field == 'complex' else self.c_after_conv
        self.backbone.classifier = self.backbone.classifier.__class__(in_features=self.c_after_conv, out_features=self.output_dim)

    def forward(self, x,target=None):
        x = self.backbone(x)
        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

class SqueezeNet1S(SqueezeNetS1D):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models      = cplx_models if model_field == 'complex' else real_models
        backbone    = models.SqueezeNet()
        self.outchannel = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)

class Forward_Tail:
    def forward(self,x,target=None):
        x = self.backbone(x)#(...,256,x)
        x = self.tail_layer(x)#(...,256)
        x = x.view(self.final_shape)#(...,1,16,16)
        if target is not None:
            loss = self._loss(x,target)
            return loss,x
        return x


if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
