import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.squeezenet as real_models
import mltool.torch_complex.squeezenet2D as cplx_models

from .model import BaseModel,Forward_Model,conv_shrink

################################
######## SqueezeNet ############
################################

class SqueezeNetS(Forward_Model):
    # This model is for Forward Question.
    #   input is image.
    #  output is curve.
    def __init__(self,image_type,curve_type,backbone,model_field='real',final_pool=True,first_pool=False,**kargs):
        super(SqueezeNetS, self).__init__(image_type,curve_type,**kargs)
        self.backbone            = backbone
        self.backbone.features[0]= self.backbone.features[0].__class__(1, 96, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.features[2]= self.backbone.features[2].__class__(kernel_size=3, stride=2, padding=1) if first_pool else nn.Identity()
        self.backbone.final_pool = self.backbone.features[2].__class__(kernel_size=3, stride=2, padding=1) if final_pool else nn.Identity()

        shrink_coef              = conv_shrink(self.backbone.features)*conv_shrink(self.backbone.final_pool)
        self.c_after_conv        = self.outchannel*self.input_dim //(shrink_coef**2)

        if model_field == 'complex':
            self.backbone.classifier = self.backbone.classifier.__class__(in_features=self.c_after_conv, out_features=self.output_dim)
        else:
            self.backbone.classifier = nn.Linear(in_features=self.c_after_conv, out_features=self.output_dim)
        self.shape_after_conv    = (-1,self.c_after_conv,2) if model_field == 'complex' else (-1,self.c_after_conv)

    def forward(self, x,target=None):
        x = self.backbone.features(x)
        x = self.backbone.final_pool(x)
        x = x.view(self.shape_after_conv)
        x = self.backbone.classifier(x)
        x = x.reshape(self.final_shape)
        if target is None:
            return x
        else:
            loss = self._loss(x,target)
        return loss,x

class SqueezeNet1S(SqueezeNetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models      = cplx_models if model_field == 'complex' else real_models
        backbone    = models.SqueezeNet()
        self.outchannel = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)

class SqueezeNet1SFN(SqueezeNet1S):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        super().__init__(image_type,curve_type,model_field=model_field,**kargs)
        self.backbone.classifier=nn.Sequential(nn.Linear(in_features=self.c_after_conv, out_features=self.output_dim),
                                               nn.Sigmoid())
class SqueezeNetT(SqueezeNetS):
    def forward(self, x,target=None):
        x = (x-0.5)/0.5
        x = self.backbone.features(x)
        x = self.backbone.final_pool(x)
        x = x.view(self.shape_after_conv)
        x = self.backbone.classifier(x)
        x = x.reshape(self.final_shape)
        if target is None:
            return x
        else:
            loss = self._loss(x,target)
        return loss,x

class SqueezeNet1T(SqueezeNetT):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models      = cplx_models if model_field == 'complex' else real_models
        backbone    = models.SqueezeNet()
        self.outchannel = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
class DuplicateExpand:
    def forward(self,x,target=None):
        x = torch.stack([x,torch.zeros_like(x)],dim=-1)
        x = self.backbone.features(x)
        x = self.backbone.final_pool(x)
        x = x.view(-1, self.c_after_conv,2)
        x = self.backbone.classifier(x)
        x = x.reshape(self.final_shape)
        if target is None:
            return x
        else:
            loss = self._loss(x,target)
        return loss,x

class SqueezeNet1CDUPL(DuplicateExpand,SqueezeNet1S):
    def __init__(self,curve_feature,output_field,**kargs):
        super().__init__(curve_feature,output_field,model_field='complex',first_pool=False,**kargs)


if __name__=="__main__":
    from config import DataType
    print('''you can require the model_type,image_type,curve_type
             if the model_type is complex, both image_type and curve_type need to be torch_complex
             if the model_type is real, the image_type should be real but the curve_type can either be real or complex
          ''')
    print('for real model')
    image_type=DataType('real',(1,16,16))
    curve_type=DataType('real',(1,128))
    model_type='real'
    model = SqueezeNet1S(image_type,curve_type,model_field=model_type)
    x=image_type.sample()
    y=model.test(x)
    print("import shape {}".format(image_type.shape))
    print("output shape {}".format(curve_type.shape))
    print(model)
    print('==========================================')
    print()
    print('for complex model')
    image_type=DataType('real',(1,16,16))
    curve_type=DataType('complex',(1,128,2))
    model_type='complex'
    model = SqueezeNet1S(image_type,curve_type,model_field=model_type)
    x=image_type.sample()
    y=model.test(x)
    print("import shape {}".format(image_type.shape))
    print("output shape {}".format(curve_type.shape))
    print(model)
