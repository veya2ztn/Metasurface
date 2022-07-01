
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as models
from .model import BaseModel,Inverse_Model,conv_shrink
######  mlup
class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, in_channels, output_channels, stride,mid_channels=None,upsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if mid_channels is None:mid_channels=output_channels
        optpad = stride-1
        self.conv1    = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1      = norm_layer(mid_channels)
        self.conv2    = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,padding=1,bias=False)
        self.bn2      = norm_layer(mid_channels)
        self.conv3    = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=False)
        self.bn3      = norm_layer(output_channels)
        self.relu     = nn.ReLU(inplace=True)
        #self.relu  = nn.LeakyReLU(negative_slope=0.6, inplace=True)
        self.stride   = stride
        self.upsample = upsample
        if self.upsample is None:
            if output_channels!=in_channels or stride >1:
                self.upsample = nn.Sequential(
                        nn.Conv2d(in_channels,
                                           output_channels,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           bias=False) ,
                        norm_layer(output_channels),
                    )
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
class Uplayer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dim = dim
        for k in range(1,dim,2):
            self.layers.append(nn.Conv1d(1,2,kernel_size=k,padding=k//2))
    def forward(self,x):

        x=[layer(x) for layer in self.layers]
        x=torch.stack(x,-2)
        x=x.reshape(-1,1,self.dim,self.dim)
        return x

class MLUPInverse(Inverse_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        dim = curve_type.data_shape[-1]
        self.embedding       = Uplayer(dim)
        self.backbone        = models.resnet18()
        self.backbone.conv1  = self.backbone.conv1.__class__(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.backbone.maxpool= self.backbone.maxpool.__class__(kernel_size=3, stride=2, padding=1)
        self.backbone.avgpool= torch.nn.AdaptiveAvgPool2d(16)
        self.downsample      = nn.Sequential(Bottleneck(256,64,1),Bottleneck(64,32,1),Bottleneck(32,8,1),Bottleneck(8,1,1))
        self.final_layer     = torch.nn.Sigmoid()
    def forward(self,x,target=None):
        x = self.embedding(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.avgpool(x)
        x = self.downsample(x)
        x = x.reshape(self.final_shape)
        x = self.final_layer(x)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
