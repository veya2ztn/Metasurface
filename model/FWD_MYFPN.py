from torch.nn.modules.conv import _ConvNd,_pair
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.models.resnet as real_models
from .model import BaseModel,Forward_Model,conv_shrink

def get_mask(size,dilation_size=(1,1),kernel_size=(3,3),stride=(1,1)):
    #stride  =1
    size1,size2=size[-2:]
    mask       =torch.torch.zeros((*size[:-2],size1+1,size2+1))
    k_s_1,k_s_2=kernel_size
    d_s_1,d_s_2=dilation_size
    k_1 = (k_s_1-1)*(d_s_1-1)+k_s_1
    k_2 = (k_s_2-1)*(d_s_2-1)+k_s_2
    start_1=(size1-k_1)//2+1
    end_1  =(size1+k_1)//2+1
    start_2=(size2-k_2)//2+1
    end_2  =(size2+k_2)//2+1
    mask[...,start_1:end_1:d_s_1,start_2:end_2:d_s_2]=1
    return mask
def config_list(size,stride):
    record=[]
    s = stride
    for p in range(1,size//2):
        for d in range(1,2*p+s-1+16):
            if (2*p+s-1)%d !=0:continue
            record.append([d,(2*p+s-1)//d+1])
    return  record
class Unified_Conv2d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        registered_input_size,
        config_list:list,
        stride= 1,
        groups=1,
        bias= False,
    ):
        stride      = s1,s2=_pair(stride)
        input_size  =  w, h=_pair(registered_input_size)
        self.registered_w   = w
        self.registered_h   = h
        padding_mode= 'circular'
        padding     = (0,0)
        if out_channels == -1:out_channels=in_channels
        if out_channels == -2:out_channels=in_channels//len(config_list)
        mask=[]
        for dilation,kernel_size in config_list:
            kernel_size = k1,k2=_pair(kernel_size)
            dilation    = d1,d2=_pair(dilation)
            mask.append(get_mask(size=(out_channels,in_channels,w,h),dilation_size=(d1,d2),kernel_size=(k1,k2)))
        mask=torch.cat(mask)
        out_channels,in_channels,w,h= mask.shape
        kernel_size = (w,h)
        dilation    = (1,1)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.mask = mask
        self.out_w= self.registered_w//s1
        self.out_h= self.registered_h//s2

    def forward(self, input):
        if input.is_cuda and (not self.mask.is_cuda):
            self.mask=self.mask.cuda()
#         b,c,w,h=input.shape
#         assert (w == self.registered_w) and (h==self.registered_h)
        return torch.nn.functional.conv2d(F.pad(input,pad=(self.registered_h//2,self.registered_h//2,self.registered_w//2,self.registered_w//2),mode='circular'),
                                          self.weight*self.mask,stride=self.stride)
class Unified_Conv2d_set(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        registered_input_size,
        config_list:list,
        kernel_size=1,
        stride = 1,
        dilation = 1,
        groups = 1,
        bias = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        mask=[]
        for p,d,k in config_list:
            kernel_size = k1,k2=_pair(k)
            dilation    = d1,d2=_pair(d)
            padding     = p1,p2=_pair(p)
            self.layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,dilation=dilation,
                                         stride=stride,padding=padding,padding_mode= 'circular',bias=False))

    def forward(self, input):
        return torch.cat([layer(input) for layer in self.layers],1)

class CircularBottleneckWrapper:
    def __init__(self,stride =  1,size = 16,expansion1=  1,expansion2=  3):
        self.stride    =stride
        self.size      =size
        self.expansion1=expansion1
        self.expansion2=expansion2
    def __call__(self,inplanes, do_residue=True,norm_module=nn.BatchNorm2d,activator_layer=nn.LeakyReLU):
        return CircularBottleneck(inplanes, do_residue=do_residue,norm_module=norm_module,activator_layer=activator_layer,
                                  stride =self.stride,input_size=self.size,expansion1=self.expansion1,expansion2=self.expansion2 )
class CircularBottleneck(nn.Module):
    def __init__(self, inplanes, do_residue=True,norm_module=nn.BatchNorm2d,activator_layer=nn.LeakyReLU,stride =  1,input_size = 16,expansion1=  1,expansion2=  3):
        super(CircularBottleneck, self).__init__()
        self.do_residue = do_residue
        self.stride    =stride
        self.input_size=input_size
        self.expansion1=expansion1
        self.expansion2=expansion2

        s          = 1
        self.conv1 = Unified_Conv2d(inplanes,self.expansion1,self.input_size,config_list(self.input_size,s),stride=s)
        s          = 1
        self.conv2 = Unified_Conv2d(self.conv1.out_channels,self.expansion1,self.conv1.out_w,config_list(self.conv1.out_w,s),stride=s)
        s          = stride
        self.conv3 = Unified_Conv2d(self.conv2.out_channels,self.expansion2,self.conv2.out_w,config_list(self.conv2.out_w,s),stride=s)

        self.bn1   = norm_module(self.conv1.out_channels)
        self.bn2   = norm_module(self.conv2.out_channels)
        self.bn3   = norm_module(self.conv3.out_channels)
        self.nonli = activator_layer(inplace=True)

        if stride != 1 and do_residue:
            self.residuer = nn.Sequential(
                Unified_Conv2d(inplanes,self.expansion2,self.conv2.out_w,config_list(self.conv2.out_w,s),stride=s),
                norm_module(self.conv3.out_channels),
            )
        else:
            self.residuer = None
        self.out_channels = self.conv3.out_channels
    def forward(self, x):
        if self.do_residue:
            residual = self.residuer(x) if self.residuer is not None else x
        else:
            residual = 0

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonli(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonli(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out+= residual
        out = self.nonli(out)

        return out
class DenseHalfDownLayer(nn.Module):
    def __init__(self,input_dim,output_dim,final_activator = nn.Sigmoid,inner_activator = nn.Tanh):
        super().__init__()
        output_dim = int(output_dim)
        if output_dim > 256:
            self.layers = nn.Sequential(nn.Linear(input_dim,output_dim),final_activator())
        else:
            self.layers = nn.Sequential(nn.Linear(input_dim,256),inner_activator())
            i=2
            dim=256
            while dim//2 > output_dim:
                self.layers.add_module(f'{i}',nn.Linear(dim,dim//2));i+=1
                self.layers.add_module(f'{i}',inner_activator());i+=1
                dim=dim//2
                if dim <= 16:break
            self.layers.add_module(f'{i}',torch.nn.Linear(dim,output_dim));i+=1
            self.layers.add_module(f'{i}',final_activator());i+=1

    def forward(self,x):
        return self.layers(x)
class Unified_Conv2d_Block(nn.Module):
    def __init__(self, in_planes, ou_planes,registered_input_size,config_list,normer=nn.BatchNorm2d,activator=nn.Tanh(),stride= 1,**kargs):
        super().__init__()
        self.entangle   = Unified_Conv2d(in_planes,ou_planes,registered_input_size,config_list,stride=stride,**kargs)

        self.layernrm   = normer(self.entangle.out_channels) if registered_input_size/stride!=1 else None
        self.activate   = activator
    def forward(self,x):
        x = self.entangle(x)
        if self.layernrm is not None:
            x = self.layernrm(x)
        x = self.activate(x)
        return x
class Unified_Model(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.shallow_layers =  nn.ModuleList([
        	Unified_Conv2d_Block(1,3,16,config_list(16, 1),stride= 1),
        	Unified_Conv2d_Block(1,3,16,config_list(16, 2),stride= 2),
        	Unified_Conv2d_Block(1,3,16,config_list(16, 4),stride= 4),
        	Unified_Conv2d_Block(1,3,16,config_list(16, 8),stride= 8),
        	Unified_Conv2d_Block(1,3,16,config_list(16,16),stride=16),]
        )
        self.deep_layers    = real_models.resnet18()
        #modify the model to right config
        self.deep_layers.conv1  = self.deep_layers.conv1.__class__(self.shallow_layers[0].entangle.out_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.deep_layers.maxpool= nn.Identity()

        self.pool_layers =  nn.ModuleList([
                nn.AdaptiveAvgPool2d(output_size=8),
                nn.AdaptiveAvgPool2d(output_size=4),
                nn.AdaptiveAvgPool2d(output_size=2),
                nn.AdaptiveAvgPool2d(output_size=1),]
        )

        self.downsample_laysers=  nn.ModuleList([
            nn.Conv2d(81+ 64, 8*2,kernel_size=8),
            nn.Conv2d(51+128,16*2,kernel_size=4),
            nn.Conv2d(51+256,32*2,kernel_size=2),
        ])
        self.tail_layer = DenseHalfDownLayer(738,self.output_dim)
    def forward(self,x_o,target=None):
        #shallow_list = [layer(x) for layer in self.shallow_layers]
        #now do deep layer
        s  = self.shallow_layers[0](x_o)
        x  = self.deep_layers.conv1(s)
        x  = self.deep_layers.bn1(x)
        x  = self.deep_layers.relu(x)
        x  = self.deep_layers.maxpool(x)

        x  = self.deep_layers.layer1(x);
        s  = self.pool_layers[0](torch.cat([s,x],dim=1))
        out= self.downsample_laysers[0](s)

        x  = self.deep_layers.layer2(x)
        s  = self.shallow_layers[1](x_o)
        s  = self.pool_layers[1](torch.cat([s,x],dim=1))
        s  = self.downsample_laysers[1](s)
        out= torch.cat([out,s],dim=1)

        x  = self.deep_layers.layer3(x)
        s  = self.shallow_layers[2](x_o)
        s  = self.pool_layers[2](torch.cat([s,x],dim=1))
        s  = self.downsample_laysers[2](s)
        out= torch.cat([out,s],dim=1)

        x  = self.deep_layers.layer4(x)
        s  = self.shallow_layers[3](x_o)
        s  = self.pool_layers[3](torch.cat([s,x],dim=1))
        out= torch.cat([out,s,self.shallow_layers[4](x_o)],dim=1)
        del x
        del s
        out= out.squeeze(-1).squeeze(-1)
        out= self.tail_layer(out)
        out= out.reshape(self.final_shape)
        if target is None:return out
        else:
            loss = self._loss(out,target)
        return out
