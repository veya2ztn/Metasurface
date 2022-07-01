import torch.nn as nn
import torch
import sys

import numpy as np
import random
class NanValueError(Exception):pass


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_layer = None
        self.optimizer  = None
        self.usefocal_loss= False
        self.focal_lossQ  = False
        pass

    def _loss(self):
        raise NotImplementedError

    def _accu(self):
        raise NotImplementedError

    def forward(self,x,target=None):
        raise NotImplementedError

    def test(self,x,target=None):
        with torch.no_grad():
            return self(x,target)

    def fit(self,X_train,y_train):
        def closure():
            self.optimizer.zero_grad()
            loss,outputs = self(X_train,y_train)
            loss.backward()
            return loss
        loss=self.optimizer.step(closure)
        if torch.isnan(loss):
            raise NanValueError
        return loss

    def save_to(self,path):
        checkpoint=self.all_state_dict()
        torch.save(checkpoint,path)

    def load_from(self,path):
        checkpoint = torch.load(path)
        if ('state_dict' not in checkpoint):
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            print("we find existed optimizer checkpoint")
            if hasattr(self,'optimizer') and self.optimizer is not None:
                print("we load it to the self.optimizer")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("there is no self.optimizer,pass")
        if 'rnd_seed' in checkpoint:
            print("we find existed rnd_seed checkpoint")
            print("we will load it")
            rng_states = checkpoint['rnd_seed']
            random.setstate(rng_states["random_state"])
            np.random.set_state(rng_states["np_random_state"])
            torch.set_rng_state(rng_states["torch_random_state"])
            torch.cuda.set_rng_state_all(rng_states["torch_cuda_random_state"])
        if 'use_focal_loss' in checkpoint:self.focal_lossQ=checkpoint['use_focal_loss']

    def reset(self):
        weights_init = kaiming_init
        self.apply(weights_init)

    def all_state_dict(self,epoch=None,mode="light"):
        checkpoint={}
        checkpoint['epoch']  =  epoch
        checkpoint['state_dict']    = self.state_dict()
        checkpoint['use_focal_loss']= self.focal_lossQ
        if mode != "light":
            checkpoint['optimizer']     = self.optimizer.state_dict()
            checkpoint['rnd_seed']     = {
                "random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
            }
        return checkpoint

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:m.bias.data.fill_(0)
    elif isinstance(m,(nc.Linear,nc.Conv2d)):
        pass


class BaseModel(_Model):
    def __init__(self,input_type,output_type,**kargs):
        '''
        image_type: - field (real or complex)
                    - shape (MxN)
        curve_type: - field (real or complex)
                    - shape (out_put_fea,2) or (out_put_fea, )
        model_field is the field data field model receive,
            if set 'real', it means the model's layer like Conv, BatchNorm, is 'real' model
            if set 'complex', all the model layer will be convert to a complex form and the output field
                will be forced set 'complex'
        '''
        super(BaseModel, self).__init__()
        self.input_type   = input_type
        self.output_type  = output_type
        self.input_field  = input_type.field
        self.input_shape  = input_type.shape
        self.output_field = output_type.field
        self.output_shape = output_type.shape
        self.input_dim    = np.prod(self.input_shape)  # so the input data is (...,256)
        #self.output_dim   = np.prod(self.output_shape)  # so the output data is (...,128) and will be divide to (...,128,2) for complex num
        self.output_dim   = np.prod(self.output_shape)
        self.final_shape  = [-1]+list(self.output_type.data_shape)

class Inverse_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(curve_type,image_type,**kargs)
        self.model_class  = 'inverse'
    def _loss(self,x,target):
        return torch.nn.BCELoss()(x,target)
    def _accu(self,real,pred):
        real = torch.round(real)
        pred = torch.round(pred)
        return 1-(real==pred).float().mean()

class Forward_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.model_class  = 'forward'
    def _loss(self,x,target):
        return torch.nn.MSELoss()(x,target)
    def _accu(self,real,pred):
        return torch.arange(1)[0]
        #raise NotImplementedError


class Tandem_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(curve_type,image_type,**kargs)
        self.model_class  = 'tandem'

class Demtan_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(curve_type,image_type,**kargs)
        self.model_class  = 'demtan'

class Generative_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(curve_type,image_type,**kargs)
        self.model_class  = 'GAN'

class Discriminator_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(curve_type,image_type,**kargs)
        self.model_class  = 'DIS'

class Forward_Digital_Model(BaseModel):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.model_class  = 'forward digital'
    def _loss(self,x,target):
        return torch.nn.BCELoss()(x,target)
    def _accu(self,real,pred):
        real = torch.round(real)
        pred = torch.round(pred)
        return 1-(real==pred).float().mean() #for the model autosave,the save accu is inversed
        #raise NotImplementedError

class BaseModelBackup(_Model):
    def __init__(self,out_put_fea,output_field,
                      in_put_fea=(16,16),model_field='real',branches=1,**kargs):
        '''
        out_put_fea is the feature number model should return, must consistent with dataset
        output_field is the field type model should return, must consistent with dataset
            if 'complex' mean the target data shape (B,branch,fea_num,2)
            if 'real' mean the target data shape (B,branch,fea_num)
        in_put_fea is the feature number model receive, must consistent with dataset
            this value is used to compute dimension squeeze via Conv2d
            default is (16x16) is a binary matrix
        model_field is the field data field model receive,
            if set 'real', it means the model's layer like Conv, BatchNorm, is 'real' model
            if set 'complex', all the model layer will be convert to a complex form and the output field
                will be forced set 'complex'
        branches is the branch num, must consistent with dataset
        '''
        super(BaseModel, self).__init__()
        self.in_size     = np.prod(in_put_fea)
        self.out_put_fea = out_put_fea
        self.branches    = branches
        if model_field == 'complex':
            output_field = 'complex'
            self.out_features = self.out_put_fea*self.branches
        else:
            output_field = output_field
            self.out_features = self.out_put_fea*self.branches*(2 if output_field=='complex' else 1)
        if output_field is 'complex':
            self.final_shape = [-1,self.branches,self.out_put_fea,2]
        else:
            self.final_shape = [-1,self.branches,self.out_put_fea]
        self.output_field= output_field


def conv_stride_list(model):
    a=[]
    def weight_init(m):
        if hasattr(m,'stride'):
            if isinstance(m.stride,tuple):m.stride=m.stride[0]
            if m.stride > 1: a.append(m.stride)
    _=model.apply(weight_init)
    return a

def conv_shrink(model):
    a = conv_stride_list(model)
    if len(a)==0:a=1
    return np.prod(a)

if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
