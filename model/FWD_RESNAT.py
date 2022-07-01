from .FWD_RESNAT_OLD import *

class TimeDistributed_Conv(nn.Module):
    def __init__(self, layer, time_steps):
        super(TimeDistributed_Conv, self).__init__()
        self.conv1  = torch.nn.Conv1d(layer.in_features,layer.out_features,kernel_size=1,bias=True)
    def forward(self, x):
        batch_size, time_steps, H, W = x.size()
        x     = x.reshape(batch_size, time_steps, H*W)
        x     = x.permute(0,2,1)
        x     = self.conv1(x).permute(0,2,1)
        return x

class ResnetNatureModule_z(Forward_Model):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.relu    = nn.LeakyReLU()
        self.layer0  = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            )
        self.conv3   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False)
        self.layer11 = conv_module_type1(128,256)
        self.layer12 = conv_module_type1(128,256)
        self.layer2  = conv_module_type2(256,128)
        self.layer3  = conv_module_type2(256,128)
        self.NLPtail = None

    def forward(self,x,target=None):
        x = self.layer0(x)
        x = self.conv3(x)
        y = self.relu(x)
        x = self.layer11(x)
        y = self.layer12(y)
        x = x+y
        x = self.relu(x)
        y = self.layer2(x)
        x = x+y
        x = self.relu(x)
        y = self.layer3(x)
        x = x+y
        x = self.relu(x)
        x = self.NLPtail(x)

        x = x.reshape(self.final_shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
class ResnetNature_z10(ResnetNatureModule_z):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Tanh(),
        )
class ResnetNature_z11(ResnetNatureModule_z):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Sigmoid(),
        )
class ResnetNature_z100(ResnetNatureModule_z):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed_Conv(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Tanh(),
        )
class ResnetNature_z101(ResnetNatureModule_z):
    def __init__(self,image_type,curve_type,**kargs):
        super().__init__(image_type,curve_type,**kargs)
        self.NLPtail= nn.Sequential(
            TimeDistributed_Conv(nn.Linear(256,64), time_steps = 256),
            BiLSTM(64,128,64),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,self.output_dim),
            nn.Sigmoid(),
        )
