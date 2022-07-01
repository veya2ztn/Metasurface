from .FWD_RS2D import *

class Resnet18SPeakWise(ResnetS):
    '''
    for this model, the dataset must be the 'norm curve' data which is in [0,1]
    and mostly value is 0, only several peak. ___^__^_
    '''
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models  = cplx_models if model_field == 'complex' else real_models
        backbone=models.resnet18()
        self.outchannel  = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
        fc = self.backbone.fc
        self.find_peak = nn.Sequential(
            fc.__class__(in_features=self.c_after_conv, out_features=2000),
            nn.Sigmoid(),
            fc.__class__(in_features=2000, out_features=self.output_dim),
            nn.Sigmoid(),
        )
        self.find_val  = nn.Sequential(
            fc.__class__(in_features=self.c_after_conv, out_features=2000),
            nn.Sigmoid(),
            fc.__class__(in_features=2000, out_features=self.output_dim),
            nn.Sigmoid(),
        )
        self.model_class  = 'forward'
        self.backbone.fc = nn.Identity()
        self.focal_lossQ  = False
        self.usefocal_loss= True
    def forward(self, x,target=None):
        x   = self.backbone(x)  ;#print(x.shape)
        peak= self.find_peak(x)
        val = self.find_val(x)
        x   = torch.cat([val,peak],-1)
        x   = x.reshape(-1,1,256)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

    def _loss(self,x,t):
        feat_pred = x[..., :128]
        feat_real = t[..., :128]
        peak_pred = x[...,-128:]
        peak_real = t[...,-128:]


        feat_loss = ((feat_pred - feat_real)**2)
        feat_loss = feat_loss*torch.exp(peak_pred)
        feat_loss = feat_loss.mean()

        if self.focal_lossQ:
            peak_loss= FocalLoss()(peak_pred,peak_real)
        else:
            peak_loss= torch.nn.BCELoss()(peak_pred,peak_real)
        return feat_loss+5*peak_loss

    def _accu(self,real,pred):
        return torch.arange(1)[0]
class Resnet18SDigital(ResnetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models  = cplx_models if model_field == 'complex' else real_models
        backbone=models.resnet18()
        self.outchannel  = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
        fc = self.backbone.fc
        self.backbone.fc  = nn.Sequential(
            self.backbone.fc.__class__(in_features=self.c_after_conv, out_features=2000),
            nn.Sigmoid(),
            self.backbone.fc.__class__(in_features=2000, out_features=self.output_dim),
            nn.Sigmoid(),
        )
        self.model_class  = 'forward digital'
        self.focal_lossQ  = False
        self.usefocal_loss= True
    def _loss(self,x,target):
        if self.focal_lossQ:
            return FocalLoss()(x,target)
        else:
            return torch.nn.BCELoss()(x,target)
    def _accu(self,real,pred):
        real = torch.round(real)
        pred = torch.round(pred)
        return torch.eq(real, pred).prod(-1).float().mean()
    def _fullaccu(self,real,pred):
        real = torch.round(real)
        pred = torch.round(pred)
        bb=(real.flatten() == 1).nonzero().flatten()
        aa=(pred.flatten() == 1).nonzero().flatten()
        bingo = np.intersect1d(aa.cpu(),bb.cpu())
        accu1 = torch.Tensor([len(bingo)/(len(aa)+1e-8)])
        accu2 = torch.eq(real, pred).prod(-1).float().mean()
        return 1-accu1,1-accu2 #for the model autosave,the save accu is inversed
        #raise NotImplementedError
class Resnet18Sonehot(ResnetS):
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        models  = cplx_models if model_field == 'complex' else real_models
        backbone=models.resnet18()
        self.outchannel  = 512
        super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
        self.model_class  = 'forward digital'

    def _loss(self,x,target):
        target = target.long()
        return torch.nn.CrossEntropyLoss()(x,target)
    def _accu(self,x,target):
        target = target.long()
        real = target
        pred = torch.argmax(x,1)
        bingo= (real==pred).sum()
        return bingo/len(real)
    def _fullaccu(self,x,target):
        target = target.long()
        accu1 = self._loss(x,target)
        accu2 = self._accu(x,target)
        return accu1,1-accu2



# The Guass Sum idea is so stupid. If the \sigma (which is B here) to small, it is just
# a \delta function and the best amplitude parameter is just the curve itself.
# no better information comes when use Guass Sum.
# class Resnet18SGuassSum(ResnetS):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#         models  = cplx_models if model_field == 'complex' else real_models
#         backbone=models.resnet18()
#         self.outchannel  = 512
#         super().__init__(image_type,curve_type,backbone,model_field=model_field,**kargs)
#         x = torch.arange(self.output_dim).unsqueeze(1).float()
#         n = self.output_dim
#         y = x.unsqueeze(0).expand(n, n, 1)
#         x = x.unsqueeze(1).expand(n, n, 1)
#         dist = ((x-y)**2).sum(2)
#         self.dist = dist.unsqueeze(2)
#         self.amplitude= nn.Sequential(
#             self.backbone.fc.__class__(in_features=self.c_after_conv, out_features=2000),
#             nn.Sigmoid(),
#             self.backbone.fc.__class__(in_features=2000, out_features=self.output_dim),
#             nn.Sigmoid(),
#         )
#         self.width=nn.Sequential(
#             self.backbone.fc.__class__(in_features=self.c_after_conv, out_features=2000),
#             nn.Sigmoid(),
#             self.backbone.fc.__class__(in_features=2000, out_features=self.output_dim),
#             nn.Sigmoid(),
#         )
#         self.backbone.fc = nn.Identity()
#     def forward(self, x,target=None):
#         device = next(self.parameters()).device
#         if self.dist.device != device:
#             self.dist=self.dist.to(device)
#         x   = self.backbone(x)  ;#print(x.shape)
#         A   = 2*self.amplitude(x)-1 # rescale to [-1,1]
#         B   = self.width(x)+0.25  # set as 1 is need to take more consideration, maybe its better as 2 or 3
#         x   = torch.bmm(self.dist,B.permute(1,0).unsqueeze(1)).permute(2,1,0)
#         x   = torch.exp(-x)
#         x   = torch.bmm(x,A.unsqueeze(2)).squeeze(2)
#         x   = x.reshape(self.final_shape)
#         if target is None:return x
#         else:
#             loss = self._loss(x,target)
#         return loss,x
# class ResnetNatureGuassSum(ResnetNature):
#     def __init__(self,image_type,curve_type,model_field='real',**kargs):
#         super().__init__(image_type,curve_type,**kargs)
#         x = torch.arange(self.output_dim).unsqueeze(1).float()
#         n = self.output_dim
#         y = x.unsqueeze(0).expand(n, n, 1)
#         x = x.unsqueeze(1).expand(n, n, 1)
#         dist = ((x-y)**2).sum(2)
#         self.dist = dist.unsqueeze(2)
#         self.amplitude= nn.Sequential(
#             GRUTD(256,128,10),
#             nn.Linear(2560,200),
#             nn.Sigmoid(),
#             nn.Linear(200,128),
#             nn.Sigmoid(),
#         )
#         self.width=nn.Sequential(
#             GRUTD(256,128,10),
#             nn.Linear(2560,200),
#             nn.Sigmoid(),
#             nn.Linear(200,128),
#             nn.Sigmoid(),
#         )
#
#     def forward(self,x,target=None):
#         device = next(self.parameters()).device
#         if self.dist.device != device:self.dist=self.dist.to(device)
#         x = self.layer0(x)
#         x = self.conv3(x)
#         y = self.relu(x)
#         x = self.layer11(x)
#         y = self.layer12(y)
#         x = x+y
#         x = self.relu(x)
#         y = self.layer2(x)
#         x = x+y
#         x = self.relu(x)
#         y = self.layer3(x)
#         x = x+y
#         x = self.relu(x)
#         A   = 2*self.amplitude(x)-1 # rescale to [-1,1]
#         B   = self.width(x)+0.25  # set as 1 is need to take more consideration, maybe its better as 2 or 3
#         x   = torch.bmm(self.dist,B.permute(1,0).unsqueeze(1)).permute(2,1,0)
#         x   = torch.exp(-x)
#         x   = torch.bmm(x,A.unsqueeze(2)).squeeze(2)
#         x   = x.reshape(self.final_shape)
#         if target is None:return x
#         else:
#             loss = self._loss(x,target)
#         return loss,x

class TransferMatrix(Forward_Model):
    def __init__(self,image_type,curve_type,backbone,**kargs):
        super(TransferMatrix, self).__init__(image_type,curve_type,**kargs)
        #image_type=DataType('real',(1,16,16))
        curve_type=curve_type.__class__('real',(1,128))
        self.N11 = backbone(image_type,curve_type,model_field='real',**kargs)
        self.N12 = backbone(image_type,curve_type,model_field='real',**kargs)
        self.N21 = backbone(image_type,curve_type,model_field='real',**kargs)
        self.N22 = backbone(image_type,curve_type,model_field='real',**kargs)
    def forward(self,x,target=None):
        M11 = self.N11(x)
        M12 = self.N12(x)
        M21 = self.N21(x)
        M22 = self.N22(x)
        Rnm    =  (M12 - M21)**2 + (M11 + M22)**2
        R_real =  (M12**2+ M22**2 -M11**2- M21**2)/Rnm
        R_imag =  (-2*M11*M12 - 2*M21*M22)/Rnm
        R  = torch.stack([R_real,R_imag],-1)
        if target is not None:
            loss_unimodular = ((M11*M22+M12*M21 - 1)**2).mean()
            loss_mse = self._loss(R,target)
            loss = loss_unimodular + loss_mse
            return loss,R
        else:
            return R
class TransferMatrixResnet18(TransferMatrix):
    def __init__(self,image_type,curve_type,**kargs):
        super(TransferMatrixResnet18, self).__init__(image_type,curve_type,Resnet18S,**kargs)
class TransferMatrixResnetNature(TransferMatrix):
    def __init__(self,image_type,curve_type,**kargs):
        super(TransferMatrixResnetNature, self).__init__(image_type,curve_type,ResnetNature,**kargs)
if __name__=="__main__":
    model = Resnet18S(20,'real')
    print(model)
