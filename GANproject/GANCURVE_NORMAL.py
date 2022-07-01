import torch
import torch.nn as nn
from torch.autograd import Variable
from config_read import read_config
from train_base import struct_config
from mltool.ModelArchi.GANModel.SNwgan import WGAN_CP
from mltool.fastprogress import master_bar,progress_bar
from mltool.loggingsystem import LoggingSystem
from mltool.dataaccelerate import DataSimfetcher
from mltool.visualization import *

from utils import random_h_flip,random_v_flip
import numpy as np
from train_base import show_tandem_demo,show_gan_image_demo,DATAROOT,SAVEROOT
from model.criterion import SSIMError

import os,json,random,time
def get_data(infinity_prefetcher,epoch,device):
    vectors,images = infinity_prefetcher.next()
    images=images.to(device)
    vectors=vectors.to(device)
    images=(images-0.5)/0.5
    random_v_flip(images)
    random_h_flip(images)
    #trick -- 13 Add noise to inputs, decay over time
    noises = torch.randn_like(images)
    coeff  = 0.3*np.exp(-epoch/10)
    images = (1-coeff)*images+coeff*noises
    return vectors,images
def get_lattened_vector(curves,sample_num,device):
    #z = torch.rand((sample_num, 100, 1, 1)).to(device)
    z = torch.Tensor(sample_num, 100, 1, 1).normal_(0,1).to(device)
    z[:,:curve_dim,0,0]=curves[:,0,:]
    z = Variable(z)
    return z
def train_phase(model,mode):
    model.G.train()
    model.D.train()
    model.I2C.eval()
    model.G.zero_grad()
    model.D.zero_grad()
    for p in model.I2C.parameters():p.requires_grad = True
    if mode=='D':
        for p in model.D.parameters():p.requires_grad = True
        for p in model.G.parameters():p.requires_grad = False
    elif mode =='G':
        for p in model.D.parameters():p.requires_grad = False
        for p in model.G.parameters():p.requires_grad = True
    else:
        raise NotImplementedError
class infinite_batcher:
    def __init__(self,data_loader):
        self.length=len(data_loader)
        self.now=-1
        self.data_loader=data_loader
        self.prefetcher = None
    def next(self):
        if (self.now >= self.length) or (self.now == -1):
            if self.prefetcher is not None:del self.prefetcher
            self.prefetcher = DataSimfetcher(self.data_loader)
            self.now=0
        self.now+=1
        return self.prefetcher.next()

def plot16x16(images):
    graph_fig, graph_axes = plt.subplots(nrows=4, ncols=4, figsize=(16,16))
    graph_axes = graph_axes.flatten()
    for image,ax in zip(images,graph_axes):
        _=ax.imshow(image,cmap='hot',vmin=0, vmax=1)
        _=ax.set_xticks(())
        _=ax.set_yticks(())
    return graph_fig
def tround(tensor,val):
    t=tensor*0
    t[tensor<val]=0
    t[tensor>=val]=1
    return t

#from mltool.visualization import *
PROJECTROOT="checkpoints/PLGG250SMSDatasetB1NES32,curve,simple.on6847/Resnet18KSFNLeakReLUTend/26_02_44-seed-25441"
image2curve_project_json=os.path.join(PROJECTROOT,"project_config.json")
project_config = read_config(image2curve_project_json)
Image2Curve,project,db=struct_config(project_config,db = None,build_model=True)

train_loader = project.train_loader
valid_loader = project.valid_loader

random_seed=random.randint(1, 100000)
TIME_NOW  = time.strftime("%d_%H_%M")
TRIAL_NOW = '{}-seed-{}'.format(TIME_NOW,random_seed)
#TRIAL_NOW = "28_17_41-seed-76999"
save_checkpoint   = os.path.join(SAVEROOT,'checkpoints','GAN_CURVE','WGAN_EDI',TRIAL_NOW)
#TRIAL_NOW = 'enhance_dirc_data'
#save_checkpoint   = os.path.join(SAVEROOT,'checkpoints','GAN_CURVE','WGAN_WIL',TRIAL_NOW)



class Config:pass
args=Config()
args.is_train =True
args.dataroot='/home/tianning/MachineLearning/PyTorch-WassersteinGANs/data'
args.dataset ='mnist'
args.download =True
args.epochs =50
args.batch_size =64
args.cuda=True
args.load_D =False
args.load_G =False
args.generator_iters =10000
args.channels = 1
args.model ='WGAN-CP'
model = WGAN_CP(args)
model.I2C=Image2Curve
modelGD_state_dict=torch.load("checkpoints/GAN_PATTERN/WGAN_normal/28_11_46-seed-66172/weights/demo_valid_at5000")
#modelGD_state_dict=torch.load("checkpoints/GAN_CURVE/WGAN_randn/28_17_41-seed-76999/weights/demo_valid_at5000")
#modelGD_state_dict=torch.load("checkpoints/GAN_CURVE/WGAN_WIL/enhance_dirc_data/weights/demo_valid_at1500")
model.G.load_state_dict(modelGD_state_dict['G_state_dict'])
model.D.load_state_dict(modelGD_state_dict['D_state_dict'])
model.I2C.load_from(os.path.join(PROJECTROOT,'best','epoch572.best_MSError_0.0012'))

logsys         = LoggingSystem(True,save_checkpoint)
demo_image_dir = os.path.join(save_checkpoint,'demo')
save_weight_dir= os.path.join(save_checkpoint,'weights')
if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)
if not os.path.exists(save_weight_dir):os.makedirs(save_weight_dir)
d_lr = 0.00008
g_lr = 0.00001
c_lr = 0.001
IC_balance = 0.05
disc_iter = 5
gen_iter  = 1
epoches = 10000
epoch = 0
threothod_list = [0.4,0.5,0.6,0.8];
doearlystop= True
model.d_optimizer = torch.optim.RMSprop(model.D.parameters(), lr=d_lr)
model.g_optimizer = torch.optim.RMSprop(model.G.parameters(), lr=g_lr)
model.c_optimizer = torch.optim.Adam(model.G.parameters(), lr=c_lr)
model.d_optimizer.load_state_dict(modelGD_state_dict['D_optimizer'])
model.g_optimizer.load_state_dict(modelGD_state_dict['G_optimizer'])
#model.c_optimizer.load_state_dict(modelGD_state_dict['C_optimizer'])

hparam_dict   = {'d_lr': d_lr,'g_lr': g_lr,'c_lr':c_lr,'disc_iter':disc_iter,'gen_iter':gen_iter,'IC_balance':IC_balance}
metric_dict   = {}
accu_list=['valid_curve_loss']+[f'valid_pred_loss_{val}' for val in threothod_list]
for accu_type in accu_list:metric_dict[accu_type]         = None
_   = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)
_   = logsys.create_model_saver(accu_list=accu_list,es_max_window=30)
for accu_type in accu_list:metric_dict['best_'+accu_type] = None

device     = next(model.D.parameters()).device
print(f"Project at {save_checkpoint}")

mb         = logsys.create_master_bar(epoches,banner_info="")
d_loss=g_loss=sign_loss=curve_cycle_loss=image_cycle_loss=c_loss=i_loss=-1
losses={'S':[],'C':[],'I':[],'D':[]}
d_losses=[]
g_losses=[]
lattenedim=100
curve_dim=32


image_cycle_losser=SSIMError()
curve_cycle_losser=torch.nn.MSELoss()

batches=len(train_loader)
infinity_prefetcher=infinite_batcher(train_loader)
one  = torch.FloatTensor([1]).to(device)
mone = one * -1
earlystopQ = False
valid_curves = valid_loader.dataset.curvedata
valid_vector = valid_loader.dataset.vector
valid_image  = valid_loader.dataset.imagedata
for epoch in mb:
    epoch+=1
    if epoch%500==0:
        sample_num = 8
        random_sample=np.random.choice(range(len(fake_images)),sample_num,replace=False).tolist()
        I_PRED = fake_images[random_sample,0].cpu().detach().numpy()
        I_REAL = images[random_sample,0].cpu().detach().numpy()
        I_PRED = (I_PRED*0.5)+0.5;
        I_REAL = ((I_REAL*0.5)+0.5).round();
        real_v = vectors[random_sample,0].cpu().detach().numpy()
        pred_v = fake_vectors[random_sample,0].cpu().detach().numpy()
        score  = ((real_v-pred_v)**2).mean(1)

        data = list(zip(score,real_v,pred_v,I_REAL,I_PRED))
        fake_image_figure=show_tandem_demo(data,nrows=4,ncols=2)
        fake_image_name  = f'demo_train_at{epoch}'
        fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
        logsys.add_figure('demo_train',fake_image_figure,epoch)


        model.G.eval()
        model.I2C.eval()
        random_sample=np.random.choice(range(len(valid_vector)),sample_num,replace=False).tolist()
        input_vector= valid_vector[random_sample].to(device)
        input_image = valid_image[random_sample]
        z = get_lattened_vector(input_vector,sample_num,device) #(9,100,1,1)
        with torch.no_grad():
            fake_images = model.G(z)
            fakebinary  = ((fake_images*0.5)+0.5)
            fake_vectors= model.I2C(fakebinary)
            fakebinary  = fakebinary.round()
            pred_vectors= model.I2C(fakebinary)
        I_FAKE = fake_images[:,0].cpu().detach().numpy()
        I_REAL = input_image[:,0].cpu().detach().numpy()
        I_FAKE = ((I_FAKE*0.5)+0.5)
        I_PRED = I_FAKE.round()
        I_REAL = ((I_REAL*0.5)+0.5).round();
        real_v = input_vector[:,0].cpu().detach().numpy()
        fake_v = fake_vectors[:,0].cpu().detach().numpy()
        pred_v = pred_vectors[:,0].cpu().detach().numpy()
        score  = ((real_v-pred_v)**2).mean(1)
        data = list(zip(score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED))
        fake_image_figure=show_gan_image_demo(data,nrows=4,ncols=2)
        fake_image_name  = f'demo_valid_at{epoch}'
        fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
        logsys.add_figure('demo_valid',fake_image_figure,epoch  )

        save_weight_name=f'epoch-{epoch}'
        model.save_to(os.path.join(save_weight_dir,fake_image_name))

    train_phase(model,'D')
    d_loss_temp=[]
    for d_iter in range(disc_iter):
        model.D.zero_grad()
        #for p in model.D.parameters():_=p.data.clamp_(-model.weight_cliping_limit, model.weight_cliping_limit)
        vectors,images =get_data(infinity_prefetcher,epoch,device)
        if (images.size()[0] != 1000):continue
        # Train discriminator
        # WGAN - Training discriminator more iterations than generator
        # Train with real images
        d_loss_real = model.D(images)
        d_loss_real = d_loss_real.mean(0).view(1)
        d_loss_real.backward(one)


        # Train with fake images
        z = get_lattened_vector(vectors,images.size(0),device)
        fake_images = model.G(z)
        fake_images_enhance  = images*torch.Tensor(images.size(0), 1, 16, 16).uniform_(0.8,0.95).to(device)
        #fake_images_enhance = torch.Tensor(images.size(0), 1, 16, 16).uniform_(-0.9,0.9).to(device)
        fake_images = torch.cat([fake_images,fake_images_enhance])
        d_loss_fake = model.D(fake_images)
        d_loss_fake = d_loss_fake.mean(0).view(1)
        d_loss_fake.backward(mone)

        d_loss = d_loss_fake - d_loss_real
        model.d_optimizer.step()
        Wasserstein_D = d_loss_real - d_loss_fake
        d_loss=Wasserstein_D.item()
        d_loss_temp.append(d_loss)
        mb.lwrite('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, c_loss: {:.4f} i_loss: {:.4f}'.format(epoch, epoches, d_loss, g_loss,c_loss,i_loss),end='\r')
    g_loss_temp=[]
    c_loss_temp=[]
    i_loss_temp=[]
    train_phase(model,'G')
    for g_iter in range(gen_iter):
        vectors,images =get_data(infinity_prefetcher,epoch,device)
        z = get_lattened_vector(vectors,images.size(0),device)
        fake_images = model.G(z)
        g_loss = model.D(fake_images)
        g_loss = g_loss.mean().mean(0).view(1)

        if np.random.uniform()<1:
            train_curveQ=True
        else:
            train_curveQ=False
        model.G.zero_grad()
        if train_curveQ:
            g_loss.backward(one,retain_graph=True)
        else:
            g_loss.backward(one)
        model.g_optimizer.step()

        g_loss = -g_loss.item()
        g_loss_temp.append(g_loss)

        if train_curveQ:
            fakebinary  = (fake_images*0.5)+0.5
            fake_vectors = model.I2C(fakebinary)

            c_loss = curve_cycle_losser(fake_vectors,vectors)
            i_loss = image_cycle_losser(fake_images,images)
            condition_loss = (1-IC_balance)*c_loss + IC_balance*i_loss
            model.G.zero_grad()
            condition_loss.backward()
            model.c_optimizer.step()
            c_loss = c_loss.item()
            i_loss = i_loss.item()
        c_loss_temp.append(c_loss)
        i_loss_temp.append(i_loss)
        mb.lwrite('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, c_loss: {:.4f} i_loss: {:.4f}'.format(epoch, epoches, d_loss, g_loss,c_loss,i_loss),end='\r')

    d_loss=np.mean(d_loss_temp)
    g_loss=np.mean(g_loss_temp)
    c_loss=np.mean(c_loss_temp)
    i_loss=np.mean(i_loss_temp)

    ## inference
    model.G.eval()
    model.I2C.eval()
    input_vector= valid_vector.to(device)
    z = get_lattened_vector(input_vector,len(input_vector),device) #(9,100,1,1)
    score_list=[]
    with torch.no_grad():
        fake_images = model.G(z)
        fakebinary  = (fake_images*0.5)+0.5
        fake_vectors= model.I2C(fakebinary)
        score_fake_v= nn.MSELoss()(fake_vectors,input_vector).item()
        score_list.append(score_fake_v)
        for accu_type,val in zip(accu_list[1:],[0.4,0.5,0.6,0.8]):
            fakebinary  = tround(fake_images*0.5+0.5,val)
            pred_vectors= model.I2C(fakebinary)
            score_fake_v= nn.MSELoss()(pred_vectors,input_vector).item()
            score_list.append(score_fake_v)

    for accu,accu_type in zip(score_list,accu_list):
        best = metric_dict['best_'+accu_type] if metric_dict['best_'+accu_type] is not None else np.inf
        if accu< best:best = accu
        metric_dict[accu_type] = accu
        metric_dict['best_'+accu_type]=best
        logsys.record(accu_type, accu, epoch)
        logsys.record('best_'+accu_type, best, epoch)
    do_save    = True if epoch % 100 in range(1,10) else False
    earlystopQ = logsys.save_best_ckpt(model,metric_dict,epoch,doearlystop=doearlystop,do_save=do_save)
    logsys.recorder.add_scalars('GAN_loss',{'d_loss': d_loss,'g_loss': g_loss},epoch)
    logsys.record('curve_loss',c_loss,epoch)
    logsys.record('image_loss',i_loss,epoch)
    if earlystopQ :
        master_bar.lwrite("====== early stop! ==========")
        break_epoch = epoch
        break

logsys.save_scalars()
logsys.close()
