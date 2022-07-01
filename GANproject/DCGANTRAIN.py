import torch
import torch.nn as nn
from torch.autograd import Variable
from config_read import read_config
from train_base import struct_config
from mltool.ModelArchi.GANModel.SNdcgan import DCGAN_MODEL
from mltool.fastprogress import master_bar,progress_bar
from mltool.loggingsystem import LoggingSystem
from mltool.dataaccelerate import DataSimfetcher
from mltool.visualization import *

from utils import random_h_flip,random_v_flip
import numpy as np
import os,json,random,time
import sys
from model.criterion import SSIMError
from GAN_utils import *
import re

def train_phase(model,mode):
    model.G.train()
    model.D.train()
    model.G.zero_grad()
    model.D.zero_grad()
    if mode=='D':
        for p in model.D.parameters():p.requires_grad = True
        for p in model.G.parameters():p.requires_grad = False
    elif mode =='G':
        for p in model.D.parameters():p.requires_grad = False
        for p in model.G.parameters():p.requires_grad = True
    else:
        raise NotImplementedError

def plot16x16(images):
    graph_fig, graph_axes = plt.subplots(nrows=4, ncols=4, figsize=(16,16))
    graph_axes = graph_axes.flatten()
    for image,ax in zip(images,graph_axes):
        _=ax.imshow(image,cmap='hot',vmin=0, vmax=1)
        _=ax.set_xticks(())
        _=ax.set_yticks(())
    return graph_fig

def get_lattened_vector(sample_num,args):
    device       = args.device
    lattendim    = args.lattendim
    CURVE_REGION = args.CURVE_REGION
    if CURVE_REGION == "m1to1":
        z = torch.randn((sample_num, lattendim, 1, 1)).to(device)
    elif CURVE_REGION == "0to1":
        z = torch.rand((sample_num, lattendim, 1, 1)).to(device)
    z = Variable(z)
    return z

def load_checkpoints(args,verbose=True):
    TRAIN_MODE       = args.TRAIN_MODE
    PROJECTROOT      = args.PROJECTROOT
    d_lr             = args.d_lr
    g_lr             = args.g_lr
    disc_iter        = args.disc_iter
    gen_iter         = args.gen_iter
    IS_balance       = args.IS_balance
    GAN_metric       = args.GAN_metric #"MSELoss"
    if TRAIN_MODE in ["continue_train"]:
        print("Warning: you are now try to continue train a project")
        save_checkpoint   = args.save_checkpoint = os.path.dirname(os.path.dirname(args.last_weight_path))
        project_json_file = os.path.join(save_checkpoint,"project_config.json")
        if os.path.exists(project_json_file):
            new_args = args.load(project_json_file)
            _ = args.check_different(new_args,args)
            print("Warning: we will use the new configuration to furthur train")
            try:
                args.epoch_start=int(re.findall(r"at(.*)", args.last_weight_path)[0])
            except:
                print(args)
                args.epoch_start=int(re.findall(r"epoch-(.*)", args.last_weight_path)[0])
        else:
            hyparam_json_file = os.path.join(save_checkpoint,'hyperaram.json')
            hparam_dict = load_hyparam(hyparam_json_file)
            if os.path.exists(hyparam_json_file):
                print("Warning: it is the old version checkpoint file")
                print(f"we will load hyperparameter from {hyparam_json_file}")
                d_lr        = hparam_dict['d_lr']
                g_lr        = hparam_dict['g_lr']
                c_lr        = hparam_dict['c_lr']
                disc_iter   = hparam_dict['disc_iter']
                gen_iter    = hparam_dict['gen_iter']
                balance_coef= hparam_dict['balance_coef']
            else:
                print("Warning: No any project information detected, please check at")
                print(f"---> {save_checkpoint}")
    elif TRAIN_MODE in ["new_start","inhert_start"]:
        DATASETNAME          = [p for p in PROJECTROOT.split('/') if "SMSDataset" in p][0]
        TIME_NOW             = time.strftime("%m_%d_%H_%M_%S")
        CURVE_REGION         = args.CURVE_REGION
        TRIAL_NOW            = TIME_NOW
        save_checkpoint      = os.path.join(SAVEROOT,'checkpoints',f'DCGAN_PATTERN.{DATASETNAME}',F'DCGAN_{CURVE_REGION}_norm',TRIAL_NOW)
        args.save_checkpoint = save_checkpoint
        if not os.path.exists(save_checkpoint):os.makedirs(save_checkpoint)
        project_json_file    = os.path.join(save_checkpoint,"project_config.json")
        args.save(project_json_file)
        print(f'new checkpoint file created, we save the configuration into {project_json_file}')
    else:raise NotImplementedError
    hparam_dict = {'d_lr': d_lr,'g_lr': g_lr,'balance_coef':IS_balance,
                   'disc_iter':disc_iter,'gen_iter':gen_iter,'GAN_TYPE':args.GAN_TYPE}

    print("========================================")
    print(f"Project at {save_checkpoint}")
    if verbose:args.disp()
    return save_checkpoint,hparam_dict

def  get_model(args):
    GANMODEL_args.model = args.GAN_TYPE #'DCGAN'
    model = DCGAN_MODEL(GANMODEL_args)
    if not hasattr(args,"optimizer_type"):
        model.d_optimizer = torch.optim.Adam(model.D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        model.g_optimizer = torch.optim.Adam(model.G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    elif args.optimizer_type == "Adam":
        model.d_optimizer = torch.optim.Adam(model.D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        model.g_optimizer = torch.optim.Adam(model.G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    elif args.optimizer_type == "SGD":

        model.d_optimizer = torch.optim.SGD(model.D.parameters(), lr=args.d_lr, momentum=0.9)
        model.g_optimizer = torch.optim.SGD(model.G.parameters(), lr=args.g_lr, momentum=0.9)
    else:raise NotImplementedError
    return model

def load_model_and_dataloader(args):
    TRAIN_MODE       = args.TRAIN_MODE
    PROJECTROOT      = args.PROJECTROOT
    GAN_TYPE         = args.GAN_TYPE # DCGAN
    batch_size       = args.batch_size
    last_weight_path = args.last_weight_path
    d_lr             = args.d_lr
    g_lr             = args.g_lr

    image2curve_project_json= os.path.join(PROJECTROOT,"project_config.json")
    project_config          = read_config(image2curve_project_json)
    Image2Curve,project,db  = struct_config(project_config,db = None,build_model=False)

    train_loader = project.train_loader
    valid_loader = project.valid_loader
    train_loader = torch.utils.data.DataLoader(dataset=db.dataset_train,batch_size=batch_size,pin_memory=True,shuffle=True)

    model=get_model(args)
    print(f"the model we use is {args.GAN_TYPE}")
    if TRAIN_MODE in ["continue_train","inhert_start"]:
        print("Warning:we will use given checkpoint to continue train")
        print(f'we will use weight {last_weight_path} to continue train')
        modelGD_state_dict=torch.load(last_weight_path)
        model.G.load_state_dict(modelGD_state_dict['G_state_dict'])
        model.D.load_state_dict(modelGD_state_dict['D_state_dict'])
        if  'D_optimizer' in modelGD_state_dict:model.d_optimizer.load_state_dict(modelGD_state_dict['D_optimizer'])
        if  'G_optimizer' in modelGD_state_dict:model.g_optimizer.load_state_dict(modelGD_state_dict['G_optimizer'])
    device     = next(model.D.parameters()).device
    args.device=device
    return model,train_loader,valid_loader

def self_different_loss(tensor,*args,**kargs):
    loss = 0
    repeat=10
    tensor=(tensor*0.5)+0.5
    for _ in range(repeat):
        loss+=torch.nn.MSELoss()(tensor[torch.randperm(len(tensor))],tensor[torch.randperm(len(tensor))])
    loss = loss/repeat
    loss = (loss-0.5)**2
    # self_variation=((tensor.mean(0)-0.5)**2).mean()
    # loss = loss+self_variation
    return loss

def compute_ku(data):
    mean = data.mean(0)
    var  = data.var(0)
    ku   = ((data - mean) ** 4).mean(0) / (var**2) #计算峰度
    return ku

def infer_pattern_gan(save_checkpoint):
    #save_checkpoint = "checkpoints/DCGAN_PATTERN.SMSDatasetB1NES32,curve,simple.multilength/DCGAN_m1to1_norm/11_12_23_15_31"
    args = the_default_GAN_pattern_config
    demo_image_dir = os.path.join(save_checkpoint,'demo')
    save_weight_dir= os.path.join(save_checkpoint,'weights')
    model    = get_model(args)
    args.device = next(model.D.parameters()).device
    logsys   = LoggingSystem(True,save_checkpoint)
    _   = logsys.create_recorder(hparam_dict={},metric_dict={})
    for p in os.listdir(save_weight_dir):
        epoch = epoch = int(re.findall(r"at(.*)", p)[0])
        weight_path = os.path.join(save_weight_dir,p)
        z = get_lattened_vector(2000,args)
        fake_images = model.G(z)
        # random_sample=np.random.choice(range(len(fake_images)),16,replace=False).tolist()
        # images = fake_images[random_sample]
        # fake_image_figure=plot16x16(images.reshape(16,16,16).cpu().detach().numpy())
        # fake_image_name  = f'demo_train_at{epoch}'
        # fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
        # logsys.add_figure('demo_train',fake_image_figure,epoch)
        figure=plt.figure(dpi=100)
        errorbarplot(fake_images.reshape(-1,256).detach().cpu().numpy())
        plt.savefig(os.path.join(demo_image_dir,f"statis_info_at{epoch}"))
        logsys.add_figure('statis_info',figure,epoch)

class scale_saver:
    def __init__(self):
        self.recorder={}
        self.reset   ={}
    def record(self,name,s):
        if name not in self.reset:self.reset[name]=True
        if self.reset[name]:
            self.recorder[name]=[]
            self.reset[name]   =False
        self.recorder[name].append(s)

    def pop(self,name):
        value = np.mean(self.recorder[name])
        self.recorder[name]=[]
        return value

    def new_record(self):
        for name in self.reset.keys():self.reset[name]=True

    def get(self,name):
        if name not in self.recorder:return -1
        value = self.recorder[name][-1]
        return value

    def get_mean(self,name):
        if name not in self.recorder:return -1
        value = np.mean(self.recorder[name])
        return value

def train_pattern_gan(args):
    save_checkpoint,hparam_dict = load_checkpoints(args)
    demo_image_dir = os.path.join(save_checkpoint,'demo')
    save_weight_dir= os.path.join(save_checkpoint,'weights')
    if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)
    if not os.path.exists(save_weight_dir):os.makedirs(save_weight_dir)
    model,train_loader,valid_loader = load_model_and_dataloader(args)
    d_lr       = args.d_lr
    g_lr       = args.g_lr
    IS_balance = args.IS_balance
    disc_iter  = args.disc_iter
    gen_iter   = args.gen_iter
    epoches    = args.epoches
    epoch_start= args.epoch_start
    lattendim = args.lattendim

    logsys      = LoggingSystem(True,save_checkpoint)
    metric_dict = {'loss': None}
    _   = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)
    device     = next(model.D.parameters()).device
    d_loss=g_loss=sign_loss=curve_cycle_loss=image_cycle_loss=c_loss=i_loss=different=-1

    batches=len(train_loader)
    infinity_prefetcher=infinite_batcher(train_loader)
    one  = torch.FloatTensor([1]).to(device)
    mone = one * -1

    model.loss=torch.nn.MSELoss()
    #image_cycle_losser=SSIMError()
    image_cycle_losser=self_different_loss
    curve_cycle_losser=torch.nn.MSELoss()

    print(f"Project at {save_checkpoint}")
    check_diff_num = 30

    check_diff_random_sample=np.random.choice(range(args.batch_size),check_diff_num,replace=False).tolist()
    finished= False
    turn    =  0
    SR      = scale_saver()
    IS_balance1=IS_balance2=IS_balance
    while True:
        if finished:break
        mb     = logsys.create_master_bar(epoches,banner_info="")
        for epoch in mb:
            SR.new_record()
            if epoch==epoches-1:finished=True
            if epoch< epoch_start:continue
            if epoch%args.infer_epoch==0 and epoch!=0:
                fake_images = ((fake_images*0.5)+0.5).reshape(-1,256).detach().cpu().numpy()
                mean = np.mean(fake_images,0)
                var  = np.var(fake_images,0)
                #sc   = np.mean((fake_images - mean) ** 3,0)
                ku   = np.mean((fake_images - mean) ** 4,0) / (var**2)
                random_sample=np.random.choice(range(len(fake_images)),16,replace=False).tolist()
                images = fake_images[random_sample]
                fake_image_figure=plot16x16(images.reshape(16,16,16))
                fake_image_name  = f'demo_train_at{epoch}_{turn}'
                fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
                logsys.add_figure(f'demo_train_{turn}',fake_image_figure,epoch)

                figure=plt.figure(dpi=100)
                errorbarplot(fake_images)
                plt.savefig(os.path.join(demo_image_dir,f"statis_var_at{epoch}_{turn}"))
                logsys.add_figure(f'statis_var_{turn}',figure,epoch)

                figure=plt.figure(dpi=100)
                plt.errorbar(np.arange(len(mean)), mean, yerr = ku,alpha=0.4,color='r')
                plt.savefig(os.path.join(demo_image_dir,f"statis_kurtosis_at{epoch}_{turn}"))
                logsys.add_figure(f'statis_kurtosis_{turn}',figure,epoch)

                if epoch%args.save_infer_epoch!=0:
                    save_model(model,os.path.join(save_weight_dir,f'demo_train_at{epoch}'),mode='light')
            if (epoch%args.save_infer_epoch==0 and epoch!=0) or finished:
                save_model(model,os.path.join(save_weight_dir,f'demo_train_at{epoch}'),mode='full')
            train_phase(model,'D')
            for d_iter in range(disc_iter):
                #for p in model.D.parameters():_=p.data.clamp_(-model.weight_cliping_limit, model.weight_cliping_limit)
                vectors,images =get_data(infinity_prefetcher,epoch,device)
                real_labels  = torch.Tensor(images.size(0),1).uniform_(0.7,1.2).to(device)
                fake_labels  = torch.Tensor(images.size(0),1).uniform_(0.0,0.3).to(device)
                z = get_lattened_vector(images.size(0),args)
                fake_images = model.G(z)
                outputs     = model.D(fake_images)
                d_loss_fake = model.loss(outputs, fake_labels)
                # Compute BCE_Loss using real images
                outputs     = model.D(images)
                d_loss_real = model.loss(outputs, real_labels)
                d_loss = d_loss_real + d_loss_fake

                model.D.zero_grad()
                d_loss.backward()
                model.d_optimizer.step()
                d_loss=d_loss.item()
                SR.record("d_loss",d_loss)
                string = [f"'Step [{epoch}/{epoches}]"]+["{}:{:.4f} ".format(name,SR.get(name)) for name in SR.recorder.keys()]
                mb.lwrite(",".join(string),end='\r')
            train_phase(model,'G')
            for g_iter in range(gen_iter):
                z = get_lattened_vector(images.size(0),args)
                fake_labels = torch.ones(z.size(0),1).to(device)
                fake_images = model.G(z)
                fake_binary = (fake_images*0.5)+0.5
                fake_sign   = model.D(fake_images)
                g_loss      = model.loss(fake_sign, real_labels)
                i_loss      = image_cycle_losser(fake_images,images)
                # only for RDN case
                s_loss      = ((fake_binary.mean(0)-0.5)**2).mean()
                k_loss      = ((compute_ku(fake_binary)-1)**2).mean()
                f_loss      = 1-(fake_images**2).mean()
                if (fake_binary.mean(0).max()<0.6) and (fake_binary.mean(0).min()>0.4):
                    IS_balance1 = IS_balance1*1.1
                    IS_balance2 = IS_balance2*0.9
                    IS_balance1 = min(IS_balance,IS_balance1)
                else:
                    IS_balance1=IS_balance2=IS_balance
                t_loss      = (1-IS_balance1)*g_loss+IS_balance1*i_loss+IS_balance2*s_loss+0.1*IS_balance1*k_loss+0.01*IS_balance2*f_loss
                model.G.zero_grad()
                t_loss.backward()
                model.g_optimizer.step()

                SR.record("g_loss",g_loss.item())
                SR.record("i_loss",i_loss.item())
                SR.record("s_loss",s_loss.item())
                SR.record("k_loss",k_loss.item())
                SR.record("f_loss",f_loss.item())
                #mb.lwrite('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, i_loss: {:.4f},diff:{:.4f}'.format(epoch, epoches, d_loss, g_loss,i_loss,different),end='\r')
                string = [f"'Step [{epoch}/{epoches}]"]+["{}:{:.4f} ".format(name,SR.get(name)) for name in SR.recorder.keys()]
                mb.lwrite(",".join(string),end='\r')
            check_images = fake_images[check_diff_random_sample]
            check_images  = ((check_images*0.5)+0.5)
            different = np.mean([torch.nn.MSELoss()(check_images[i],check_images[j]).item() for i in range(check_diff_num) for j in range(i+1)])
            SR.record("diversity",different)
            if different < 0.1 and epoch>3:
                print("\n\n=================================")
                print("=================================")
                print("=================================")
                print(f"falling into mode collapse,diff={different},restart")
                del model
                torch.cuda.empty_cache()
                model    = get_model(args)
                turn    += 1
                break
            scalars_dict = {}
            logsys.recorder.add_scalars('GAN_loss',{'d_loss': SR.get_mean('d_loss'),'g_loss': SR.get_mean('g_loss')},epoch)
            logsys.record('different',different,epoch)
            for name in SR.recorder.keys():
                if name not in ['d_loss','g_loss']:
                    logsys.record(f"{name}{turn}",SR.get_mean(name),epoch)
    logsys.save_scalars()
    logsys.close()

if __name__=="__main__":
    args = the_default_GAN_pattern_config
    if len(sys.argv)==2:
        project_file = sys.argv[1]
        args = args.copy({
            'PROJECTROOT':project_file,
            'last_weight_path':"checkpoints/DCGAN_PATTERN.SMSDatasetB1NES32,curve,simple.multilength/DCGAN_m1to1_norm/11_13_20_18_12/weights/demo_train_at40",
            'TRAIN_MODE':"inhert_start",
        })
    train_pattern_gan(args)
