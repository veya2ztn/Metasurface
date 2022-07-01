import torch
import torch.nn as nn
from torch.autograd import Variable
from config_read import read_config
from train_base import struct_config
from mltool.ModelArchi.GANModel.SNdcgan import DCGAN_MODEL
from mltool.fastprogress import master_bar,progress_bar
from mltool.loggingsystem import LoggingSystem
from mltool.visualization import *

import re
import numpy as np
from train_base import show_tandem_demo,show_gan_image_demo,DATAROOT,SAVEROOT
from model.criterion import SSIMError,SelfEnhanceLoss3
from GAN_utils import *
import os,json,random,time
import json
# torch.backends.cudnn.enabled  = False
# torch.backends.cudnn.benchmark=True

def train_phase(model,mode):
    model.G.train()
    model.D.train()
    model.I2C.eval()
    model.G.zero_grad()
    model.D.zero_grad()
    for p in model.I2C.parameters():p.requires_grad = False

    if mode=='D':
        for p in model.D.parameters():p.requires_grad = True
        for p in model.G.parameters():p.requires_grad = False
    elif mode =='G':
        for p in model.D.parameters():p.requires_grad = False
        for p in model.G.parameters():p.requires_grad = True
    else:
        raise NotImplementedError
def get_lattened_vector(curves,sample_num,args):
    device       = args.device
    lattendim    = args.lattendim
    CURVE_REGION = args.CURVE_REGION
    curve_dim    = args.curve_dim
    if CURVE_REGION == "m1to1":
        z = torch.randn((sample_num, lattendim, 1, 1)).to(device)
    elif CURVE_REGION == "0to1":
        z = torch.rand((sample_num, lattendim, 1, 1)).to(device)
    z[:,:curve_dim,0,0]=curves[:,0,:]
    z = Variable(z)
    return z
def have_a_image_snap(model,input_vector,input_image,args):
    model.G.eval()
    model.I2C.eval()
    #nrows,ncols = args.snap_size
    nrows,ncols = 2,4
    sample_num=nrows*ncols
    random_sample=np.random.choice(range(len(input_vector)),sample_num,replace=False).tolist()
    input_vector =input_vector[random_sample]
    input_image  =input_image[random_sample]
    z = get_lattened_vector(input_vector,sample_num,args) #(9,100,1,1)
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
    fake_image_figure=show_gan_image_demo(data,nrows=nrows,ncols=ncols)
    return fake_image_figure
def load_config_from_json(path):
    with open(path,'r',encoding='utf-8') as f:config_dict=json.load(f)
    return Config(config_dict)
def compute_ku(data):
    mean = data.mean(0)
    var  = data.var(0)
    ku   = ((data - mean) ** 4).mean(0) / (var**2+0.01) #计算峰度
    return ku
def get_model(args,Image2Curve):
    GANMODEL_args.model = args.GAN_TYPE #'DCGAN'
    model = DCGAN_MODEL(GANMODEL_args)
    model.I2C=Image2Curve
    model.I2C.load_from(args.FEDMODEL_WEIGHT)
    model.d_optimizer = torch.optim.Adam(model.D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    model.g_optimizer = torch.optim.Adam(model.G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    model.c_optimizer = torch.optim.SGD(model.G.parameters(), lr=args.c_lr)

    if args.TRAIN_MODE in ["new_start"]:
        if args.GAN_PRE_WEIGHT is not None:
            print("Warning:detact use pre-train GAN branch as start")
            print(f"load pre-train weight at {args.GAN_PRE_WEIGHT}")
            modelGD_state_dict=torch.load(args.GAN_PRE_WEIGHT)
            model.G.load_state_dict(modelGD_state_dict['G_state_dict'])
            model.D.load_state_dict(modelGD_state_dict['D_state_dict'])
        else:
            print("Warning:we dont use warm-up pretrain GAN part ")
    elif args.TRAIN_MODE in ["continue_train"]:
        print("Warning:we will use given checkpoint to continue train")
        print(f'we will use weight {args.last_weight_path} to continue train')
        modelGD_state_dict=torch.load(args.last_weight_path)
        print("<-----   load network weight   ----->")
        model.G.load_state_dict(modelGD_state_dict['G_state_dict'])
        model.D.load_state_dict(modelGD_state_dict['D_state_dict'])
        print("<-----   load optimizer weight   ----->")
        if 'D_optimizer' in modelGD_state_dict:model.d_optimizer.load_state_dict(modelGD_state_dict['D_optimizer'])
        if 'G_optimizer' in modelGD_state_dict:model.g_optimizer.load_state_dict(modelGD_state_dict['G_optimizer'])
        if 'C_optimizer' in modelGD_state_dict:model.c_optimizer.load_state_dict(modelGD_state_dict['C_optimizer'])

    else:
        raise NotADirectoryError
    return model
def load_model_and_dataloader(args):
    TRAIN_MODE       = args.TRAIN_MODE
    PROJECTROOT      = args.FORWARD_PROJECT_ROOT
    GAN_TYPE         = args.GAN_TYPE # DCGAN
    GAN_PRE_WEIGHT   = args.GAN_PRE_WEIGHT
    last_weight_path = args.last_weight_path
    FEDMODEL_WEIGHT  = args.FEDMODEL_WEIGHT
    d_lr             = args.d_lr
    g_lr             = args.g_lr
    c_lr             = args.c_lr
    batch_size       = args.batch_size if hasattr(args,'batch_size') else None
    image2curve_project_json= os.path.join(PROJECTROOT,"project_config.json")
    project_config          = read_config(image2curve_project_json)
    project_config.model.pre_train_weight=None
    Image2Curve,project,db  = struct_config(project_config,db = None,build_model=True)

    train_loader = project.train_loader
    valid_loader = project.valid_loader
    if batch_size is not None:
        train_loader = torch.utils.data.DataLoader(dataset=db.dataset_train,batch_size=batch_size,pin_memory=True,shuffle=True)
        if valid_loader.batch_size>batch_size:
            valid_loader = torch.utils.data.DataLoader(dataset=db.dataset_train,batch_size=batch_size)
    model      = get_model(args,Image2Curve)
    device     = next(model.D.parameters()).device
    args.device=device
    return model,train_loader,valid_loader
def load_checkpoints(args,verbose=True):
    TRAIN_MODE       = args.TRAIN_MODE
    PROJECTROOT      = args.FORWARD_PROJECT_ROOT = os.path.dirname(os.path.dirname(args.FEDMODEL_WEIGHT))

    GAN_PRE_WEIGHT   = args.GAN_PRE_WEIGHT
    d_lr             = args.d_lr
    g_lr             = args.g_lr
    c_lr             = args.c_lr
    disc_iter        = args.disc_iter
    gen_iter         = args.gen_iter
    balance_coef     = args.balance_coef
    GAN_metric       = args.GAN_metric #"MSELoss"
    if TRAIN_MODE in ["continue_train"]:
        print("Warning: you are now try to continue train a project")
        save_checkpoint   = args.save_checkpoint = os.path.dirname(os.path.dirname(args.last_weight_path))
        project_json_file = os.path.join(save_checkpoint,"project_config.json")
        assert args.epoch_start == 0
        if os.path.exists(project_json_file):
            new_args = load_config_from_json(project_json_file)
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
    elif TRAIN_MODE in ["new_start"]:
        DATASETNAME= PROJECTROOT.split('/')[1]
        FWDMODEL   = PROJECTROOT.split('/')[-2]
        train_type = f'DCGAN_CURVE.{DATASETNAME}'
        model_class= FWDMODEL
        TIME_NOW   = time.strftime("%m_%d_%H_%M_%S")
        # import socket
        # hostname   = socket.gethostname()
        # TRIAL_NOW  = '{}-host-{}'.format(TIME_NOW,hostname)
        TRIAL_NOW = TIME_NOW
        save_checkpoint   = os.path.join(SAVEROOT,'checkpoints',train_type,model_class,TRIAL_NOW)
        args.save_checkpoint = save_checkpoint
        if not os.path.exists(save_checkpoint):os.makedirs(save_checkpoint)
        project_json_file = os.path.join(save_checkpoint,"project_config.json")
        args.save(project_json_file)
        print(f'new checkpoint file created, we save the configuration into {project_json_file}')

    hparam_dict = {'d_lr': d_lr,'g_lr': g_lr,'c_lr':c_lr,
                   'disc_iter':disc_iter,'gen_iter':gen_iter,
                   'balance_coef':balance_coef,'metric':GAN_metric,"GAN_PRE_TRAIN":0 if GAN_PRE_WEIGHT is None else 1}
    hparam_dict['balance_coef']="use_ssim_as_balance" if balance_coef == 'use_ssim_as_balance' else ",".join([f"{num}" for num in balance_coef])
    hparam_dict['TWOCC'] = args.train_with_one_compoent_constrain
    print("========================================")
    print(f"Project at {save_checkpoint}")
    if verbose:args.disp()
    return save_checkpoint,hparam_dict
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

####################
### LOAD MODEL #####
####################
def train_curve_gan(args):
    save_checkpoint,hparam_dict = load_checkpoints(args)
    demo_image_dir = os.path.join(save_checkpoint,'demo')
    save_weight_dir= os.path.join(save_checkpoint,'weights')
    if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)
    if not os.path.exists(save_weight_dir):os.makedirs(save_weight_dir)

    model,train_loader,valid_loader = load_model_and_dataloader(args)

    d_lr        = hparam_dict['d_lr']
    g_lr        = hparam_dict['g_lr']
    c_lr        = hparam_dict['c_lr']
    disc_iter   = hparam_dict['disc_iter']
    gen_iter    = hparam_dict['gen_iter']
    #balance_coef= hparam_dict['balance_coef']
    balance_coef= args.balance_coef
    train_wiht_OC=hparam_dict['TWOCC']

    epoches    		= args.epoches
    doearlystop     = args.doearlystop
    doearlystop     = False
    infer_epoch     = args.infer_epoch
    lattendim  	    = args.lattendim
    curve_dim 	    = args.curve_dim
    threothod_list  = args.threothod_list
    epoch_start     = args.epoch_start
    logsys          = LoggingSystem(True,save_checkpoint)
    batches=train_loader.batch_size
    print(f">>> train batch size is {batches}")
    metric_dict   = {}
    accu_list=['valid_curve_loss']+['valid_pred_loss_0.5']+[f'valid_pred_loss_{val}_polished' for val in threothod_list]
    for accu_type in accu_list:metric_dict[accu_type]          = None
    metric_dict['best_valid_curve_loss']= None
    _   = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)
    best_accu_saved=accu_list[:2]
    _   = logsys.create_model_saver(accu_list=best_accu_saved,es_max_window=100,block_best_interval=300,epoches=epoches)
    for accu_type in accu_list:metric_dict['best_'+accu_type] = None


    print('use hyparam:')
    print(hparam_dict)
    device = args.device
    mb     = logsys.create_master_bar(epoches,banner_info="")
    d_loss = g_loss=sign_loss=curve_cycle_loss=image_cycle_loss=c_loss=i_loss=-1

    image_cycle_losser= SSIMError()
    if hparam_dict['metric']=="MSELoss":
        curve_cycle_losser= torch.nn.MSELoss()
    elif hparam_dict['metric']=="SEALoss3":
        curve_cycle_losser= SelfEnhanceLoss3()
    model.loss        = torch.nn.MSELoss()


    infinity_prefetcher=infinite_batcher(train_loader)
    earlystopQ = False
    valid_curves = valid_loader.dataset.curvedata
    valid_vector = valid_loader.dataset.vector.to(device)
    valid_image  = valid_loader.dataset.imagedata
    SR           = scale_saver()
    turn         =0
    check_diff_num=30
    for epoch in mb:
        if epoch< epoch_start:continue
        train_phase(model,'D')
        for d_iter in range(disc_iter):
            model.D.zero_grad()
            #for p in model.D.parameters():_=p.data.clamp_(-model.weight_cliping_limit, model.weight_cliping_limit)
            vectors,images =get_data(infinity_prefetcher,epoch,device)
            real_labels  = torch.Tensor(images.size(0),1).uniform_(0.7,1.2).to(device)
            fake_labels  = torch.Tensor(images.size(0),1).uniform_(0.0,0.3).to(device)
            z = get_lattened_vector(vectors,images.size(0),args)
            fake_images = model.G(z)
            fakebinary  = (fake_images*0.5)+0.5
            outputs     = model.D(fake_images)
            d_loss_fake = model.loss(outputs, fake_labels)
            outputs     = model.D(images)
            d_loss_real = model.loss(outputs, real_labels)
            d_loss      = d_loss_real + d_loss_fake

            model.D.zero_grad()
            d_loss.backward()
            model.d_optimizer.step()

            # different    = torch.nn.MSELoss()(fakebinary[check_diff_index_1].detach(),fakebinary[check_diff_index_2].detach()).item()
            # SR.record("diversity",different)
            SR.record("d_loss",d_loss.item())
            string = [f"'Step [{epoch}/{epoches}]"]+["{}:{:.4f} ".format(name,SR.get(name)) for name in SR.recorder.keys()]
            mb.lwrite(",".join(string),end='\r')
            #mb.lwrite('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, c_loss: {:.4f} i_loss: {:.4f}'.format(epoch, epoches, d_loss, g_loss,c_loss,i_loss),end='\r')

        train_phase(model,'G')
        for g_iter in range(gen_iter):
            vectors,images =get_data(infinity_prefetcher,epoch,device)
            z = get_lattened_vector(vectors,images.size(0),args)
            real_labels = torch.ones(z.size(0),1).to(device)
            fake_images = model.G(z)
            fake_sign   = model.D(fake_images)
            fakebinary  = (fake_images*0.5)+0.5
            if train_wiht_OC:fakebinary=one_component(fakebinary)
            fake_vectors= model.I2C(fakebinary)
            g_loss      = model.loss(fake_sign, real_labels)
            c_loss      = curve_cycle_losser(fake_vectors,vectors)
            i_loss      = image_cycle_losser(fake_images,images)
            p_loss      = self_different_loss(fake_images)
            s_loss      = ((fakebinary.mean(0)-0.5)**2).mean()
            k_loss      = ((compute_ku(fakebinary)-1)**2).mean()
            if balance_coef == 'use_ssim_as_balance':
                t_loss      = (i_loss-0.5)*g_loss+(1-i_loss+0.5)*c_loss
            else:
                t_loss      = balance_coef[0]*g_loss+balance_coef[1]*c_loss+\
                              balance_coef[2]*i_loss+balance_coef[3]*p_loss+\
                              0.01* balance_coef[3]*s_loss+balance_coef[3]*k_loss
            model.G.zero_grad()
            t_loss.backward()
            model.g_optimizer.step()

            SR.record("g_loss",g_loss.item())
            SR.record("c_loss",c_loss.item())
            SR.record("i_loss",i_loss.item())
            SR.record("p_loss",p_loss.item())
            SR.record("s_loss",s_loss.item())
            SR.record("k_loss",k_loss.item())
            #mb.lwrite('Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, i_loss: {:.4f},diff:{:.4f}'.format(epoch, epoches, d_loss, g_loss,i_loss,different),end='\r')
            string = [f"'Step [{epoch}/{epoches}]"]+["{}:{:.4f} ".format(name,SR.get(name)) for name in SR.recorder.keys()]
            mb.lwrite(",".join(string),end='\r')

        check_diff_random_sample=np.random.choice(range(len(fakebinary)),check_diff_num,replace=False).tolist()
        check_diff_index_1    = [check_diff_random_sample[i] for i in range(check_diff_num) for j in range(i+1)]
        check_diff_index_2    = [check_diff_random_sample[j] for i in range(check_diff_num) for j in range(i+1)]
        different    = torch.nn.MSELoss()(fakebinary[check_diff_index_1].detach(),fakebinary[check_diff_index_2].detach()).item()
        SR.record("diversity",different)
        # if different < 0.1 and epoch>1:
        #     print("\n\n=================================")
        #     print("=================================")
        #     print("=================================")
        #     print(f"falling into mode collapse,diff={different},restart")
        #     del model
        #     torch.cuda.empty_cache()
        #     #model    = get_model(args)
        #     turn    += 1
        #     break

        if (epoch%infer_epoch==0 and epoch!=0) or earlystopQ or epoch==epoches-1:
            theimages   = fakebinary.reshape(-1,256).detach().cpu().numpy()
            mean        = np.mean(theimages,0)
            var         = np.var(theimages,0)
            ku          = np.mean((theimages - mean) ** 4,0) / (var**2+0.01)

        logsys.save_latest_ckpt(model,epoch)
        ## inference
        model.G.eval()
        model.I2C.eval()
        input_vector= valid_vector.to(device)
        z = get_lattened_vector(input_vector,len(input_vector),args)
        score_list=[]
        with torch.no_grad():
            fake_images = model.G(z)
            ### performance for the origin generated image
            fakebinary  = fake_images*0.5+0.5
            fake_vectors= model.I2C(fakebinary)
            score_fake_v= nn.MSELoss()(fake_vectors,input_vector).item()
            score_list.append(score_fake_v)
            ### performance for the round generated image
            fakebinary  = tround(fake_images*0.5+0.5,0.5)
            fake_vectors= model.I2C(fakebinary)
            score_fake_v= nn.MSELoss()(fake_vectors,input_vector).item()
            score_list.append(score_fake_v)
            for val in threothod_list:
                fakebinary  = tround(fake_images*0.5+0.5,val)
                fakebinary  = one_component(fakebinary)
                pred_vectors= model.I2C(fakebinary)
                score_fake_v= nn.MSELoss()(pred_vectors,input_vector).item()
                score_list.append(score_fake_v)

        for accu,accu_type in zip(score_list,accu_list):
            best = metric_dict['best_'+accu_type] if metric_dict['best_'+accu_type] is not None else np.inf
            if accu< best:metric_dict['best_'+accu_type]= accu
            metric_dict[accu_type]        = accu
            logsys.record(accu_type, metric_dict[accu_type], epoch)
            logsys.record('best_'+accu_type, metric_dict['best_'+accu_type], epoch)

        earlystopQ = logsys.save_best_ckpt(model,metric_dict,epoch,doearlystop=doearlystop)
        logsys.recorder.add_scalars('GAN_loss',{'d_loss': SR.get_mean('d_loss'),'g_loss': SR.get_mean('g_loss')},epoch)
        logsys.record('different',different,epoch)
        for name in SR.recorder.keys():
            if name not in ['d_loss','g_loss']:
                logsys.record(f"{name}{turn}",SR.get_mean(name),epoch)

        if (epoch%infer_epoch==0 and epoch!=0) or earlystopQ or epoch==epoches-1:
            model.G.eval()
            model.I2C.eval()

            figure=plt.figure(dpi=100)
            plt.errorbar(np.arange(len(mean)), mean, yerr = var,alpha=0.4,color='r')
            plt.savefig(os.path.join(demo_image_dir,f"statis_var_at{epoch}_{turn}"))
            logsys.add_figure(f'statis_var_{turn}',figure,epoch)

            figure=plt.figure(dpi=100)
            plt.errorbar(np.arange(len(mean)), mean, yerr = ku,alpha=0.4,color='r')
            plt.savefig(os.path.join(demo_image_dir,f"statis_kurtosis_at{epoch}_{turn}"))
            logsys.add_figure(f'statis_kurtosis_{turn}',figure,epoch)

            vectors,images   = get_data(infinity_prefetcher,epoch,device)
            fake_image_figure= have_a_image_snap(model,vectors,images,args)
            fake_image_name  = f'demo_train_at{epoch}'
            fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
            logsys.add_figure('demo_train',fake_image_figure,epoch)


            fake_image_figure= have_a_image_snap(model,valid_vector,valid_image,args)
            fake_image_name  = f'demo_valid_at{epoch}'
            fake_image_figure.savefig(os.path.join(demo_image_dir,fake_image_name))
            logsys.add_figure('demo_valid',fake_image_figure,epoch  )

            save_weight_name=f'epoch-{epoch}'
            model.save_to(os.path.join(save_weight_dir,save_weight_name),mode="light")
        if earlystopQ :
            print("=============================")
            print("=============================")
            print("====== early stop! ==========")
            break_epoch = epoch
            break

    logsys.save_scalars()
    logsys.recorder.close()

if __name__=="__main__":
    args = the_default_GAN_curve_config
    args = args.copy({
        'FEDMODEL_WEIGHT':"checkpoints/SMSDatasetB1NES32,curve,simple.multilength/Resnet18KSFNLeakReLUTend/on108000/10_28_17_33_01-seed-84259/best/epoch112.best_MSError_0.0042",
        'GAN_PRE_WEIGHT':"checkpoints/DCGAN_PATTERN.SMSDatasetB1NES32,curve,simple.multilength/DCGAN_m1to1_norm/11_14_00_55_58/weights/demo_train_at100",
        'TRAIN_MODE':"continue_train",
        'last_weight_path':"checkpoints/DCGAN_CURVE.SMSDatasetB1NES32,curve,simple.multilength/on108000/11_15_21_07_34/routine/11_16_18_44.epoch-2999"
    })
    train_curve_gan(args)
