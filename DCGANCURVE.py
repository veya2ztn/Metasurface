import torch
import torch.nn as nn
from torch.autograd import Variable

from mltool.ModelArchi.GANModel.SNdcgan import DCGAN_MODEL
from mltool.fastprogress import master_bar,progress_bar
from mltool.loggingsystem import LoggingSystem
from mltool.visualization import *

import re,os,json,random,time
import numpy as np
from GAN_utils import *

from model.criterion import SSIMError,SelfEnhanceLoss3,VaryingLoss,MeanLoss,Kurtosis
from config import the_polygon_250_curve_config,the_default_GAN_pattern_config

# torch.backends.cudnn.enabled  = False
# torch.backends.cudnn.benchmark=True

def train_phase(model,mode):
    model.G.train()
    model.D.train()
    model.G.zero_grad()
    model.D.zero_grad()

    if hasattr(model,'I2C'):
        model.I2C.eval()
        for p in model.I2C.parameters():p.requires_grad = False
    if mode=='D':
        for p in model.D.parameters():p.requires_grad = True
        for p in model.G.parameters():p.requires_grad = False
    elif mode =='G':
        for p in model.D.parameters():p.requires_grad = False
        for p in model.G.parameters():p.requires_grad = True
    else:
        raise NotImplementedError

def get_lattened_vector(curves,args):
    sample_num   = len(curves)
    curve_dim    = curves.shape[-1]
    device       = args.device
    lattendim    = args.model.lattendim
    CURVE_REGION = args.model.CURVE_REGION
    curves       = curves.to(device)
    if CURVE_REGION == "m1to1":
        z = torch.randn((sample_num, lattendim, 1, 1)).to(device)
    elif CURVE_REGION == "0to1":
        z = torch.rand((sample_num, lattendim, 1, 1)).to(device)
    if args.GAN_stratagy == "GAN_CURVE":
        z[:,:curve_dim,0,0]=curves[:,0,:]
    z = Variable(z)
    #if args.train.Dis.use_soft_label:
    if args.data.use_soft_label:
        real_labels  = torch.Tensor(z.size(0),1).uniform_(0.7,1.2).to(device)
        fake_labels  = torch.Tensor(z.size(0),1).uniform_(0.0,0.3).to(device)
    else:
        real_labels  =  torch.ones(z.size(0),1).to(device)
        fake_labels  = torch.zeros(z.size(0),1).to(device)
    return z,real_labels,fake_labels

def get_all_status(model,input_vector,input_image_data,args):
    model.G.eval()
    model.D.eval()

    realbinary  = model.reconstruct(input_image_data)
    I_REAL      = realbinary.cpu().detach().numpy()
    real_v      = input_vector[:,0].cpu().detach().numpy()
    z,real_labels,fake_labels = get_lattened_vector(input_vector,args) #(9,100,1,1)
    with torch.no_grad():
        fake_images = model.G(z)
        fakebinary0 = model.reconstruct(fake_images)
        fakebinary  = fakebinary0.round()
        I_FAKE      = fakebinary0.cpu().detach().numpy()
        I_PRED      = fakebinary.cpu().detach().numpy()

        if args.GAN_stratagy == "GAN_CURVE":
            fake_vectors= model.I2C(fakebinary0)
            pred_vectors= model.I2C(fakebinary)
            fake_v = fake_vectors[:,0].cpu().detach().numpy()
            pred_v = pred_vectors[:,0].cpu().detach().numpy()
            score = ((real_v-pred_v)**2).mean(1)
        else:
            score=real_v=fake_v=pred_v=None
        data  = (score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED)
    return data

####################
### LOAD MODEL #####
####################
def train_D_branch(model,infinity_prefetcher,epoch,logsys,args):
    logsys.train()
    train_phase(model,'D')
    device            = args.device
    fine_epoch        = args.fine_epoch
    #loss_tasks        = args.model.loss_tasks_D['config']


    vectors,real_images        = get_data(infinity_prefetcher,epoch,args)
    z,real_labels,fake_labels  = get_lattened_vector(vectors,args)
    fake_images = model.G(z)

    model.D.zero_grad()

    outputs     = model.D(fake_images)
    d_loss_fake = model.D.metric(outputs, fake_labels)
    d_loss_fake.backward()

    outputs     = model.D(real_images)
    d_loss_real = model.D.metric(outputs, real_labels)
    d_loss_real.backward()

    d_loss      = d_loss_real + d_loss_fake
    if model.D.version == "WGAN_GP":
        gradient_penalty = model.D.calculate_gradient_penalty(real_images.data, fake_images.data)
        gradient_penalty.backward()
        d_loss+=gradient_penalty

    #d_loss.backward()
    model.d_optimizer.step()

    logsys.record(f"d_loss_fake" ,d_loss_fake.item(),fine_epoch)
    logsys.record(f"d_loss_real" ,d_loss_real.item(),fine_epoch)
    logsys.record(f"d_loss_total",d_loss.item(),fine_epoch)
    args.fine_epoch+=1
    loss_list = [d_loss.item(),d_loss_fake.item(),d_loss_real.item()]
    return loss_list

def train_G_branch(model,infinity_prefetcher,epoch,logsys,args):
    logsys.train()
    train_phase(model,'G')
    device            = args.device
    fine_epoch        = args.fine_epoch
    loss_tasks        = args.model.loss_tasks_G['config']

    vectors,real_images    =  get_data(infinity_prefetcher,epoch,args)
    z,real_labels,fake_labels  = get_lattened_vector(vectors,args)

    fake_images = model.G(z)
    fake_sign   = model.D(fake_images)
    gen_loss    = model.D.metric(fake_sign,real_labels)
    logsys.record(f"label_gen_loss",gen_loss.item(),fine_epoch)
    t_loss      = gen_loss
    loss_list   = [gen_loss.item()]
    fakebinary  = model.reconstruct(fake_images)# depend on the I2C branch input requirement
    if args.GAN_stratagy == "GAN_CURVE":fake_vectors= model.I2C(fakebinary)
    for loss_object,loss_metric,coef in loss_tasks:
        if loss_object   == "vectors":
            if args.GAN_stratagy != "GAN_CURVE":continue
            pred = fake_vectors;real = vectors
        elif loss_object == "images":
            pred = fake_images;real = real_images
        elif loss_object == "IQuality":
            pred = fakebinary;real = None
        # elif loss_object == "labels":
        #     pred = fake_sign;real = real_labels
        else:continue
        loss = loss_machine_pool[loss_metric](pred,real)
        loss_list.append(loss.item())
        logsys.record(f"{loss_object}_{loss_metric}",loss.item(),fine_epoch)
        t_loss += coef*loss

    args.fine_epoch+=1
    model.G.zero_grad()
    t_loss.backward()
    model.g_optimizer.step()
    return [t_loss.item()]+loss_list

def test_branch(model,input_vector,input_images,accu_list,
                image_metric=SSIMError(),
                curve_metric=torch.nn.MSELoss()):
    model.G.eval()
    model.D.eval()
    z,real_labels,fake_labels  = get_lattened_vector(input_vector,args)
    score_pool={}
    with torch.no_grad():
        fake_images = model.G(z)
        fakebinary0 = model.reconstruct(fake_images)
        #realbinary  = model.reconstruct(input_images)
        if check_mode_collapse(fakebinary0,30,eps=0.1):raise ModeCollapseError
        realbinary  = input_images
        for accu in accu_list:
            split=accu.split('_')
            do_OC = False
            if len(split)==1:val = -1
            elif len(split)==2:val = float(split[1])
            elif len(split)==3:
                val = float(split[1])
                do_OC = (split[2]=="OC")
            fakebinary  = tround(fakebinary0,val) if val > 0 else fakebinary0
            if do_OC:
                fakebinary  = one_component(fakebinary)

            if args.GAN_stratagy == "GAN_CURVE":
                pred_vectors= model.I2C(fakebinary)
                score_pool[accu]=curve_metric(pred_vectors,input_vector).item()
            else:
                accu_image      = accu
                score_pool[accu]=image_metric(fakebinary,input_images).item()
    return score_pool
####################

####### 统一GAN flow中的image norm 和curve norm ######
def train_curve_gan(args):
    args,save_checkpoint,hparam_dict = load_checkpoints(args)

    save_weight_dir= os.path.join(save_checkpoint,'weights')
    if not os.path.exists(save_weight_dir):os.makedirs(save_weight_dir)

    # load data
    model,train_loader,valid_loader = load_model_and_dataloader(args)
    infinity_prefetcher=infinite_batcher(train_loader)

    doearlystop     = args.train.doearlystop # False
    infer_epoch     = args.train.infer_epoch
    valid_per_epoch = args.train.valid_per_epoch
    do_inference    = args.train.do_inference
    warm_up_epoch   = args.train.warm_up_epoch

    logsys          = LoggingSystem(True,save_checkpoint,seed=1000)
    accu_list     = ['val']+['val_0.5']+[f'val_{val}_OC' for val in args.train.threothod_list]
    metric_dict   = logsys.initial_metric_dict(accu_list)
    metric_dict   = metric_dict.metric_dict
    _             = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)
    epoches       = args.train.epoches
    save_accu_list= accu_list[:2]
    _             = logsys.create_model_saver(accu_list=save_accu_list,
                                            #earlystop_config=args.earlystop,
                                            epoches=epoches)

    checkpointer = {
        "model_D": model.D,
        "model_G": model.D,
        "optimizer_G": model.g_optimizer,
        "optimizer_D": model.d_optimizer
    }
    start_epoch = -1
    if args.last_weight_path is not None:
        start_epoch=logsys.load_checkpoint(checkpointer,args.last_weight_path)

    FULLNAME     = args.FULLNAME #
    banner       = logsys.banner_initial(epoches,FULLNAME)
    mb           = logsys.create_master_bar(epoches,banner_info="")
    earlystopQ   = False
    valid_vector = valid_loader.dataset.vector.to(args.device)
    valid_images = valid_loader.dataset.imagedata.to(args.device)
    args.fine_epoch = 0
    disc_iter=args.train.disc_iter
    gen_iter =args.train.gen_iter
    do_train =False
    for epoch in mb:
        if epoch<=start_epoch:continue
        if do_train:
            train_phase(model,'D')
            for d_iter in range(args.train.disc_iter):
                train_D_loss_list = train_D_branch(model,infinity_prefetcher,epoch,logsys,args)
                mb.lwrite("Step [{}:{}/{} d_loss:{:.4f}]".format(epoch,d_iter,disc_iter,train_D_loss_list[0]),end='\r')
            train_phase(model,'G')
            for g_iter in range(args.train.gen_iter):
                train_G_loss_list = train_G_branch(model,infinity_prefetcher,epoch,logsys,args)
                mb.lwrite("Step [{}:{}/{} g_loss:{:.4f}]".format(epoch,g_iter,gen_iter,train_G_loss_list[0]),end='\r')
            logsys.save_latest_ckpt(checkpointer,epoch)
        try:
            if epoch%valid_per_epoch ==0:
                valid_acc_pool = test_branch(model,valid_vector,valid_images,accu_list)
                update_accu    = logsys.metric_dict.update(valid_acc_pool,epoch)
                metric_dict    = logsys.metric_dict.metric_dict
                for accu_type in accu_list:
                    logsys.record(accu_type, valid_acc_pool[accu_type], epoch)
                    logsys.record('best_'+accu_type, metric_dict['best_'+accu_type][accu_type], epoch)
                #logsys.banner_show(epoch,FULLNAME,train_losses=[train_D_loss_list[0]+train_G_loss_list[0]])
                logsys.banner_show(epoch,FULLNAME,train_losses=[-1])
                earlystopQ = logsys.save_best_ckpt(checkpointer,metric_dict,epoch,doearlystop=doearlystop)

            #if do_inference and ((epoch%infer_epoch==0 and epoch!=0) or earlystopQ or epoch==epoches-1):
                data=(score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED) = get_all_status(model,valid_vector,valid_images,args)
                _=inference_image(data,epoch,logsys)
                if args.GAN_stratagy == "GAN_CURVE":
                    nrows,ncols = 2,4
                    random_sample=np.random.choice(range(len(I_REAL)),nrows*ncols,replace=False).tolist()
                    data=(score[random_sample],real_v[random_sample],fake_v[random_sample],pred_v[random_sample],
                                               I_REAL[random_sample],I_FAKE[random_sample],I_PRED[random_sample])
                    _=inference_curve(data,epoch,nrows,ncols,logsys)
                save_weight_name=f'epoch-{epoch}'
                model.save_to(os.path.join(save_weight_dir,save_weight_name),mode="light")
        except ModeCollapseError:
            if epoch < warm_up_epoch:continue
            print(f"detected mode collapse at epoch:{epoch},will reset random set and restart")
            raise RestartError

    logsys.save_scalars()
    logsys.recorder.close()

if __name__=="__main__":
    args = the_default_GAN_pattern_config

    train_curve_gan(args)
