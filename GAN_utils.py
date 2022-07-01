import os,json,time,math,shutil,random, socket

import torch
import numpy as np
from dataset import random_h_flip,random_v_flip
from config import Config,loss_machine_pool
from mltool.ModelArchi.GANModel.SNdcgan import DCGAN_MODEL
from mltool.dataaccelerate import DataSimfetcher,infinite_batcher
from config_read import read_config
from train_base import struct_config
from plot_demo import show_gan_image_demo,plot16x16,plt
with open(".DATARoot.json",'r') as f:RootDict=json.load(f)
DATAROOT  = RootDict['DATAROOT']
SAVEROOT  = RootDict['SAVEROOT']

def one_component(origin_tensor):
    assert isinstance(origin_tensor,torch.Tensor)
    tensor = torch.nn.functional.pad(origin_tensor,(1,1,1,1))
    w,h=tensor.shape[-2:]
    #change zero pole
    zero_pole = tensor[...,:w-2,1:h-1]*tensor[...,1:w-1,:h-2]*tensor[..., 2:w,1:h-1]*tensor[...,1:w-1,2:h]
    zero_pole = zero_pole*(1-tensor[...,1:w-1,1:h-1])
    #change one pole
    one_pole = (1-tensor[...,:w-2,1:h-1])*(1-tensor[...,1:w-1,:h-2])*(1-tensor[..., 2:w,1:h-1])*(1-tensor[...,1:w-1,2:h])
    one_pole = one_pole*(tensor[...,1:w-1,1:h-1])
    return origin_tensor+zero_pole-one_pole
def tround(tensor,val=0.5):
    t=tensor*0
    t[tensor<val]=0
    t[tensor>=val]=1
    return t

### get model and_dataloader
class reconstruct:
    def __init__(self,method):
        self.method = method
    def __repr__(self):
        return f"reconstructer {method}"
    def __call__(self,data):
        if   self.method == "[-1,1]->[0,1]":
            return data*0.5+0.5
        elif self.method == "[0,1]->[-1,1]":
            return (data-0.5)/0.5
        elif self.method == "[-1,1]->[0,1]->OC":
            return one_component(data*0.5+0.5)
        elif self.method == "[0,1]->OC":
            return one_component(data)
def get_model(args,Image2Curve):
    GANMODEL_args=Config({})
    GANMODEL_args.channels   = 1
    GANMODEL_args.GAN_TYPE = args.model.GAN_TYPE #'DCGAN'
    model = DCGAN_MODEL(GANMODEL_args)
    model.d_optimizer = torch.optim.Adam(model.D.parameters(), lr=args.train.d_lr, betas=(0.5, 0.999))
    model.g_optimizer = torch.optim.Adam(model.G.parameters(), lr=args.train.g_lr, betas=(0.5, 0.999))

    if args.GAN_stratagy == "GAN_CURVE":
        model.I2C=Image2Curve
        model.I2C.load_from(args.model.FEDMODEL_WEIGHT)
        model.c_optimizer =  torch.optim.SGD(model.G.parameters(), lr=args.train.c_lr)

    if args.TRAIN_MODE in ["new_start"]:
        if args.model.GAN_PRE_WEIGHT is not None:
            print("Warning:detact use pre-train GAN branch as start")
            print(f"load pre-train weight at {args.model.GAN_PRE_WEIGHT}")
            modelGD_state_dict=torch.load(args.model.GAN_PRE_WEIGHT)
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
    PROJECTROOT      = args.model.FORWARD_PROJECT_ROOT
    FEDMODEL_WEIGHT  = args.model.FEDMODEL_WEIGHT
    GAN_TYPE         = args.model.GAN_TYPE # DCGAN
    GAN_PRE_WEIGHT   = args.model.GAN_PRE_WEIGHT

    last_weight_path = args.last_weight_path

    d_lr             = args.train.d_lr
    g_lr             = args.train.g_lr
    c_lr             = args.train.c_lr
    batch_size       = args.train.batch_size

    image2curve_project_json= os.path.join(PROJECTROOT,"project_config.json")
    project_config          = read_config(image2curve_project_json)
    project_config.model.pre_train_weight=None
    Image2Curve,project,db  = struct_config(project_config,db = None,build_model=(args.GAN_stratagy == "GAN_CURVE"))
    if db.dataset_train.normf == "mean2zero":
        if args.model.CURVE_REGION != "m1to1":
            print("!!!!Warning: the dataset norm is mean2zero, which mean the curve value is in [-1,1].")
            print("However,  you assign the curve region {self.CURVE_REGION}")
    elif db.dataset_train.normf == "none":
        if args.model.CURVE_REGION != "m1to1":
            print("!!!!Warning: the dataset norm is none, which mean the curve value is in [0,1].")
            print("However,  you assign the curve region {self.CURVE_REGION}")

    train_loader = project.train_loader
    valid_loader = project.valid_loader
    if batch_size is not None:
        del train_loader
        train_loader = torch.utils.data.DataLoader(dataset=db.dataset_train,batch_size=batch_size,pin_memory=True,shuffle=True)
        del valid_loader
        valid_loader = torch.utils.data.DataLoader(dataset=db.dataset_valid,batch_size=batch_size)
        if valid_loader.batch_size>batch_size:
            valid_loader = torch.utils.data.DataLoader(dataset=db.dataset_train,batch_size=batch_size)

    model      = get_model(args,Image2Curve)
    model.reconstruct = reconstruct(args.Gnorm_type)
    device     = next(model.D.parameters()).device
    args.device=device
    return model,train_loader,valid_loader

def get_data(infinity_prefetcher,epoch,args):
    device = args.device
    vectors,images = infinity_prefetcher.next()
    images         = images.to(device)
    vectors        = vectors.to(device)
    if args.data.input_image_type == "-1to1":
        images = (images-0.5)/0.5
    if args.data.input_image_flip:
        random_v_flip(images)
        random_h_flip(images)
    if args.data.input_image_relax:
        noises = torch.randn_like(images)
        coeff  = 0.3*np.exp(-epoch/10)
        images = (1-coeff)*images+coeff*noises
    return vectors,images


def load_config_from_json(path):
    with open(path,'r',encoding='utf-8') as f:config_dict=json.load(f)
    return Config(config_dict)
def load_checkpoints(args,verbose=True):
    TRAIN_MODE       = args.TRAIN_MODE
    if TRAIN_MODE in ["continue_train","reproduce","divarication"]:
        assert args.last_weight_path is not None
        print("Warning: you are now try to continue train a project")
        old_save_checkpoint   = os.path.dirname(os.path.dirname(args.last_weight_path))
        old_project_json_file = os.path.join(old_save_checkpoint,"project_config.json")
        old_args = load_config_from_json(project_json_file)
        if TRAIN_MODE == "continue_train":
            print(f"you choose continue train, we will reload status from last checkpoint")
            print(f"checkpoint path:{args.last_weight_path}")
            print(f"please check the model, optimizer, RDN status")
            sameQ = Config.check_different(old_args,args)
            if not sameQ:
                print(f"detected different from the set args and json file in checkpoint")
                print(f"we will use old args, if you want to use new args, please use TRAIN_MODE:divarication")
            args = old_args
            args.save_checkpoint = old_save_checkpoint
        elif  TRAIN_MODE == "reproduce":
            print(f"you choose reproduce train, we will only reload the config from checkpoints")
            print(f"checkpoint path:{args.last_weight_path}")
            print(f"please care the RDN seed set")
            TRAIN_MODE = "new_start"
            args = old_args
            args.last_weight_path = None
        elif  TRAIN_MODE == "divarication":
            print(f"you choose divarication train, we will use new args and continue training task in a new project")
            print(f"checkpoint path:{args.last_weight_path}")
            TRAIN_MODE = "new_start"

    if TRAIN_MODE in ["new_start"]:

        print(f"initalize a new {args.GAN_stratagy} training")
        if args.data.use_soft_label:
            assert (args.model.GAN_TYPE == "DCGAN_M")
        PROJECTROOT      = args.model.FORWARD_PROJECT_ROOT = os.path.dirname(os.path.dirname(args.model.FEDMODEL_WEIGHT))
        DATASETNAME      = PROJECTROOT.split('/')[1]
        FWDMODEL         = PROJECTROOT.split('/')[-2]
        GAN_stratagy     = args.GAN_stratagy
        train_type     = f'{GAN_stratagy}.{DATASETNAME}'
        model_tyep     = f'{args.model.GAN_TYPE}.{FWDMODEL}' if  args.GAN_stratagy == "GAN_CURVE" else args.model.GAN_TYPE
        args.FULLNAME  = f'{model_tyep}.{DATASETNAME}'
        TIME_NOW       = time.strftime("%m_%d_%H_%M_%S")
        if args.name_rule == 'follow_host':
            TRIAL_NOW      = '{}-host-{}'.format(TIME_NOW,socket.gethostname())
        elif args.name_rule == 'test':
            TRIAL_NOW = "test"
        else:
            TRIAL_NOW = TIME_NOW
        save_checkpoint= os.path.join(SAVEROOT,'checkpoints',train_type, f'{model_tyep}',TRIAL_NOW)
        args.save_checkpoint = save_checkpoint
        project_json_file = os.path.join(save_checkpoint,"project_config.json")
        print(args)
        args.save(project_json_file)
        print(f'new checkpoint file created, we save the configuration into {project_json_file}')

    if args.data.input_image_type == "-1to1":
        args.Gnorm_type  = "[-1,1]->[0,1]"
    else:
        raise NotImplementedError
    GAN_PRE_WEIGHT   = args.model.GAN_PRE_WEIGHT
    d_lr             = args.train.d_lr
    g_lr             = args.train.g_lr
    c_lr             = args.train.c_lr
    disc_iter        = args.train.disc_iter
    gen_iter         = args.train.gen_iter
    hparam_dict = {'d_lr': d_lr,'g_lr': g_lr,'c_lr':c_lr,'disc_iter':disc_iter,'gen_iter':gen_iter,
                   'G_metrics':args.model.loss_tasks_G['name'],
                   'TWOCC':"OC" in args.Gnorm_type,
                   "GAN_from_new":1 if GAN_PRE_WEIGHT is None else 0
                   }

    print("============== Config ==================")
    print(f"Project at {save_checkpoint}")
    if verbose:args.disp()
    return args,save_checkpoint,hparam_dict

def infer_pattern_gan(save_checkpoint):
    #save_checkpoint = "checkpoints/DCGAN_PATTERN.SMSDatasetB1NES32,curve,simple.multilength/DCGAN_m1to1_norm/11_12_23_15_31"
    args = the_default_GAN_pattern_config
    demo_image_dir = os.path.join(save_checkpoint,'demo')
    save_weight_dir= os.path.join(save_checkpoint,'weights')
    if not os.path.exists(save_weight_dir):os.makedirs(save_weight_dir)
    if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)

    model    = get_model(args)
    args.device = next(model.D.parameters()).device
    logsys   = LoggingSystem(True,save_checkpoint)
    _   = logsys.create_recorder(hparam_dict={},metric_dict={})
    for p in os.listdir(save_weight_dir):
        epoch = epoch = int(re.findall(r"at(.*)", p)[0])
        weight_path = os.path.join(save_weight_dir,p)
        z = get_lattened_vector(2000,args)
        fake_images = model.G(z)
        figure=plt.figure(dpi=100)
        errorbarplot(fake_images.reshape(-1,256).detach().cpu().numpy())
        plt.savefig(os.path.join(demo_image_dir,f"statis_info_at{epoch}"))
        logsys.add_figure('statis_info',figure,epoch)


def inference_image(data,epoch,logsys,turn=0,flag="test"):
    demo_image_dir = os.path.join(logsys.ckpt_root,'demo')
    if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)
    score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED = data
    theimages   = I_FAKE.reshape(-1,256)
    mean        = np.mean(theimages,0)
    var         = np.var(theimages,0)
    ku          = np.mean((theimages - mean) ** 4,0) / (var**2+0.01)

    figure=plt.figure(dpi=100)
    plt.errorbar(np.arange(len(mean)), mean, yerr = var,alpha=0.4,color='r')
    plt.savefig(os.path.join(demo_image_dir,f"statis_var_at{epoch}_{turn}"))
    logsys.add_figure(f'statis_var_{turn}_{flag}',figure,epoch)

    figure=plt.figure(dpi=100)
    plt.errorbar(np.arange(len(mean)), mean, yerr = ku,alpha=0.4,color='r')
    plt.savefig(os.path.join(demo_image_dir,f"statis_kurtosis_at{epoch}_{turn}"))
    logsys.add_figure(f'statis_kurtosis_{turn}_{flag}',figure,epoch)

    random_sample=np.random.choice(range(len(I_PRED)),16,replace=False).tolist()
    images = I_PRED[random_sample]
    fake_image_figure= plot16x16(images.reshape(16,16,16))
    fake_image_figure.savefig(os.path.join(demo_image_dir,f'demo_image_at{epoch}_{turn}'))
    logsys.add_figure(f'demo_image_{turn}_{flag}',fake_image_figure,epoch)

def inference_curve(data,epoch,nrows,ncols,logsys,turn=0,flag="test"):
    demo_image_dir = os.path.join(logsys.ckpt_root,'demo')
    if not os.path.exists(demo_image_dir):os.makedirs(demo_image_dir)
    score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED = data
    data = list(zip(score,real_v,fake_v,pred_v,I_REAL,I_FAKE,I_PRED))
    fake_image_figure= show_gan_image_demo(data,nrows=nrows,ncols=ncols)
    fake_image_name  = f'demo_{flag}_at{epoch}'
    fake_image_figure.savefig(os.path.join(logsys.ckpt_root,'demo_image',fake_image_name))
    logsys.add_figure(f'demo_{flag}',fake_image_figure,epoch)



def save_model(model,path,mode='full'):
    checkpoint={}
    checkpoint['D_state_dict']    = model.D.state_dict()
    checkpoint['G_state_dict']    = model.G.state_dict()
    if mode != 'light':
        checkpoint['D_optimizer']     = model.d_optimizer.state_dict()
        checkpoint['G_optimizer']     = model.g_optimizer.state_dict()
    torch.save(checkpoint,path)
def save_hyparam(_dict,path):
    with open(path,'w') as f:json.dump(_dict,f)
def load_hyparam(path):
    with open(path,'r') as f:
        return json.load(f)
def calculate_gradient_penalty(Discriminator,real_images, fake_images,args):
    args.train.GP_lambda = 0.2
    batch_size = len(real_images)
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(args.device)
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(args.device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = Discriminator(interpolated)
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(args.device),
                               create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.train.GP_lambda
    return grad_penalty

class ModeCollapseError(NotImplementedError):pass
class RestartError(NotImplementedError):pass
def check_mode_collapse(fakebinary,check_diff_num,eps=0.01):
    check_diff_random_sample= np.random.choice(range(len(fakebinary)),check_diff_num,replace=False).tolist()
    check_diff_index_1      = [check_diff_random_sample[i] for i in range(check_diff_num) for j in range(i+1)]
    check_diff_index_2      = [check_diff_random_sample[j] for i in range(check_diff_num) for j in range(i+1)]
    different    = torch.nn.MSELoss()(fakebinary[check_diff_index_1].detach(),fakebinary[check_diff_index_2].detach()).item()
    if different < 0.1:
        return True
    else:
        return False
