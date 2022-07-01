import os,sys,time,glob,random,copy,logging,pickle,argparse,json
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from dataset import get_balance_2_classifier_dataset,get_FAST_B1NE_dataset,get_FAST_B1NE_dataset_online
from mltool.dataaccelerate import DataSimfetcher,infinite_batcher
from mltool.tableprint.printer import summary_table_info
from mltool.loggingsystem import LoggingSystem
from mltool.fastprogress import IN_NOTEBOOK
import nas.train_utils as train_utils

#with open(f".DATARoot.json",'r') as f:RootDict=json.load(f)
from config.config import RootDict
ROOTDIR  = RootDict['DATAROOT']
SAVEROOT = RootDict['SAVEROOT']
SAVEROOT = os.path.join(SAVEROOT,"checkpoints")
def search_model(args):

    genotype_string=None
    if hasattr(args.model,'arch') and (args.model.arch is not None):
        if os.path.isdir(args.model.arch):
            # use searched result, we will get
            with open(os.path.join(args.model.arch,"config.json"),'r') as f:search_args=json.load(f)
            args.train.init_channels= search_args['train']['init_channels']
            args.train.nodes        = search_args['train']['nodes']
            args.train.layers       = search_args['train']['layers']
            args.run.dataset        = search_args['run']['dataset']
            print(f"we will use the searched model I{args.train.init_channels}_N{args.train.nodes}_L{args.train.layers} at Dataset:{args.run.dataset} ")
            genotypes_file = os.path.join(args.model.arch,"genotype")
            with open(os.path.join(args.model.arch,"best/genotype"),'r') as f:
                genotype_string=(f.read())
            args.model.arch = f"I{args.train.init_channels}_N{args.train.nodes}_L{args.train.layers}_{str(abs(hash(genotype_string)))}"
        else:
            genotype_string = "genotypes.%s" % args.model.arch

    if args.run.seed == "random":
        args.run.seed = random.randint(1, 100000)
    if args.run.last_ckpt:
        save_dir = args.run.last_ckpt
        load_trained_model = True
    else:
        timenow=time.strftime("%Y%m%d%H%M%S")
        if args.run.test_mode:timenow="test"
        if args.model.arch is not None:
            save_dir = os.path.join(SAVEROOT,f"train-{args.search.search_space}-{args.data.dataset}")
            save_dir = os.path.join(save_dir,args.model.arch,f"{timenow}-{args.run.seed}")
        else:
            save_dir = os.path.join(SAVEROOT,f"search-{args.search.search_space}-{args.search.method}-{args.data.dataset}")
            save_dir = os.path.join(save_dir,args.search.op_names,f"{timenow}-{args.run.seed}")
        load_trained_model = False

    logsys = LoggingSystem(True,save_dir,seed=None)
    rng_seed = train_utils.RNGSeed(args.run.seed)
    config_save_path = os.path.join(save_dir,"config.json")
    if not os.path.exists(config_save_path):args.save(config_save_path)

    if (args.continue_mode=="get_genotype") or (args.continue_mode=="continue"):
        assert args.run.last_ckpt
        old_args  =  args.load(config_save_path,keys=['run','search','train','model','data','scheduler'])
        old_args.train.epochs = args.train.epochs
        old_args.run.test_mode = args.run.test_mode
        continue_mode = args.continue_mode
        args = old_args
        args.continue_mode = continue_mode
        print("====> use old args")
    else:
        print("====> use new args")
    logsys.info(args)

    genotype=None
    if hasattr(args.model,'arch') and (args.model.arch is not None):
        from search_spaces.operation import genotypes_searched as genotypes
        genotype = eval("genotypes.%s" % args.model.arch)

    summary_dir = os.path.join(save_dir, "summary")
    if not os.path.exists(summary_dir):os.makedirs(summary_dir)
    logsys.info(f"checkpoint dir:{save_dir}")


    ####### <------ dataset loader ---------> #######

    dataset_train,dataset_valid = get_dataset(args)
    print(f"trainset length:{len(dataset_train)}")
    print(f"validset length:{len(dataset_valid)}")
    train_queue = torch.utils.data.DataLoader(dataset_train, batch_size=args.train.batch_size,pin_memory=False,shuffle=True, num_workers=1)
    #train_queue = [dataset_train.vector[:args.batch_size].cuda(),dataset_train.imagedata[:args.batch_size].cuda()]
    valid_queue = torch.utils.data.DataLoader(dataset_valid, batch_size=args.train.batch_size,pin_memory=False, num_workers=1)
    infinity_prefetcher_valid=infinite_batcher(valid_queue)
    ####### <------ dataset loader ---------> #######

    ####### <------ monitor loader ---------> #######
    args.epochs  = args.train.epochs
    accu_list = dataset_train.accu_list#["ClassifierA","ClassifierP","ClassifierN"]
    saved_accu_type = accu_list[0]

    logsys.Q_batch_loss_record=True
    metric_dict       = logsys.initial_metric_dict(accu_list)
    writer            = logsys.create_recorder(hparam_dict={},metric_dict=metric_dict.recorder_pool)

    ####### <------ monitor loader ---------> #######

    ####### <------ model and train loader ---------> #######
    Network,Architect = choose_network_and_architect(args)
    criterion         = dataset_train.criterion(args.data.criterion_type)()
    print("===========================")
    print("the criterion is:")
    print(criterion)
    print("===========================")
    #print(np.prod(dataset_train.curve_type_shape))
    model = Network(
        args.model.init_channels,
        np.prod(dataset_train.curve_type_shape),
        args.model.nodes,
        args.model.layers,
        **{ "criterion": criterion,
            "genotype" : genotype,
            "auxiliary": args.model.auxiliary,
            "search_space_name": args.search.search_space,
            "exclude_zero": args.search.exclude_zero,
            "track_running_stats": args.search.track_running_stats,
            "op_names":args.search.op_names,#<--- use this one to choose different module
            "predict_type":f"{dataset_train.type_predicted}+{dataset_train.normf}",
        }
    )
    from mltool.universal_model_util import get_model_para_detail

    # checkpoint = torch.load('test.pt')
    # for name,p in checkpoint['state_dict'].items():
    #     if "cells.0._ops.0" not in name:continue
    #     #if "weight" not in name:continue
    #     print("{:40} {:5} {}".format(name,np.prod(p.shape),p.shape))
    # layer = model.cells[0]._ops[0]
    # print(layer)
    # print(get_model_para_detail(layer))
    # raise
    model = model.cuda()
    criterion = criterion.cuda()
    #model.store_init_weights() !!!! notice this should be claimt in Network Module define
    optimizer, scheduler = train_utils.setup_optimizer(model, args)
    architect = Architect(model, args, writer) if hasattr(args.model,'arch') and (args.model.arch is None) else None

    ####### <------ model and train loader  ---------> #######

    start_epochs = 0
    metric_dict_ckpt=None
    history=None
    #save_dir = "/data/Metasurface/checkpoints/search-pcdarts-eedarts-msdataRCurve32/PCDARTS_SYMMETRY_P4Z2/20210803142508-540-origin"
    if load_trained_model:
        print(f"load state_dict from {save_dir}")
        start_epochs ,metric_dict_ckpt= train_utils.load(save_dir, rng_seed, model, optimizer, architect)
        print(f"load successfully at epoch{start_epochs}")
        print(f"if you set a dynamic drop_path_prob change via epoch, please check the consistency bewteen your  first and second train.")
        if metric_dict_ckpt is not None: metric_dict.load(metric_dict_ckpt)
        if (args.continue_mode=="get_genotype"):
            print(architect.genotype())
            for deleted_zero in [0,1]:
                for withnone in [0,1]:# we would use automate rule to deal with none case
                    genotype = architect.genotype(withnone=withnone,deleted_zero=deleted_zero)
                    name     = f"genotype.deleted_zero_{deleted_zero}.withnone_{withnone}"
                    print(name)
                    print(genotype)
                    with open(os.path.join(save_dir, name),'w') as f:
                        f.write(f"{genotype}")
            graph_fig_normal,graph_fig_reduce= architect.weight_distribution()
            logsys.add_figure('best_weight_normal',graph_fig_normal,start_epochs)
            logsys.add_figure('best_weight_reduce',graph_fig_reduce,start_epochs)
            raise


    FULLNAME   = f"{Network.__name__}-{args.data.dataset}-{args.run.seed}"
    banner            = logsys.banner_initial(args.train.epochs,FULLNAME)
    master_bar        = logsys.create_master_bar(args.train.epochs)
    logsys.train_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=train_queue.batch_size)
    logsys.valid_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=valid_queue.batch_size)

    model.criterion = criterion
    model.accu_list = accu_list
    scheduler.last_epoch = start_epochs -1
    drop_out_limit  = 500
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes._base').disabled = True

    for epoch in master_bar:
        if epoch<start_epochs:continue
        lr = scheduler.get_lr()[0]
        logsys.record('the_lr_use_now',lr, epoch)
        model.drop_path_prob = args.train.drop_path_prob * epoch / args.train.epochs if epoch<drop_out_limit else args.train.drop_path_prob
        #<--------------- update model  --------------------->
        train_loss=train_obj = train(args,train_queue, infinity_prefetcher_valid, model,architect, optimizer, lr,epoch,logsys)
        #train_loss=train_obj=-1
        logsys.record('training_loss', train_loss, epoch)
        #logsys.info('epoch %d lr %e', epoch, lr)
        if "update_lr_state" in dir(scheduler):scheduler.update_lr_state(train_obj)
        #<--------------- update architect --------------------->
        if architect is not None:
            architect.baseline = train_obj
            #architect.update_history()
            #architect.log_vars(epoch, writer)
            #genotype = architect.genotype()
            #logsys.info(f"genotype:{genotype}")
            graph_fig_normal,graph_fig_reduce= architect.weight_distribution()
            logsys.add_figure('weight_normal',graph_fig_normal,epoch)
            logsys.add_figure('weight_reduce',graph_fig_reduce,epoch)
        #<---------------  update valid --------------------->
        valid_acc_pool = infer(valid_queue,model,epoch=epoch,logsys=logsys,discrete=args.search.discrete)
        #<---------------         --------------------->
        for accu_type in accu_list:logsys.record(accu_type, valid_acc_pool[accu_type], epoch)
        update_accu    = logsys.metric_dict.update(valid_acc_pool,epoch)
        if update_accu[saved_accu_type]:
            train_utils.save(save_dir,epoch+1,rng_seed,model,optimizer,
                            architect=architect,save_history=(architect is not None),
                            metric_dict=metric_dict,level="best")
        logsys.banner_show(epoch,FULLNAME)
        if args.run.test_mode:raise
        train_utils.save(save_dir,epoch+1,rng_seed,model,optimizer,
                         architect=architect,metric_dict=metric_dict,
                         save_history=(architect is not None))
        #raise

def get_dataset(args):
    if args.data.dataset == "msdata":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='T')
    elif args.data.dataset == "msdataR":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='R')
    elif args.data.dataset == "msdataT":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='T')
    elif args.data.dataset == "msdataR_PLG250":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='R',dataset='PLG',range_clip=[751,1001])
    elif args.data.dataset == "msdataT_PLG250":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='T',dataset='PLG',range_clip=[751,1001])
    elif args.data.dataset == "msdataR_PTN":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='R',dataset='PTN')
    elif args.data.dataset == "msdataT_PTN":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='T',dataset='PTN')
    elif args.data.dataset == "msdataR_PLR250":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='R',dataset='PLR',range_clip=[751,1001])
    elif args.data.dataset == "msdataT_PLR250":
        dataset_train,dataset_valid = get_balance_2_classifier_dataset(curve_branch='T',dataset='PLR',range_clip=[751,1001])
    elif args.data.dataset == "msdataRCurve32":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='R',
                                        dataset='RDN',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero')
    elif args.data.dataset == "msdataTCurve32":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='T',
                                        dataset='RDN',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero')
    elif args.data.dataset == "msdataRCurve32PLG":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='R',
                                        dataset='PLG',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero')
    elif args.data.dataset == "msdataTCurve32PLG":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='T',
                                        dataset='PLG',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero')
    elif args.data.dataset == "msdataRCurve32PLG250":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='R',
                                        dataset='PLG',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero',range_clip=[751,1001])
    elif args.data.dataset == "msdataTCurve32PLG250":
        dataset_train,dataset_valid = get_FAST_B1NE_dataset(curve_branch='T',
                                        dataset='PLG',FeatureNum=32,
                                        download=True,
                                        normf='mean2zero',range_clip=[751,1001])
    else:
        raise NotImplementedError
    if args.model.arch is None:  # this mean we are in searching mode,
        import copy
        dataset_valid            = copy.deepcopy(dataset_train)
        totally_train_length     = len(dataset_train)
        dataset_train.pick_index = range(3000,totally_train_length)
        dataset_valid.pick_index = range(3000)
    return dataset_train,dataset_valid


def choose_network_and_architect(args):
    # choose architect
    if   args.search.method in ["edarts", "gdarts", "eedarts"]:
        from nas.architect.architect_edarts import ArchitectEDARTS as Architect
    elif args.search.method in ["darts", "fdarts"]:
        from nas.architect.architect_darts import ArchitectDARTS as Architect
    elif args.search.method == "egdas":
        from nas.architect.architect_egdas import ArchitectEGDAS as Architect
    elif args.search.method == "msgdas":
        from nas.architect.architect_msdarts import ArchitectMSDARTS as Architect
    else:
        print(f"need [args.search.method] given {args.search.method}")
        raise NotImplementedError
    # choose search_space
    if args.search.search_space in ["darts", "darts_small"]:
        from nas.search_spaces.darts_search import DARTSNetwork as Network
    elif args.search.search_space == "pcdarts":
        from nas.search_spaces.pc_darts_search import PCDARTSNetwork as Network
    elif args.search.search_space == "msdarts":
        from nas.search_spaces.ms_darts_search import MSDARTSNetwork as Network
    else:
        print(f"need [args.search.search_space] given {args.search.search_space}")
        raise NotImplementedError
    return Network,Architect

def train(args,train_queue, infinity_prefetcher_valid, model,architect, optimizer, lr,epoch,logsys):
    logsys.Q_batch_loss_record = True
    objs = train_utils.AvgrageMeter()
    precs = [train_utils.AvgrageMeter() for i in model.accu_list]
    model.train()
    logsys.train()
    batches    = len(train_queue)
    device     = next(model.parameters()).device
    prefetcher = DataSimfetcher(train_queue,device)
    inter_b    = logsys.create_progress_bar(batches)
    while inter_b.update_step():
        target,input                  = prefetcher.next()
        if architect is not None:
            target_search,input_search    = infinity_prefetcher_valid.next()[:2]
            input_search                  = input_search.to(device=device,non_blocking=True)
            target_search                 = target_search.to(device=device,non_blocking=True)
            architect.step(input,target,input_search,target_search,
                        **{
                            "eta": lr,
                            "network_optimizer": optimizer,
                            "unrolled": args.search.unrolled,
                            "update_weights": True,
                        }
                    )
            optimizer.zero_grad()
            architect.zero_arch_var_grad()
            architect.set_model_alphas()
            architect.set_model_edge_weights()
        if architect is None:optimizer.zero_grad()
        if args.model.auxiliary:
            logits, logits_aux = model(input, discrete=args.search.discrete)

            loss               = model.criterion(logits, target)
            loss_aux = model.criterion(logits_aux, target)
            loss += args.model.auxiliary_weight * loss_aux
        else:
            logits = model(input, discrete=args.search.discrete)
            if isinstance(logits,tuple):logits=logits[0]
            loss   = model.criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.scheduler.grad_clip)
        optimizer.step()

        n      = input.size(0)
        objs.update(loss.item(), n)
        logsys.batch_loss_record([loss])
        outstring='[train] loss:{:.4f}'.format(objs.avg)
        inter_b.lwrite(outstring, end="\r")
        if args.run.test_mode:return -1
        #logsys.info('[train] epoch:{}_{} loss:{:.4f} '.format(epoch,inter_b.now,objs.avg))
    return objs.avg

def infer(valid_queue, model, epoch,logsys,discrete):
    objs = train_utils.AvgrageMeter()
    precs = [train_utils.AvgrageMeter() for i in model.accu_list]
    model.eval()
    logsys.eval()
    batches    = len(valid_queue)
    device     = next(model.parameters()).device
    prefetcher = DataSimfetcher(valid_queue,device)
    inter_b    = logsys.create_progress_bar(batches)
    #for step, (target, input,_) in enumerate(valid_queue):
    targets=[]
    logitss=[]
    with torch.no_grad():
        while inter_b.update_step():
            target, input   = prefetcher.next()[:2]
            logits    = model(input, **{"discrete": discrete})
            if isinstance(logits,tuple):logits=logits[0]
            #loss      = model.criterion(logits, target)
            targets.append(target)
            logitss.append(logits)
        target = torch.cat(targets)
        logits = torch.cat(logitss)
        accu_pool=valid_queue.dataset.computer_accurancy([target.cpu(),logits.cpu(),None],accu_list=model.accu_list)

    #logsys.info(f'[valid] epoch:{epoch} '+ " ".join(["{}:{:.4f}".format(accu_type,accu) for accu_type,accu in accu_pool.items()]))
    return accu_pool
