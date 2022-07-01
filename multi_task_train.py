import os,sys,time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-td","--taskdir", default='projects/undo', type=str,help="default at project/undo")
parser.add_argument("-m","--mode", default="default", type=str,choices=["test", "multiTask","assign"],help=r'''
Choose the mode:
    【default】: train the tasks in project/undo one by one.
       【test】: run a test epoch to get the GPU memory usage. Meanwhile debug the code.
 【multiTask】: run task in project/muliti_set{}. Cooperate with `bash assignjob.sh`.
     【assign】: run specific task by assigned json file.
''')
parser.add_argument("-f","--file", default=False, type=str,choices=["continue", "copy", "replicate"],help='''
    do file operation on a given json task file.
          - 【continue】: generate a json task that conntinue train a model
          - 【copy】: copy a json task with same configuration.
          - 【replicate】: copy a json task with same configuration and same random seed.
''')
parser.add_argument("-p","--path", default=False, type=str,help='the task json file')
parser.add_argument("--gpu", default=0, type=int,help='the GPU used for tasks')
parser.add_argument("--movefinished", default=1, type=int,help='delete the finished task')
parser.add_argument("--comment", default="", type=str,help='description')
parser.add_argument("--offset", default=0, type=int,help='offset num for multiTask')
args = parser.parse_args()

PROJECTFILES = args.taskdir
if len(os.listdir(PROJECTFILES)) == 0 and args.mode == 'default' and not args.file:
    print("=== No project file. abort! ==")
    raise

import shutil
import socket
hostname = socket.gethostname()

from utils import mymovefile,makerecard
from mltool.sendinfo.send2phone import send_message
from model.model import NanValueError
from config import read_config,json_to_config

import torch.multiprocessing as mp
import torch
import random
import re

def get_jobs(path):return [ j for j in os.listdir(path) if j !='running']

if args.file:
    save_checkpoint     = args.path
    _,trial_num         = os.path.split(save_checkpoint.strip("/"))
    project_config_path = os.path.join(save_checkpoint,'project_config.json')
    project_config      = json_to_config(project_config_path)
    if args.file=="continue":
        project_config.random_seed = int(re.findall(r"seed-([0-9]*)", save_checkpoint)[0])
        project_config.last_checkpoint = save_checkpoint
        project_config.train_mode = "contine_train"
    if args.file=="replicate":
        project_config.random_seed = int(re.findall(r"seed-([0-9]*)", save_checkpoint)[0])
        project_config.train_mode = "replicate"
        project_config.train['trials']=1
    print(project_config)
    project_name = f"{trial_num}.{project_config.project_name}"
    project_config.save(os.path.join(f'{args.taskdir}/{project_name}.json'))
    print(f"regenerate project json file for project <{project_name}>")
    sys.exit(0)


job_gpu     = str(args.gpu)
twostepassign = False
if  args.mode == "multiTask":
    twostepassign=True
    RUNNING_DIR  = os.path.join(f'projects/multi_set{args.gpu+args.offset}','running')
    if not os.path.exists(RUNNING_DIR):os.makedirs(RUNNING_DIR)
    if len(os.listdir(RUNNING_DIR)):
        print('last task is exit error,please cheak')
        raise

os.environ["CUDA_VISIBLE_DEVICES"] = job_gpu

RECARDFILE   = "../LOGFILE/unrecard/"
db = None
BACKUP_DIR   = 'projects/back'
FAIL_DIR     = 'projects/fail'
DONE_DIR     = 'projects/done'
tested_model_pool={}

fouce_break   = False
torch.backends.cudnn.enabled  = True
torch.backends.cudnn.benchmark= False
torch.backends.cudnn.deterministic= True # the key for continue training.

while (len(get_jobs(PROJECTFILES))>0 or args.mode == "assign")and (not fouce_break):
    if  args.mode == "assign":
        project_config_path = args.path
        project_config_dir,project_config_name  = os.path.split(args.path)
        fouce_break = True
    else:
        project_config_dir  = PROJECTFILES
        project_config_name = get_jobs(PROJECTFILES)[0]
        project_config_path = os.path.join(PROJECTFILES,project_config_name)
    if 'json' not in project_config_name:
        print('{} is not a json form and while be remove to backup file'.format(project_config_name))
        mymovefile(project_config_path,os.path.join(BACKUP_DIR,project_config_name))
        continue


    project_config = read_config(project_config_path)

    project_config.gpu = str(args.gpu)
    if  args.mode == "test":
        from train_base import struct_config
        from train_base import test_GPU_memory_usage
        now_test_model = str(project_config.model.backbone_TYPE)
        if now_test_model in tested_model_pool:
            print("{} already tested --> pass".format(project_config.project_name))
            mymovefile(project_config_path,os.path.join(BACKUP_DIR,project_config_name))
            continue
        model,project,db=struct_config(project_config,db = db,build_model=True)
        test_GPU_memory_usage(model,project)
        print("============test pass!===============")
        mymovefile(project_config_path,os.path.join(BACKUP_DIR,project_config_name))
        tested_model_pool[now_test_model]=1
    else:
        if args.comment != "":project_config.comment = args.comment
        try:
            ### move to the running file
            if twostepassign:
                now_file_at = os.path.join(RUNNING_DIR,project_config_name)
                print(f"move {project_config_path} ==> {now_file_at}")
                mymovefile(project_config_path,now_file_at)
                project_config_path = now_file_at
            normal_train=True
            if hasattr(project_config,'PROJECT_TYPE'):
                if project_config.PROJECT_TYPE == "GAN_PATTERN":
                    from DCGANTRAIN import train_pattern_ga
                    print(">>>>>>> [ we now train pattern GAN] <<<<<<<")
                    normal_train=False
                    train_pattern_gan(project_config)
                elif project_config.PROJECT_TYPE == "GAN_CURVE":
                    from DCGANCURVE import train_curve_gan
                    print(">>>>>>> [ we now train curve GAN] <<<<<<<")
                    train_curve_gan(project_config)
                    normal_train=False
                elif project_config.PROJECT_TYPE == "NAS_SEARCH":
                    from darts_search import search_model
                    print(">>>>>>> [ we now serach best architext] <<<<<<<")
                    #print(project_config)
                    #raise
                    search_model(project_config)
                    normal_train=False
            if normal_train:
                from train_base import train_for_one_task
                from train_base import struct_config
                model,project,db=struct_config(project_config,db = db,build_model=False)
                project.project_json_config_path=project_config_path
                project_root_dir=train_for_one_task(model,project)
                if project_root_dir == "move_the_file":mymovefile(project_config_path,os.path.join(DONE_DIR,project_config_name))

                print("\n---------Trainning Finish----------------")
            #send_message(f"DONE!:{hostname}:{project_config_name} || remain:{len(get_jobs('projects/undo'))}")
            if args.movefinished:mymovefile(project_config_path,os.path.join(DONE_DIR,project_config_name))
        except NanValueError:
            print("==================================")
            print("=====<|{}|>======".format(project_config.project_name))
            print("we face a inf loss problem,this project fail please check and restart")
            content = "we face a inf loss problem,this project fail"
            #send_message(f"FAIL!:{hostname}:{project_config_name}|| remain:{len(get_jobs('projects/undo'))}")
            makerecard(project_config_name,False,RECARDFILE,content)
            mymovefile(project_config_path,os.path.join(FAIL_DIR,project_config_name))
            continue
        except KeyboardInterrupt:
            raise
        except:
            #send_message(f"FAIL!:{hostname}:{project_config_name} || remain:{len(get_jobs('projects/undo'))}")
            raise
