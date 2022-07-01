import json
import os
import copy

from model.SearchedDARTSmodel import OgDARTSResult
from dataset.dataset_module import SMSDatasetN,SMSDatasetC,SMSDatasetB1NES128

DATAROOTPATH = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/.DATARoot.json"
if os.path.exists(DATAROOTPATH):
    with open(DATAROOTPATH,'r') as f:
        RootDict=json.load(f)
else:
    RootDict = {"DATAROOT": "/root/data",
                "SAVEROOT": "/root/checkpoints",
                "EXP_HUB":"exp_hub"}

DATAROOT  = RootDict['DATAROOT']
SAVEROOT  = RootDict['SAVEROOT']
EXP_HUB   = RootDict['EXP_HUB']
class Config(object):
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def save(self,path=None):
        if path is None:
            path = 'projects/undo'
            path = os.path.join(path,self.project_json_name+'.json')
        if not os.path.exists(os.path.dirname(path)):os.makedirs(os.path.dirname(path))
        with open(path,'w',encoding='utf-8') as f:
            json.dump(self.to_dict(),f,indent=4,ensure_ascii=False)

    @staticmethod
    def load(path,keys=[]):
        with open(path,'r',encoding='utf-8') as f:
            config_dict=json.load(f)
        for k in keys:
            if isinstance(k,str):
                config_dict[k]=Config(config_dict[k])
            elif isinstance(k,list):
                key = k[0]

                sub_keys = k[1]
                for sub_key in sub_keys:
                    config_dict[key][sub_key]=Config(config_dict[key][sub_key])
                config_dict[key]=Config(config_dict[key])
        return Config(config_dict)

    @staticmethod
    def check_different(args1,args2):
        flag   = True
        _dict1 = args1.to_dict()
        _dict2 = args2.to_dict()
        for key, val in _dict1.items():
            if key not in _dict2:
                print(f"_dict2 lake key {key}")
            else:
                if val != _dict2[key]:
                    print(f"value different!!")
                    print(f"{key}:| {val} <---> {_dict2[key]}|")
                    flag = False
        for key, val in _dict2.items():
            if key not in _dict1:
                print(f"_dict1 lake key {key}")
                flag = False
        return flag

    def to_dict(self):
        _dict={}
        for k, v in vars(self).items():
            if isinstance(v,Config):
                _dict[k]=v.to_dict()
            else:
                _dict[k]=v
        return _dict

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = copy.deepcopy(self)
        for key, val in new_config_dict.items():
            if isinstance(val,dict) and key in ['train','model','data']:
                attr = ret.__getattribute__(key)
                for k, v in val.items():
                    attr.__setattr__(k, v)
                ret.__setattr__(key, attr)
            else:
                ret.__setattr__(key, val)
        return ret

    def update(self,new_config_dict):
        ret = copy.deepcopy(self)
        for key, val in new_config_dict.items():
            attr = ret.__getattribute__(key)
            assert isinstance(val,dict) and isinstance(attr,dict)
            for k, v in val.items():
                    attr[k]
            ret.__setattr__(key, attr)
        return ret
    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        _str=""
        for k, v in vars(self).items():
            if isinstance(v,Config):
                string_list=v.__repr__().split('\n')
                new_str = '\n'.join(['   |'+ s for s in string_list])
                _str+=k+' = '+'\n'
                _str+=new_str+'\n'
            else:
                v=str(v)
                _str+=k+' = '+v+'\n'
        return _str

    def disp(self):
        print(self.__repr__())

def ConfigCombine(config_pool):
    configcombine={}
    for name, config_list in config_pool.items():
        if name == "base":
            for config_old in config_list:
                config = copy.deepcopy(config_old)
                for key, val in vars(config).items():
                    if (key in configcombine) and (configcombine[key]!=val):
                        print(f"repeat key:[{key}] for ")
                        print(f"val1:{configcombine[key]}")
                        print(f"val2:{val}")
                        if isinstance(val,dict):
                            print("combine--->")
                            for k,v in val.items():
                                configcombine[key][k]=v
                        else:
                            print(f"use {configcombine[key]}")
                    else:
                        configcombine[key]=val
        else:
            configcombine[name]=ConfigCombine({"base":config_list})

    return Config(configcombine)

def parse_train(config):
    name = ""
    name += "on{}.".format(config.volume) if config.volume else ""
    real_name = name
    if config.hypertuner != "optuna":
        name +="op,{}".format(config.optimizer._TYPE_)
        name +='.lr,{:.5f}'.format(config.optimizer.config['lr'])
        name +=".dr,{:.1f}".format(config.drop_rate)  if config.drop_rate else ""
    else:
        name += "optuna"
    return name,real_name,config

def parse_model(config):
    name =real_name    = config.backbone_TYPE
    if hasattr(config,'backbone_alias') and config.backbone_alias  is not None:
        real_name = config.backbone_alias
    elif hasattr(config,'backbone_config') and config.backbone_config is not None:
        real_name = eval(config.backbone_TYPE).model_name(**config.backbone_config)
    #name += ".cr,{}".format(config.criterion_type) if config.criterion_type!="default" else ""
    return real_name,name,config

def parse_data_dict(user_dic):
    name = ""
    name += user_dic['global_dataset_type'] if 'global_dataset_type' in user_dic else ""
    dataset_name = user_dic['dataset_TYPE']
    if 'SMSDataset' in dataset_name:
        # filte 'Compact dataset setting'
        type_predicted  =user_dic['dataset_args']['type_predicted']
        target_predicted=user_dic['dataset_args']['target_predicted']
        if 'range_clip' in user_dic:
            user_dic['dataset_args']['range_clip'] = user_dic['range_clip']
            del user_dic['range_clip']

        dataset_flag = eval(dataset_name).get_dataset_name(**user_dic['dataset_args'])
        name += "SMSDataset{},{},{}".format(dataset_flag,type_predicted,target_predicted)

    else:
        if 'transform_config'  in user_dic:
            transformer_setting = user_dic['transform_config']
            transformer = eval(user_dic['transform_TYPE'])(**transformer_setting)
            feature_num = transformer.output_dim(user_dic['feature_num'])
            name="({}).{}".format(feature_num,transformer.name)
            user_dic['dataset_args']={'vec_dim':feature_num}
        else:
            name += dataset_name
    real_name = name
    if 'dataset_norm' not in user_dic:user_dic['dataset_norm']='none'

    if user_dic['dataset_norm']!='none':name+=".{}_norm".format(user_dic['dataset_norm'])
    if 'image_transfermer' in  user_dic:
        if user_dic['image_transfermer']!='none':
            name+=".{}_Image".format(user_dic['image_transfermer'])
            real_name+=".{}_Image".format(user_dic['image_transfermer'])
    user_dic['datasetname']=real_name
    return name,real_name,Config(user_dic)

def Merge(dataset_config,train_config,model_config):
    configcombine={}

    train_json_name,train_name,train = parse_train(train_config)
    model_json_name,model_name,model = parse_model(model_config)
    data_json_name ,data_name,data   = parse_data_dict(dataset_config.to_dict())
    trails = 'times{0:02d}'.format(train_config.trials) if hasattr(train_config,'trials') else 'times01'
    trails = "" if trails == 'times01' else "."+trails
    #print(train_name)
    project_name = ".".join([model_json_name,data_name]) if train_name =="" else ".".join([model_name,data_name,train_name])
    project_json_name = ".".join([model_json_name,data_json_name]) if train_json_name =="" else ".".join([model_json_name,data_json_name,train_json_name])
    project_task_name = ".".join([model_json_name,data_name])
    configcombine['project_name']       = project_name
    configcombine['project_task_name']  = project_task_name
    configcombine['project_json_name']  = project_json_name + trails
    configcombine['data'] =data
    configcombine['train']=train
    configcombine['model']=model
    if train_config.hypertuner == "optuna":configcombine['train_mode']="optuna"
    return Config(configcombine)


def read_config(path):
    if '.json' not in path:path = path+'.json'
    with open(path,'r') as f:
        user_dic   = json.load(f)
        if "train" in user_dic:user_dic['train'] = read_train(user_dic['train'])
        if "model" in user_dic:user_dic['model'] = read_model(user_dic['model'])
        if "data"  in user_dic:user_dic['data']  = read_data(user_dic['data'] )
        if "search"  in user_dic:user_dic['search']  = Config(user_dic['search'] )
        if "run"  in user_dic:user_dic['run']  = Config(user_dic['run'] )
        if "scheduler"  in user_dic:user_dic['scheduler']  = read_data(user_dic['scheduler'] )
        return Config(user_dic)
def json_to_config(path):
    with open(path,'r') as f:config_dict=json.load(f)
    return Config(config_dict)
def read_train(user_dic):
    if "scheduler" in user_dic:
        user_dic['scheduler'] = Config(user_dic['scheduler'])
    if "optimizer" in user_dic:
        user_dic['optimizer'] = Config(user_dic['optimizer'])
    if "earlystop" in user_dic:
        user_dic['earlystop'] = Config(user_dic['earlystop'])
    return Config(user_dic)
def read_model(user_dic):

    if 'backbone_config' in user_dic and user_dic['backbone_config'] is not None:
        backbone_config=user_dic['backbone_config']
        #wrapper = user_dic['backbone_TYPE']
        #user_dic['backbone_TYPE']=wrapper(backbone_config)
        if 'backbone_alias' in user_dic and user_dic['backbone_alias'] is not None:
            user_dic['str_backbone_TYPE'] = user_dic['backbone_alias']
        else:
            user_dic['str_backbone_TYPE']=eval(user_dic["backbone_TYPE"]).model_name(**user_dic["backbone_config"])
    else:
        if 'backbone_TYPE'  in user_dic:
            user_dic['str_backbone_TYPE']=user_dic['backbone_TYPE']
    #for key,val in new_pool.items():user_dic[key]=val
    return Config(user_dic)
def read_data(user_dic):
    for key,val in user_dic.items():
        if 'data_curve' in key or 'data_image' in key:
            if isinstance(val,list):
                user_dic[key]=[(os.path.join(DATAROOT,v.rstrip('/')) if DATAROOT not in v else v.rstrip('/')) for v in val ]
            else:
                if val is not None:
                    val = val.replace("/root/autodl-nas/data/","")
                    user_dic[key]=os.path.join(DATAROOT,val.rstrip('/')) if DATAROOT not in val else val.rstrip('/')
                else:
                    user_dic[key] = None

    return Config(user_dic)
