import time
from config import*
import numpy as np


######################
#### train config ####
######################
#trainbases= [Train_Classification.copy({'do_extra_phase':False,'doearlystop':True})]
trainbases= [Train_Base_Default.copy({'accu_list':[#'MSError',
                                                   #'MSError_for_RDN','MSError_for_PTN','MSError_for_PLG',
                                                   #"ClassifierA","ClassifierP","ClassifierN",
                                                   "BinaryAL","BinaryPL","BinaryNL",
                                                    ],
                                     "grad_clip":None,
                                     'warm_up_epoch':100,
                                     'epoches': 500,
                                     'use_swa':False,
                                     'swa_start':20,
                                     'BATCH_SIZE':6000,
                                     'drop_rate':None,
                                     'do_extra_phase':False,
                                     'doearlystop':True,
                                     'optuna_limit_trials':20})]
hypertuner= [Normal_Train_Default]
schedulers= [Scheduler_None,
             #Scheduler_CosALR_Default
             #Scheduler_Plateau_Default,
             #Scheduler_TUTCP_Scheduler.copy({"config":{'slope':0.0003,'patience':10,'max_epoch':20,'cycle_decay':0.3}})
]
optimizers= [Optimizer_Adam.copy({"config":{"lr":0.0001}})]
earlystops= [Earlystop_NMM_Default.copy({"_TYPE_":"no_min_more",
                                          "config":{"es_max_window":40}})]
anormal_detect= [Anormal_D_DC_Default.copy({"_TYPE_":"decrease_counting",
                                      "config":{"stop_counting":30,
                                                    "wall_value":0.8,
                                                    "delta_unit" :1e-8,
                                                    }})]
train_config_list = [ConfigCombine({"base":[b,h], "scheduler":[s], "earlystop":[e],"optimizer":[o],"anormal_detect":[a]})
                        for b in trainbases for h in hypertuner for s in schedulers
                        for o in optimizers for e in earlystops
                        for a in anormal_detect]

#######################
#### model config #####
#######################
# backbones_list = [#backbone_templete.copy({'backbone_TYPE':'GAEAResultBest_20210512_Complex_8'}),
#                   #backbone_templete.copy({'backbone_TYPE':'SYMMETRYRS18KSFNLeakReLUTend'}),
#                   backbone_templete.copy({'backbone_TYPE':'OgDARTSResult',
#                                          'backbone_config':{"genotype":"/data/Metasurface/checkpoints/search-pcdarts-eedarts-msdataRCurve32/PCDARTS_SYMMETRY_P4Z2/20210803142508-540-branch-1/best/genotype"}})
#                  ]
# criterion_list    = [Config({'criterion_type':"default"})]
# model_config_list = [ConfigCombine({"base":[b,c]}) for b in backbones_list for c in criterion_list ]

dm_config_list = [
[msdataTCurve32PLG250,  backbone_templete.copy({'criterion_type':"default",'backbone_TYPE':'Resnet18KSFNLeakReLUTend','backbone_alias':'Resnet18KSFNLeakReLUTend',
})],
# [msdataTCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG_P4Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG_P4Z2_NE/best/genotype"}})],
# [msdataTCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG_Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG_Z2_NE/best/genotype"}})],
# [msdataTCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG_NORMAL_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG_NORMAL_NE/best/genotype"}})],
# [msdataRCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG_P4Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG_Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG_Z2_NE/best/genotype"}})],
# [msdataRCurve32PLG,    backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG_NORMAL_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG_NORMAL_NE/best/genotype"}})],
# [msdataTCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32RDN_P4Z2_NE_ccl'  ,'backbone_config':{'padding_mode':'circular',"genotype":"nas_result_fast_link/T32RDN_P4Z2_NE/best/genotype"}})],
# [msdataTCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32RDN_Z2_NE_ccl'    ,'backbone_config':{'padding_mode':'circular',"genotype":"nas_result_fast_link/T32RDN_Z2_NE/best/genotype"}})],
# [msdataTCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32RDN_NORMAL_NE_ccl','backbone_config':{'padding_mode':'circular',"genotype":"nas_result_fast_link/T32RDN_NORMAL_NE/best/genotype"}})],

#[msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_ccl_TNend3'  ,'backbone_config':{"tnend":True,"virtual_bond_dim":3,'padding_mode':'circular',"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_ccl_TNend4'  ,'backbone_config':{"tnend":True,"virtual_bond_dim":4,'padding_mode':'circular',"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_ccl_TNend5'  ,'backbone_config':{"tnend":True,"virtual_bond_dim":5,'padding_mode':'circular',"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_TNend3'      ,'backbone_config':{"tnend":True,"virtual_bond_dim":3,"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_TNend4'      ,'backbone_config':{"tnend":True,"virtual_bond_dim":4,"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_P4Z2_NE_TNend5'      ,'backbone_config':{"tnend":True,"virtual_bond_dim":5,"genotype":"nas_result_fast_link/R32RDN_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_Z2_NE_ccl'    ,'backbone_config':{'padding_mode':'circular',"genotype":"nas_result_fast_link/R32RDN_Z2_NE/best/genotype"}})],
# [msdataRCurve32,       backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32RDN_NORMAL_NE_ccl','backbone_config':{'padding_mode':'circular',"genotype":"nas_result_fast_link/R32RDN_NORMAL_NE/best/genotype"}})],
# [msdataRCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG250_P4Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG250_P4Z2_NE/best/genotype"}})],
# [msdataRCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG250_Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG250_Z2_NE/best/genotype"}})],
# [msdataRCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_R32PLG250_NORMAL_NE','backbone_config':{"genotype":"nas_result_fast_link/R32PLG250_NORMAL_NE/best/genotype"}})],
# [msdataTCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG250_P4Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG250_P4Z2_NE/best/genotype"}})],
# [msdataTCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG250_Z2_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG250_Z2_NE/best/genotype"}})],
# [msdataTCurve32PLG250, backbone_OgDARTSResult.copy({'backbone_alias':'DARTSearch_T32PLG250_NORMAL_NE','backbone_config':{"genotype":"nas_result_fast_link/T32PLG250_NORMAL_NE/best/genotype"}})],
# [msdataT_PLG250, backbone_OgDARTSResult.copy({'criterion_type':"CELoss",'backbone_alias':'DARTSearch_Bin_RDN_NORMAL_NE_2','backbone_config':{"genotype":"nas_result_fast_link/Bin_RDN_NORMAL_NE/best/genotype"}})],
# [msdataT_PTN   , backbone_OgDARTSResult.copy({'criterion_type':"CELoss",'backbone_alias':'DARTSearch_Bin_PTN_NORMAL_NE'   ,'backbone_config':{"genotype":"nas_result_fast_link/Bin_PTN_NORMAL_NE/best/genotype"}})],
# [msdataT_PLG250, backbone_OgDARTSResult.copy({'criterion_type':"CELoss",'backbone_alias':'DARTSearch_Bin_PLG250_NOZERO_NE_2','backbone_config':{"genotype":"nas_result_fast_link/Bin_PLG250_NOZERO_NE_2/best/genotype"}})],
# [msdataT_PLR250, backbone_OgDARTSResult.copy({'criterion_type':"CELoss",'backbone_alias':'DARTSearch_Bin_PLR250_NORMAL_NE_2','backbone_config':{"genotype":"nas_result_fast_link/Bin_PLR250_NORMAL_NE_2/best/genotype"}})],

]
# dm_config_list = [[dataset_path_16x16_list_RDN.copy({'dataset_TYPE':'MNISTdataset','dataset_args':{}}),backbone_Resnet18S.copy({'criterion_type':"CELoss"})]]

#### generate config
for train_cfg in train_config_list:
    for data_cfg,model_cfg in dm_config_list:

            cfg = Merge(data_cfg, train_cfg, model_cfg)
            TIME_NOW        = time.strftime("%m_%d_%H_%M_%S")
            cfg.create_time = TIME_NOW
            del data_cfg
            cfg.save()
