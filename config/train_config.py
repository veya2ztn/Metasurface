from .config import*
#<---------------- Base ------------->
Train_Base_Default   =  Config({'epoches': 300, 'batch_size': 1000,'volume':None,
                                'drop_rate':None,'trials':1,'accu_list':None,"show_start_status" : True,
                                "warm_up_epoch"     : 100,"valid_per_epoch"   : 1,
                                "infer_epoch"       : 10,"do_inference"      : False,
                                'doearlystop':True,'doanormaldt':False,
                                'do_extra_phase':True})
Train_Classification =  Train_Base_Default.copy({'accu_list':['ClassifierA', 'ClassifierP','ClassifierN']})

#<----------------  Scheduler ------------->

Scheduler_None = Config({"_TYPE_":None,"config":{}})

Scheduler_Plateau_Default = Config({"_TYPE_":"ReduceLROnPlateau",
                                    "config":{"mode":'min',"factor":0.9,
                                                        "patience":30, "verbose":False}})
Scheduler_CosAWR_Default  = Config({"_TYPE_":"CosineAnnealingWarmRestarts",
                                    "config":{"T_0":10,"eta_min":0,"T_mult":1,
                                                        "verbose":False}})
Scheduler_CosALR_Default  = Config({"_TYPE_":"CosineAnnealingLR",
                                    "config":{"T_max":100}})
Scheduler_Triangle_Scheduler= Config({"_TYPE_":"TriangleScheduler",
                                      "config":{'slope':0.003,
                                                'patience':10,
                                                'max_epoch':20,}
                                    })
Scheduler_TUTCP_Scheduler= Config({"_TYPE_":"LinearUpThenTriPower",
                                    "config":{'cycles': 20,
                                              'slope':0.003,
                                              'patience':5,
                                              'max_epoch':20,
                                              'cycle_decay':0.5,
                                              'trifunc':'sin'}
                                    })


#<----------------  Earlystop ------------->
Earlystop_NMM_Default  =  Config({"_TYPE_":"no_min_more","do_early_stop":True,
                                  "config":{"es_max_window":20,"trace_latest_interval":20,
                                                         "block_best_interval" :100}
                                                        })
#<----------------  Earlystop ------------->
Anormal_D_DC_Default  =  Config({"_TYPE_":"decrease_counting",
                                  "config":{"stop_counting":15,
                                            "wall_value":0.8,
                                            "delta_unit" :1e-4,
                                            }})
#<----------------  Optimizer ------------->
Optimizer_templete  = Config({"_TYPE_":"???","config":{"lr":0.01},"grad_clip":None})
Optimizer_Adam     = Optimizer_templete.copy({'_TYPE_':'Adam'})
Optimizer_SGD      = Optimizer_templete.copy({'_TYPE_':'SGD'})
Optimizer_lbfgs    = Optimizer_templete.copy({'_TYPE_':'LBFGS','config':{'history_size':30}})
Optimizer_lbfgs_60 = Optimizer_templete.copy({'_TYPE_':'LBFGS','config':{'history_size':60}})

Optuna_Train_Default  = Config({'hypertuner':"optuna",'hypertuner_config':{'n_trials':10},
                                 'optimizer_list':{'Adam':{'lr':[0.025,0.1],   'betas':[[0.5,0.99],0.999]},
                                                   #'SGD' :{'lr':[0.01,0.1],'momentum':[0.7,0.99]},
                                                  }
                                 })
Normal_Train_Default  = Config({'hypertuner':"normal"})
