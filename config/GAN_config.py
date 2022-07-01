from .config import *
#from ..dataset.criterion import *

# loss_machine_pool={'MSE':torch.nn.MSELoss(),
#                    'SL3':SelfEnhanceLoss3(),
#                    'SSIM':SSIMError(),
#                    'MEAN':MeanLoss(),
#                    'Kurt':Kurtosis(),
#                    'BCE':torch.nn.BCELoss(),
#                    'BCENW':torch.nn.BCEWithLogitsLoss(),
#                    'SDFF':VaryingLoss()
#
# }

loss_tasks_G_normal={'name':'plg_curve_1',
              'config':[['labels'   ,'MSE' , 1],
                        ['vectors'  ,'MSE' , 1],
                        ['images'   ,'SSIM', 1],
                        #['images'   ,'SDFF', 1]
                      # ['IQuality' ,'MEAN', 1],
                      # ['IQuality' ,'Kurt', 1],
]}
loss_tasks_G_random={'name':'rdn_curve_1',
              'config':[['labels'   ,'MSE' , 1],
                        ['vectors'  ,'MSE' , 1],
                        ['images'   ,'SSIM', 1],
                        ['images'   ,'SDFF', 1],
                        ['IQuality' ,'MEAN', 1],
                        ['IQuality' ,'Kurt', 1],
]}


the_default_GAN_curve_config   = Config({
    'GAN_stratagy'        : "GAN_CURVE",
    'TRAIN_MODE'          : "new_start",
    'name_rule'           : "test",
    "last_weight_path"    : None,
    'train':Config({'epoches'             : 5000,
                    'batch_size'          : 3000,
                    'doearlystop'         : True,
                    'infer_epoch'         : 100,
                    'do_inference'        : True,
                    'd_lr' 				  : 0.0001,
                	'g_lr' 				  : 0.0005,
                	'c_lr' 				  : 0.0001,
                    'disc_iter'           : 2,
                    'warm_up_epoch'       : 20,
                	'gen_iter'            : 5,
                    'valid_per_epoch'     : 1,
                    'threothod_list'      : [0.5],
                    }),
    'data':Config({'input_image_type'    : "-1to1",
                    'input_image_flip'    : True,
                    'input_image_relax'   : True,
                    'use_soft_label'      : True,
                }),
    'model':Config({
            'FEDMODEL_WEIGHT'     : "checkpoints/PLGG250SMSDatasetB1NES32,curve,simple.on11327/Resnet18KSFNLeakReLUTend/09_04_56-seed-84612/best/epoch615.best_MSError_0.0014",
        	'GAN_PRE_WEIGHT'      : "checkpoints/GAN_PATTERN.PLGG250SMSDatasetB1NES32,curve,simple.on11327/DCGAN_normal/11_06_33-seed-25276/weights/demo_train_at10000",
            'GAN_TYPE'            : "DCGAN_M", #DCGAN,DCGAN_L,DCGAN_M,WGAN_GP,
            'CURVE_REGION'        : "m1to1",
            'one_component'       : False,
            'lattendim'           : 100,
            'loss_tasks_G'        : loss_tasks_G_normal
        }),


})
the_default_GAN_pattern_config = the_default_GAN_curve_config.copy({
    'GAN_stratagy':"GAN_PATTERN",
    'model':{'FEDMODEL_WEIGHT': "checkpoints/PLGG250SMSDatasetB1NES32,curve,simple.on11327/Resnet18KSFNLeakReLUTend/09_04_56-seed-84612/best/epoch615.best_MSError_0.0014",
	          'GAN_PRE_WEIGHT': "checkpoints/GAN_PATTERN.PLGG250SMSDatasetB1NES32,curve,simple.on11327/DCGAN_normal/11_06_33-seed-25276/weights/demo_train_at10000",
              'loss_tasks_G':loss_tasks_G_normal,
            }
})

the_polygon_250_curve_config=the_default_GAN_curve_config
the_Random_curve_config = the_default_GAN_curve_config.copy(
    {'model':{'FEDMODEL_WEIGHT':"checkpoints/SMSDatasetB1NES32,curve,simple.multilength/Resnet18KSFNLeakReLUTend/on108000/10_28_17_33_01-seed-84259/best/epoch112.best_MSError_0.0042",
              'GAN_PRE_WEIGHT':"checkpoints/DCGAN_PATTERN.SMSDatasetB1NES32,curve,simple.multilength/DCGAN_m1to1_norm/11_14_00_55_58/weights/demo_train_at100",
              'loss_tasks_G':loss_tasks_G_random},
        # 'TRAIN_MODE':"continue_train",
        # 'last_weight_path':"checkpoints/DCGAN_CURVE.SMSDatasetB1NES32,curve,simple.multilength/on108000/11_15_21_07_34/routine/11_16_18_44.epoch-2999"
    })
