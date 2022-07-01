# from config import *
#
# nowargs = the_default_GAN_pattern_config
# nowargs = nowargs.copy({
#     'PROJECTROOT':"checkpoints/PTNSMSDatasetB1NES32,curve,simple.on27000/Resnet18KSFNTTend/12_08_57-seed-18124/"
# })
# nowargs.save('projects/undo/GANTASK1.json')

from config import *
PLG_FWDMODEL_WEIGHT= "checkpoints/PLG250SMSDatasetB1NES32,curve,simple.on27000/Resnet18KSFNLeakReLUTend/02_05_26-seed-7846/best/epoch648.best_MSError_0.0004"
PLG_GAN_PRE_WEIGHT = "checkpoints/GAN_PATTERN.PLG250SMSDatasetB1NES32,curve,simple.on27000/DCGAN_m1to1_norm/06_13_08-seed-43654/weights/demo_train_at10000"

#### do pattern curve train
nowargs = the_default_GAN_curve_config
RDN_FWDMODEL_WEIGHT= "checkpoints/[old-valid-size]SMSDatasetB1NES32,curve,simple.on97000/ResnetNature_a4/22_46_03-seed-84771/best/best_MAError_0.0358"
RDN_GAN_PRE_WEIGHT = "checkpoints/GAN_PATTERN.SMSDatasetB1NES32,curve,simple.on97000/DCGAN_m1to1_norm/09_08_16-seed-65939/weights/demo_train_at10000"
nowargs = nowargs.copy({
	'FEDMODEL_WEIGHT'     : RDN_FWDMODEL_WEIGHT,
	'GAN_PRE_WEIGHT'      : PLG_GAN_PRE_WEIGHT,
	'batch_size'          : 200,
    'train_with_one_compoent_constrain':True
})
nowargs.save('projects/undo/GANCURVETASK2.json')


# PTN_FWDMODEL_WEIGHT= "checkpoints/PTNSMSDatasetB1NES32,curve,simple.on27000/Resnet18KSFNLeakReLUTend/12_13_07-seed-50883/best/epoch168.best_MSError_0.0044"
# PTN_GAN_PRE_WEIGHT = "checkpoints/DCGAN_PATTERN.PTNSMSDatasetB1NES32,curve,simple.on27000/DCGAN_m1to1_norm/10_13_12_49_01/weights/demo_train_at9500"
# FWDMODEL_WEIGHT = PLG_FWDMODEL_WEIGHT
# GAN_PRE_WEIGHT  = PLG_GAN_PRE_WEIGHT
# nowargs = nowargs.copy({
# 	'FEDMODEL_WEIGHT'     : FWDMODEL_WEIGHT,
# 	'GAN_PRE_WEIGHT'      : GAN_PRE_WEIGHT,
# 	'batch_size'          : 2000,
# 	'doearlystop'         : True,
# })
#
# nowargs = nowargs.copy({
# 	'd_lr' 				  : 0.0001,
# 	'g_lr' 				  : 0.0005,
# 	'c_lr' 				  : 0.0001,
# 	'balance_coef'        : [0.5,1,0.05],
# })
#
# nowargs_list = [nowargs.copy({
# 							'd_lr' 				  : 0.0001,
# 							'g_lr' 				  : 0.0005,
# 							'c_lr' 				  : 0.0001,
# 							'balance_coef'        : [0.5,1,0.05],
# 				}) for d_lr in [0.00005,0.0001,0.00015]
#    				   for g_cof in [1,10]
# 				]
# for i,args in enumerate(nowargs_list):
# 	args.save(f'projects/undo/GANCURVETASK{i}.json')
