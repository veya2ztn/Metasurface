from .config import*

# ----------------------- DATASETS ----------------------- #
dataset_path_16x16_list_RDN = Config({
    'train_data_curve': "RDNDATASET/train_data_list",
    'train_data_image': None,
    'valid_data_curve': "RDNDATASET/valid_data_list",
    'valid_data_image': None,
    'demo__data_curve': "16x16/demo.curve.npy",
    'demo__data_image': "16x16/demo.image.npy",
    'origin_data_leng': 1001,
    'global_dataset_type':'RDN',
})
dataset_path_16x16_list_PTN = Config({
    'train_data_curve':"PTNDATASET/train_data_list",
    'train_data_image':None,
    'valid_data_curve':"PTNDATASET/valid_data_list",
    'valid_data_image':None,
    'global_dataset_type':'PTN',
})
dataset_path_16x16_list_PLG = Config({
    'train_data_curve':"PLGDATASET/train_data_list",
    'train_data_image':None,
    'valid_data_curve':"PLGDATASET/valid_data_list",
    'valid_data_image':None,
    'global_dataset_type':'PLG',
})
dataset_path_16x16_list_PLG250 = dataset_path_16x16_list_PLG.copy({
                                    'range_clip':[1001-250,1001],
                                    'global_dataset_type':'PLG250',
                                    })
dataset_path_16x16_list_PLR = Config({
    'train_data_curve':"PLRDATASET/train_data_list",
    'train_data_image':None,
    'valid_data_curve':"PLRDATASET/valid_data_list",
    'valid_data_image':None,
    'global_dataset_type':'PLR',
})
dataset_path_16x16_list_PLR250= dataset_path_16x16_list_PLR.copy({
                                    'range_clip':[1001-250,1001],
                                    'global_dataset_type':'PLR250',
                                    })

dataset_path_16x16_list_RDNpPTNpPLG_27000_together = Config({
    'train_data_curve':"ThreeDATASET_RDN+PDN+RDN/train_data_list",
    'train_data_image':None,
    'valid_data_curve':"ThreeDATASET_RDN+PDN+RDN/valid_data_list",
    'valid_data_image':None,
    'demo__data_curve':"16x16/demo.curve.npy",
    'demo__data_image':"16x16/demo.image.npy",
    'origin_data_leng':1001,
    'global_dataset_type':'ThreeDataset',

})
dataset_path_16x16_list_RDNpPTNpPLG_27000 = Config({
    'train_data_curve':[f"{v}/train_data_list" for v in ["randomly_data_27000","PTN_data_all","polygon_data_all"]],
    'train_data_image':None,
    'valid_data_curve':[f"{v}/valid_data_list" for v in ["randomly_data_27000","PTN_data_all","polygon_data_all"]],
    'valid_data_image':None,
    'dataset_args':{
        'DATAROOT':f"{DATAROOT}/ThreeDATASET_RDN+PDN+RDN",
        'partIdx':{'train':{'RDN':(0,27000),'PTN':(27000,27000*2),'PLG':(27000*2,27000*3)},
                   'test' :{'RDN':(0,3000), 'PTN':(3000,3000*2),  'PLG':(3000*2,3000*3)}
                   }},
    'demo__data_curve':"16x16/demo.curve.npy",
    'demo__data_image':"16x16/demo.image.npy",
    'origin_data_leng':1001,
    'global_dataset_type':'ThreeDataset',

})


normal_ms_balance_2_classification= Config({'dataset_TYPE':'SMSDatasetB1NES128','dataset_norm':'none',
                                            'dataset_args':{'type_predicted':'onehot','target_predicted':'balance_leftorright'}})

graph_ms_balance_2_classification = normal_ms_balance_2_classification.copy({'dataset_TYPE':'SMSDatasetGraph_Test'})


NCP_DATASET_128 = Config({'dataset_TYPE':'SMSDatasetN','dataset_norm':'mean2zero',
                          'dataset_args':{'type_predicted':'curve','target_predicted':'simple',
                                          'curve_branch':'T','FeatureNum':128,'enhance_p':'E'}
                           })

NCPT_DATASET_32 = Config({'dataset_TYPE':'SMSDatasetN','dataset_norm':'mean2zero',
                          'dataset_args':{'type_predicted':'curve','target_predicted':'simple',
                                          'curve_branch':'T','FeatureNum':32,'enhance_p':'E','range_clip':None}
                           })
CCPT_DATASET_32 = Config({'dataset_TYPE':'SMSDatasetC','dataset_norm':'mean2zero',
                          'dataset_args':{'type_predicted':'curve','target_predicted':'simple',
                                          'curve_branch':'T','FeatureNum':32,'enhance_p':'E','range_clip':None}
                           })
#NCPT_DATASET_32_250 = NCPT_DATASET_32.update({'dataset_args':{'range_clip':[751,1001]}})

NCPR_DATASET_32 = Config({'dataset_TYPE':'SMSDatasetN','dataset_norm':'mean2zero',
                          'dataset_args':{'type_predicted':'curve','target_predicted':'simple',
                                          'curve_branch':'R','FeatureNum':32,'enhance_p':'E','range_clip':None}
                           })
#NCPR_DATASET_32_250 = NCPR_DATASET_32.update({'dataset_args':{'range_clip':[751,1001]}})

msdataRCurve32       = ConfigCombine({"base":[dataset_path_16x16_list_RDN   , NCPR_DATASET_32]})
msdataTCurve32       = ConfigCombine({"base":[dataset_path_16x16_list_RDN   , NCPT_DATASET_32]})
msdataRCurve32PLG    = ConfigCombine({"base":[dataset_path_16x16_list_PLG   , NCPR_DATASET_32]})
msdataTCurve32PLG    = ConfigCombine({"base":[dataset_path_16x16_list_PLG   , NCPT_DATASET_32]})
msdataTCurve32RDNCPLX= ConfigCombine({"base":[dataset_path_16x16_list_RDN   , CCPT_DATASET_32]})


msdataRCurve32PLG250 = ConfigCombine({"base":[dataset_path_16x16_list_PLG250, NCPR_DATASET_32]})
msdataTCurve32PLG250 = ConfigCombine({"base":[dataset_path_16x16_list_PLG250, NCPT_DATASET_32]})

msdataT_PLG250 = ConfigCombine({"base":[dataset_path_16x16_list_PLG250, normal_ms_balance_2_classification]})
msdataT_PLR250 = ConfigCombine({"base":[dataset_path_16x16_list_PLR250, normal_ms_balance_2_classification]})
msdataT_PLR    = ConfigCombine({"base":[dataset_path_16x16_list_PLR   , normal_ms_balance_2_classification]})
msdataT_PTN    = ConfigCombine({"base":[dataset_path_16x16_list_PTN   , normal_ms_balance_2_classification]})
msdataT_RDN    = ConfigCombine({"base":[dataset_path_16x16_list_RDN   , normal_ms_balance_2_classification]})

PLP_DATASET_128 = Config({'dataset_TYPE':'SMSDatasetN','dataset_norm':'none',
                          'dataset_args':{'type_predicted':'multihot',
                          'target_predicted':'location_of_peaks',
                          'curve_branch':'T','FeatureNum':128,'enhance_p':'E'}})

#TODO:PDP_DATASET_128
