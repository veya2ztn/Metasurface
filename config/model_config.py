from .config import *

# ----------------------- TRANSFORMS ----------------------- #

transform_base = Config({
    'transform_TYPE':'CurveSample',
    'transform_config':{'method':'unisample'}
})
transform_wavelet=transform_base.copy({
    'transform_TYPE':'CurveWavelet',
    'transform_config':{'method':'dwt','level':6,'out_num':4}
})
transform_wavelet_one=transform_base.copy({
    'transform_TYPE':'CurveWavelet',
    'transform_config':{'method':'dwt','level':6,'out_num':1}
})
transform_sample=transform_base.copy({
    'transform_TYPE':'CurveSample',
    'transform_config':{'method':'unisample','sample_num':128}
})
transform_fourier_fft=transform_base.copy({
    'transform_TYPE':'CurveFourier',
    'transform_config':{'method':'fft'}
})
transform_fourier_dct=transform_base.copy({
    'transform_TYPE':'CurveFourier',
    'transform_config':{'method':'dct'}
})
transform_fourier_rfft=transform_base.copy({
    'transform_TYPE':'CurveFourier',
    'transform_config':{'method':'rfft'}
})


# ----------------------- BACKBONES ----------------------- #

backbone_templete        = Config({'backbone_TYPE':'???','train_batches':'auto','memory_para':[4.2,891],'curve_field':'real','image_field':'real'})
criterion_default        = Config({'criterion_type':'default'})

backbone_Resnet18S       = Config({'backbone_TYPE':'Resnet18S'       ,'train_batches':4000,'memory_para':[3,972]})
backbone_Resnet34S       = Config({'backbone_TYPE':'Resnet34S'       ,'train_batches':4000})
backbone_Resnet50S       = Config({'backbone_TYPE':'Resnet50S'       ,'train_batches':1000})
backbone_Resnet101S      = Config({'backbone_TYPE':'Resnet101S'      ,'train_batches': 800,'memory_para':[12,1280]})
backbone_Densenet121S    = Config({'backbone_TYPE':'Densenet121S'    ,'train_batches': 800,'memory_para':[ 12,828]})
backbone_Densenet169S    = Config({'backbone_TYPE':'Densenet169S'    ,'train_batches': 300})
backbone_Densenet201S    = Config({'backbone_TYPE':'Densenet201S'    ,'train_batches': 300})
backbone_SqueezeNet1S    = Config({'backbone_TYPE':'SqueezeNet1S'    ,'train_batches':3000,'memory_para':[3,802]})
backbone_Resnet18CDUPL   = Config({'backbone_TYPE':'Resnet18CDUPL'   ,'train_batches':2000,
                                   'curve_field':'complex','image_field':'complex',
                                   'memory_para':[7,1107]})
backbone_Resnet101CDUPL  = Config({'backbone_TYPE':'Resnet101CDUPL'  ,'train_batches':500,
                                   'curve_field':'complex','image_field':'complex',
                                   'memory_para':[45,1585]})
backbone_SqueezeNet1CDUPL= Config({'backbone_TYPE':'SqueezeNet1CDUPL','train_batches':1500,
                                   'curve_field':'complex','image_field':'complex',
                                   'memory_para':[47,1296]})
backbone_OgDARTSResult = backbone_templete.copy({'backbone_TYPE':'OgDARTSResult','criterion_type':'default',
                                                 'backbone_config':{"genotype":'???'}})
