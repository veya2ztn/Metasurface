from mltool.ModelArchi.ModelSearch.DARTS_Model import Network as MSDARTSNetwork
from mltool.ModelArchi.ModelSearch.DARTS_Model_Origin import Network as OgDARTSNetwork
from .model import BaseModel,Forward_Model
import torch
import os
from mltool.ModelArchi.ModelSearch.genotype import Genotype
import hashlib
class DARTSResultWrapper:
    def __init__(self,structure_file):
        self.structure_file = structure_file
    def __call__(self,image_type,curve_type,model_field='real',**kargs):
        return DARTSResult(image_type,curve_type,self.structure_file,**kargs)
    def __name__(self):
        return f"DARTS_at_{os.path.basename(self.structure_file)}"

class DARTSResult(Forward_Model):
    def __init__(self,image_type,curve_type,structure_file,structure_weight=None,model_field='real',**kargs):
        super().__init__(image_type,curve_type,**kargs)
        structure_config_dict = torch.load(structure_file)
        init_channel    = structure_config_dict['_C']
        classes_num     = structure_config_dict['_num_classes']
        model_layernum  = structure_config_dict['_layers']
        operation_config = structure_config_dict['config']
        if structure_weight is None:
            self.backbone=MSDARTSNetwork(init_channel, classes_num, model_layernum,circularQ=False,operation_config=operation_config)
        else:
            operation_weight =  torch.load(structure_weight)
            self.backbone=MSDARTSNetwork(init_channel, classes_num, model_layernum,circularQ=False,
                        operation_config=structure_config,
                        operation_weight=operation_weight)
    def forward(self, x,target=None):
        x=self.backbone(x)  ;#print(x.shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

DARTSResultBest_20210301_noZero  = DARTSResultWrapper("model/DARTSmodelConfig/best_structure_20210301_noZero.config.pt")
DARTSResultBest_20210301_useZero = DARTSResultWrapper("model/DARTSmodelConfig/best_structure_20210301_useZero.config.pt")


def get_genotype(genotype):
    if isinstance(genotype, str):
        if '/' in genotype:
            with open(genotype,'r') as f:
                genotype = f.read()
        genotype = eval(genotype)
    return genotype


class OgDARTSResult(Forward_Model):
    def __init__(self,image_type,curve_type,genotype,model_field='real',
                      init_channel     = None,classes_num      = None,node             = None,
                      layers           = None,auxiliary        = False,padding_mode='zeros',
                      **kargs):
        super().__init__(image_type,curve_type,**kargs)
        genotype         = get_genotype(genotype)
        classes_num      = curve_type.shape[-1]
        self.backbone    = OgDARTSNetwork(init_channel, classes_num, node, layers,genotype=genotype,auxiliary=auxiliary,padding_mode=padding_mode,**kargs)

    def forward(self, x,target=None):
        x=self.backbone(x).unsqueeze(1)  ;#print(x.shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

    @staticmethod
    def model_name(genotype=None):
        genotype = get_genotype(genotype)
        genotype = genotype.__str__()
        return f"DARTSearch_{hashlib.md5(genotype.encode(encoding='UTF-8')).hexdigest()}"
    def set_drop_prob(self,drop_path_prob):
        self.backbone.drop_path_prob=drop_path_prob
class OgDARTSResultWrapper:
    def __init__(self,genotype=None,**kargs):

        self.genotype_name = genotype
        self.genotype      = eval(genotype)
        self.genokargs     = kargs
    def __call__(self,image_type,curve_type,model_field='real',**kargs):
        return OgDARTSResult(image_type,curve_type,self.genotype,**self.genokargs,**kargs)
    def __name__(self):
        name = f"OgDARTS_for_{self.genotype}"
        return name
PC_DARTS_metas = Genotype(
    normal=[('sep_conv_3x3', 1),('max_pool_3x3', 0),
            ('sep_conv_3x3', 2),('max_pool_3x3', 0),
            ('skip_connect', 2),('avg_pool_3x3', 3),
            ('avg_pool_3x3', 2),('dil_conv_3x3', 3)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1),('dil_conv_5x5', 0),
            ('dil_conv_3x3', 2),('max_pool_3x3', 1),
            ('skip_connect', 3),('dil_conv_3x3', 2),
            ('skip_connect', 4),('max_pool_3x3', 0)],
            reduce_concat=range(2, 6)
    )
PC_DARTS_metas_d =Genotype(normal=[('sep_conv_3x3', 1), ('deleted', 0),
                                   ('skip_connect', 2), ('deleted', 0),
                                   ('sep_conv_3x3', 2), ('dil_conv_5x5', 0),
                                   ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)],
                           normal_concat=range(2, 6),
                           reduce=[('avg_pool_3x3', 1), ('deleted', 0),
                                   ('skip_connect', 2), ('max_pool_3x3', 1),
                                   ('dil_conv_3x3', 2), ('skip_connect', 3),
                                   ('skip_connect', 4), ('deleted', 3)],
                           reduce_concat=range(2, 6))
GAEAResultBest_20210401_d   = OgDARTSResultWrapper("PC_DARTS_metas_d",init_channel     = 16)
from mltool.ModelArchi.SymmetryCNN import cnn2symmetrycnn
Z2_GAEAResultBest_20210401_d   = lambda *arg,**kargs: cnn2symmetrycnn(GAEAResultBest_20210401_d(*arg,**kargs),type='Z2')
P4Z2_GAEAResultBest_20210401_d = lambda *arg,**kargs: cnn2symmetrycnn(GAEAResultBest_20210401_d(*arg,**kargs),type='P4Z2')

PC_DARTS_complex_1=Genotype(
    normal=[('[cplx]sep_conv_3x3', 0), ('[cplx]sep_conv_3x3', 1),
            ('[cplx]sep_conv_5x5', 2), ('[cplx]sep_conv_3x3', 0),
            ('[cplx]sep_conv_3x3', 3), ('avg_pool_3x3', 0),
            ('[cplx]sep_conv_5x5', 4), ('[cplx]sep_conv_3x3', 0)],
            normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1),        ('[cplx]dil_conv_5x5', 0),
             ('[cplx]sep_conv_5x5', 1), ('[cplx]sep_conv_5x5', 2),
             ('skip_connect', 2),       ('[cplx]dil_conv_3x3', 3),
             ('avg_pool_3x3', 1),       ('[cplx]sep_conv_7x7', 3)],
             reduce_concat=range(2, 6))
GAEAResultBest_20210511_Complex_16 = OgDARTSResultWrapper("PC_DARTS_complex_1",init_channel     = 16)
GAEAResultBest_20210512_Complex_8  = OgDARTSResultWrapper("PC_DARTS_complex_1",init_channel     = 8 )
#### old not good result part.
class DARTSResult1(Forward_Model):# this model 不是最好的config
    def __init__(self,image_type,curve_type,model_field='real',**kargs):
        super().__init__(image_type,curve_type,**kargs)
        structure_file = "model/DARTSmodelConfig/model001.pickle"
        with open(structure_file, 'rb') as f:structure_config=pickle.load(f)
        self.backbone=MSDARTSNetwork(16, 2, 8,circularQ=False,opertion_config=structure_config)
    def forward(self, x,target=None):
        x=self.backbone(x)  ;#print(x.shape)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x
