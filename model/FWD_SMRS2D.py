from .FWD_RS2D import ResnetSWrapper,ResnetS
from mltool.ModelArchi.SymmetryCNN import cnn2symmetrycnn
class SYMMETRYResnetSWrapper(ResnetSWrapper):
    def __init__(self,name,model_field,backbone_str,outchannel,tail_type,symmerty_type='P4Z2',post_process_type=None,**kargs):
        super().__init__(name,model_field,backbone_str,outchannel,tail_type,post_process_type=post_process_type,**kargs)
        self.symmerty_type = symmerty_type
    def __call__(self,image_type,curve_type):
        backbone = self.compile_backbone()
        model    = ResnetS(image_type,curve_type,backbone,model_field=self.model_field,outchannel=self.outchannel,**self.kargs)
        _        = self.modify_tail(model)
        _        = self.modify_post_process(model)
        model    = cnn2symmetrycnn(model,self.symmerty_type)
        return model
SYMMETRYRS18KSFNLeakReLUTend= SYMMETRYResnetSWrapper("SYMMETRYRS18KSFNLeakReLUTend",'real' ,'resnet18', 512,'LT:half.LT',
                                                     post_process_type='leakyrelu',symmerty_type='P4Z2')
