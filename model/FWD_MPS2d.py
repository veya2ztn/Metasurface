from .FWD_RS2D import ResnetSWrapper,ResnetS
from mltool.ModelArchi.TensorNetworkLayer.Conv2dMPS import cnn2mpscnn
class MPSCNNResnetSWrapper(ResnetSWrapper):
    def __call__(self,image_type,curve_type):
        backbone = self.compile_backbone()
        model    = ResnetS(image_type,curve_type,backbone,model_field=self.model_field,outchannel=self.outchannel,**self.kargs)
        _        = self.modify_tail(model)
        _        = self.modify_post_process(model)
        model    = cnn2mpscnn(model,'eCMPS')
        return model
MPSCNNRS18KSFNLeakReLUTend= MPSCNNResnetSWrapper("MPSCNNRS18KSFNLeakReLUTend",'real' ,'resnet18', 512,'LT:half.LT',post_process_type='leakyrelu')
