from model import BaseModel,Forward_Model
import sys
#sys.path.append('/home/tianning/Documents/MachineLearning/nas-metasurface/src')
sys.path.append('../nas-metasurface/src')
from nn.micro_decoders import MicroDecoder as Decoder
from nn.encoders import create_encoder
from rl.agent import create_agent, train_agent
import torch
class NASSegmenter(Forward_Model):
    """Create Segmenter"""

    def __init__(self,image_type,curve_type,model_field='real',final_pool=False,first_pool=False,**kargs):
        super(NASSegmenter, self).__init__(image_type,curve_type,**kargs)
        self.encoder = encoder = create_encoder(None)
        agent = create_agent(
            11,100,2,3,4,0.0001,0.95,'ppo',
            len(encoder.out_sizes),
        )
        #decoder_config, entropy, log_prob = agent.controller.sample()
        decoder_config=[[6, [0, 0, 8, 4], [4, 0, 6, 0], [7, 6, 4, 7]], [[3, 1], [3, 0], [3, 1]]]
        entropy = 35.0936
        log_prob=-35.3489
        self.decoder = decoder = Decoder(
            inp_sizes=encoder.out_sizes,
            num_classes=128,
            config=decoder_config,
            agg_size=48,
            aux_cell=True,
            repeats=1,
        )
        self.decoder_config=decoder_config
        self.entropy       =entropy
        self.log_prob      =log_prob
    def forward(self, x,target=None):
        x, aux_outs = self.decoder(self.encoder(x))
        x = torch.nn.Sigmoid()(x)
        x = x.mean((2,3)).unsqueeze(1)
        if target is None:return x
        else:
            loss = self._loss(x,target)
        return loss,x

if __name__ == "__main__":
    model = NASSegmenter(None,None)
