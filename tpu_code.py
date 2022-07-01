# VERSION = "20200220"  #@param ["20200220","nightly", "xrt==1.15.0"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION
# !pip install tensorwatch
# cd "/content/drive/My Drive/metasurface"
# import sys
# sys.path.insert(0, '/content/drive/My Drive/metasurface')

from mltool.MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores
from mltool.MLlog import ModelSaver,AverageMeter,RecordLoss,Curve_data,IdentyMeter,LossStores
from mltool.fastprogress import master_bar, progress_bar
from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.optim.radam import RAdam
from mltool.visualization import *
import os
import torch
import numpy as np
from torch.autograd import Variable
#from dataset import FourierDirect
import json
#from model import get_resnet_101
from utils import show_a_demo,record_data
from Curve2vector import CurveFourier,CurveSample,CurveWavelet
from dataset import *
import numpy as np
import os
import time


from config import DataType
from model.FWD_RS2D import ResnetNature
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

import numpy as np
import os
import time

if __name__ == '__main__':
    CURVETRAIN="./data/train.curve.npy"
    IMAGETRAIN="./data/train.image.npy"
    CURVE_TEST="./data/test.curve.npy"
    IMAGE_TEST="./data/test.image.npy"
    CURVE_DEMO="./data/demo.curve.npy"
    IMAGE_DEMO="./data/demo.image.npy"

    # Define Parameters
    FLAGS = {}
    FLAGS['datadir'] = "/tmp/mnist"
    FLAGS['batch_size'] = 700
    FLAGS['num_workers'] = 4
    FLAGS['learning_rate'] = 0.01
    FLAGS['momentum'] = 0.5
    FLAGS['num_epochs'] = 10
    FLAGS['num_cores'] = 8
    FLAGS['log_steps'] = 3
    FLAGS['metrics_debug'] = False

    transformer= CurveSample(method='unisample',feature='norm',sample_num=128)
    train_dataset = MetaSurfaceSetLight(CURVETRAIN,IMAGETRAIN,transformer,vec_dim=50,case_type='train',normf='none',verbose=False)
    test_dataset  = MetaSurfaceSetLight(CURVE_TEST,IMAGE_TEST,transformer,vec_dim=50,case_type='test',normf='none',verbose=False)

    def train_mnist():
      torch.manual_seed(1)

      # Get and shard dataset into dataloaders


      train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
      train_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=FLAGS['batch_size'],
          sampler=train_sampler,
          num_workers=FLAGS['num_workers'],
          drop_last=True)
      test_loader = torch.utils.data.DataLoader(
          test_dataset,
          batch_size=FLAGS['batch_size'],
          shuffle=False,
          num_workers=FLAGS['num_workers'],
          drop_last=True)

      # Scale learning rate to world size
      lr = FLAGS['learning_rate'] * xm.xrt_world_size()


      image_type=DataType('real',(1,16,16))
      curve_type=DataType('real',(1,128))
      model_type='real'

      # Get loss function, optimizer, and model
      device = xm.xla_device()
      model  = ResnetNature(image_type,curve_type).to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      #loss_fn = nn.NLLLoss()

      def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (c_train, i_train) in enumerate(loader):

          optimizer.zero_grad()
          loss,output = model(i_train,c_train)

          loss.backward()
          xm.optimizer_step(optimizer)

          tracker.add(FLAGS['batch_size'])
          if x % FLAGS['log_steps'] == 0:
            print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                xm.get_ordinal(), x, loss.item(), tracker.rate(),
                tracker.global_rate(), time.asctime()), flush=True)

      def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        accuracy=[]
        for c_train, i_train,real in loader:
          output = model(i_train)
          real_p = transformer.vector2curve(output)
          data   = i_train
          pred   = real_p
          target = c_train
          #accu_  = transformer.curve_loss(output,c_train).mean()
          accu_  = transformer.curve_loss(real_p,real).mean()
          accuracy.append(accu_)
        accuracy = np.mean(accuracy)
        print('[xla:{}] Accuracy={:.2f}%'.format(
            xm.get_ordinal(), accuracy), flush=True)
        return accuracy, data, pred, target


      #print("start train")
      # Train and eval loops
      accuracy = 0.0
      data, pred, target = None, None, None
      for epoch in range(1, FLAGS['num_epochs'] + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        accuracy, data, pred, target  = test_loop_fn(para_loader.per_device_loader(device))
        if FLAGS['metrics_debug']:
          xm.master_print(met.metrics_report(), flush=True)

      return accuracy, data, pred, target

    # Start training processes
    def _mp_fn(rank, flags):
        global FLAGS
        FLAGS = flags
        torch.set_default_tensor_type('torch.FloatTensor')
        accuracy, data, pred, target = train_mnist()
        # if rank == 0:
        #   # Retrieve tensors that are on TPU core 0 and plot.
        #   plot_results(data.cpu(), pred.cpu(), target.cpu())

    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'],start_method='fork')
