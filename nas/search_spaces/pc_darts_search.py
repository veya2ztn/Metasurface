import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



from .model_search_base import SuperNetwork
from . import genotypes as  gtype

from mltool.ModelArchi.ModelSearch.operations_define import *



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def node_map(num):
    if num == 0: return "a"
    if num == 1: return "b"
    return num-2

CNNModulelist={
"Z2": Z2_Conv2d,
"P4Z2":P4Z2_Conv2d,
"P4": P4_Conv2d,
}
class MixedOp(nn.Module):
    def __init__(self, C, stride, op_names):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in op_names:
            op = OPS[primitive](C // 4, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // 4, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, : dim_2 // 4, :, :]
        xtemp2 = x[:, dim_2 // 4 :, :, :]
        try:
            temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        except:
            [print(f"shape1:{w.shape} -->  shape2:{xtemp.shape} -->{op.__class__} -->shape3:{op(xtemp).shape}") for w, op in zip(weights, self._ops)]
            raise
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, 4)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans


class Cell(nn.Module):
    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        op_names,
        abs_symmetry=False,
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if not abs_symmetry:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False) if reduction_prev else ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        else:
            stride = 2 if reduction_prev else 1
            CNNModule = CNNModulelist[abs_symmetry]
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, stride, 0, CNNModule=CNNModule, affine=False,active_symmetry_fix=True)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, op_names)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                weights2[offset + j] * self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class PCDARTSNetwork(SuperNetwork):
    def __init__(
        self,
        C,
        num_classes,
        nodes,
        layers,
        criterion=None,
        search_space_name=None,
        op_names="PCDARTS",
        multiplier=4,
        stem_multiplier=3,
        predict_type=None,abs_symmetry=False,
        **kwargs
    ):
        assert search_space_name == "pcdarts"
        assert criterion is not None
        super(PCDARTSNetwork, self).__init__(C, num_classes, nodes, layers, criterion)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = nodes
        self._multiplier = multiplier
        self.search_space = search_space_name
        self.predict_type = predict_type
        self.stem_multiplier=stem_multiplier
        # These variables required by architect
        print(f"use operation class {op_names}")
        if isinstance(op_names,str):
            if "+" in op_names:
                op_names=op_names.split('+')
            else:
                op_names=[op_names]
            ops_list = []
            for op_name in op_names:ops_list+=list(eval(f"gtype.{op_name}"))
        elif isinstance(op_names,list):
            ops_list = op_names
        else:
            raise NotImplementedError
        self.op_names = ops_list
        print("will search arch from below operations")
        _=[print(op) for op in self.op_names]


        self._num_ops = len(self.op_names)
        self.search_reduce_cell = True

        self.n_inputs = 2
        self.add_output_node = False

        C_curr = stem_multiplier * C

        if "[auto][symP4]sep_conv_3x3" in self.op_names:abs_symmetry="P4"
        if "[auto][symZ2]sep_conv_3x3" in self.op_names:abs_symmetry="Z2"
        if "[auto][symP4Z2]sep_conv_3x3" in self.op_names:abs_symmetry="P4Z2"

        if not abs_symmetry:
            self.stem = nn.Sequential(
                nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
        else:
            CNNModule = CNNModulelist[abs_symmetry]
            self.stem = nn.Sequential(
                CNNModule(1, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
        reduction_layer_idx = [layers // 3, 2 * layers // 3]
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in reduction_layer_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                nodes,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                self.op_names,abs_symmetry=abs_symmetry,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if (predict_type is None) or ("onehot" in predict_type) :
            self.classifier     = nn.Linear(C_prev, num_classes)
        elif predict_type == 'curve+mean2zero':
            assert type(self._criterion) is not torch.nn.CrossEntropyLoss
            #self.global_pooling = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(C_prev, num_classes),nn.Tanh())
        elif predict_type == 'curve+none':
            assert type(self._criterion) is not torch.nn.CrossEntropyLoss
            #self.global_pooling = nn.Identity()
            self.classifier = nn.Sequential(nn.Linear(C_prev, num_classes),nn.Sigmoid())
        else:
            print(f"the predict_type={predict_type} is not set")
            raise NotImplementedError
        # Store init parameters for norm computation.
        self.store_init_weights()

    def new(self):
        model_new = PCDARTSNetwork(
            self._C,
            self._num_classes,
            self._steps,
            self._layers,
            criterion=self._criterion,
            search_space_name=self.search_space,
            op_names=self.op_names,
            multiplier=self._multiplier,
            stem_multiplier=self.stem_multiplier
        ).cuda()
        return model_new

    def forward(self, input, discrete=False):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas["reduce"]
                n = 3
                start = 2
                weights2 = self.edges["reduce"][0:2]
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = self.edges["reduce"][start:end]
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = self.alphas["normal"]
                n = 3
                start = 2
                weights2 = self.edges["normal"][0:2]
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = self.edges["normal"][start:end]
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
            #print(s1.shape)
        out = self.global_pooling(s1)
        #print(out.shape)
        logits = self.classifier(out.view(out.size(0), -1))
        if 'curve' in self.predict_type:
            logits=logits.unsqueeze(1)
        return logits, None

    def _loss(self, input, target):
        logits, _ = self(input)
        return self._criterion(logits, target)



    # def genotype(self):

    #  def _parse(weights,weights2):
    #    gene = []
    #    n = 2
    #    start = 0
    #    for i in range(self._steps):
    #      end = start + n
    #      W = weights[start:end].copy()
    #      W2 = weights2[start:end].copy()
    #      for j in range(n):
    #        W[j,:]=W[j,:]*W2[j]
    #      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
    #
    #      #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
    #      for j in edges:
    #        k_best = None
    #        for k in range(len(W[j])):
    #          if k != PRIMITIVES.index('none'):
    #            if k_best is None or W[j][k] > W[j][k_best]:
    #              k_best = k
    #        gene.append((PRIMITIVES[k_best], j))
    #      start = end
    #      n += 1
    #    return gene
    #  n = 3
    #  start = 2
    #  weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    #  weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    #  for i in range(self._steps-1):
    #    end = start + n
    #    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
    #    tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
    #    start = end
    #    n += 1
    #    weightsr2 = torch.cat([weightsr2,tw2],dim=0)
    #    weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    #  gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    #  gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    #  concat = range(2+self._steps-self._multiplier, self._steps+2)
    #  genotype = Genotype(
    #    normal=gene_normal, normal_concat=concat,
    #    reduce=gene_reduce, reduce_concat=concat
    #  )
    #  return genotype
