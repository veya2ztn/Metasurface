from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from mltool.ModelArchi.ModelSearch.genotype import Genotype
class SuperNetwork(nn.Module):
    def __init__(self, C, num_classes, nodes, layers, criterion):
        super(SuperNetwork, self).__init__()
        self._C = C
        self._nodes = nodes
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion

    @abstractmethod
    def forward(self, input, discrete):
        pass

    @abstractmethod
    def new(self):
        pass

    def _parse(self, weights,deleted_zero=False, withnone=False):
        # weights is metrix  (edges_num = [n]+[n+1]+[n+2]+..., ops_num)
        # edges equal to the (inital_input[set as 2] + before node) so it is i+2
        n     = 2
        start = 0
        pick_up_num = 2
        # print("============================")
        # print(weights)
        # print("============================")
        weight_divide_indexes=[]
        for i in range(self._nodes):
            end = start + n
            weight_divide_indexes.append([start,end])
            n   = n+1
            start = end
        # so the weight will be divided into [0,2] [2,5] [5,9] [9,14]
        gene  = []
        for i, (start, end) in enumerate(weight_divide_indexes):
            W = weights[start:end].copy() #(K, ops_num)
            node_information=[]
            for i,line in enumerate(W):
                arg_sort_idx        = np.argsort(line) # arg sort from small to big
                argmax_in_this_line = arg_sort_idx[-1]
                max_val_in_possible = line[argmax_in_this_line]/(np.sum(line)+1e-5)
                # we will take the largest value in this line as its sign
                # but if it is "none" we will check the normed value, and only let "none" pass for big than 0.98
                if (self.op_names[argmax_in_this_line] == "none" and max_val_in_possible<0.98) or (not withnone):
                    argmax_in_this_line = arg_sort_idx[-2]
                max_val_in_this_line        = line[argmax_in_this_line]
                ops_for_max_val_in_this_line= self.op_names[argmax_in_this_line]
                node_information.append([i,max_val_in_this_line, argmax_in_this_line, ops_for_max_val_in_this_line])

            #print(node_information)
            # now, we will pick the best two line
            # the rule is first pick out the ops is not none one
            # then pick from none
            node_information_not_none = [k for k in node_information if k[-1] !="none"]
            line_order = np.argsort([k[1] for k in node_information_not_none])[::-1]
            if len(line_order) < pick_up_num:
                active_ind = [node_information_not_none[idx][0] for idx in line_order]
                node_information_is_none  = [k for k in node_information if k[-1] =="none"]
                line_order = np.argsort([k[1] for k in node_information_is_none])[::-1]
                for idx in line_order:
                    if len(active_ind) >= pick_up_num:break
                    active_ind.append(node_information_is_none[idx][0])
            else:
                active_ind = [node_information_not_none[idx][0] for idx in line_order[:pick_up_num]]

            for j in active_ind:
                ops_best_idx = np.argmax(W[j])
                if W[j].sum()<1e-6 and deleted_zero:
                    gene.append(("deleted", j))
                else:
                    gene.append((self.op_names[ops_best_idx], j))
        return gene
    def old_parse(self, weights,deleted_zero=False, withnone=False):
        gene = []
        n = 2
        start = 0
        for i in range(self._nodes):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(
                range(i + 2),
                key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if ((self.op_names[k] != "none") or withnone)
                ),
            )[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if ((self.op_names[k] != "none") or withnone):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                if W[j].sum()<1e-10 and deleted_zero:
                    gene.append(("deleted", j))
                else:
                    gene.append((self.op_names[k_best], j))
            start = end
            n += 1
        return gene

    def genotype(self, weights,**kargs):
        normal_weights = weights["normal"]
        reduce_weights = weights["reduce"]
        gene_normal = self._parse(normal_weights,**kargs)
        gene_reduce = self._parse(reduce_weights,**kargs)

        concat = range(2 + self._nodes - self._multiplier, self._nodes + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
            init_channels=self._C,
            num_classes=self._num_classes,
            layers= self._layers,
            nodes = self._steps,
        )


        return genotype


    def store_init_weights(self):
        self.init_parameters = {}
        for name, w in self.named_parameters():
            self.init_parameters[name] = torch.Tensor(w.cpu().data).cuda()

    def get_save_states(self):
        return {
            "state_dict": self.state_dict(),
            "init_parameters": self.init_parameters,
        }

    def load_states(self, save_states):
        self.load_state_dict(save_states["state_dict"])
        self.init_parameters = save_states["init_parameters"]

    def compute_norm(self, from_init=False):
        norms = {}
        for name, w in self.named_parameters():
            if from_init:
                norms[name] = torch.norm(self.init_parameters[name] - w.data, 2)
            else:
                norms[name] = torch.norm(w.data, 2)
        return norms

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_edge_weights(self, edges):
        self.edges = edges

    def _loss(self, input, target, discrete=False):
        logits, _ = self(input, discrete=discrete)
        return self._criterion(logits, target)
