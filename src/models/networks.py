# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torchvision
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from collections import OrderedDict
from typing import Tuple

def get_network(hparams, log_prior_prob: torch.Tensor= None):

    if hparams['net_type'] == 'mlp':
        # mlp is used for the MNIST datasets
        if hparams['dataset_name'] == 'CompositionalMNIST':
            lin1 = torch.nn.Linear(3 * 28 * 28, 390)
        else:
            lin1 = torch.nn.Linear(2 * 14 * 14, 390)
        lin2 = torch.nn.Linear(390, 390)        
        if hparams['fc_type'] in ['add_energy']:
            lin3 = AdditiveEnergyModel(
                d_in= 390, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        elif hparams['fc_type'] == 'add_energy_multi_attr':
            lin3 = AdditiveEnergyModelMultiAttr(
                d_in= 390, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        else:
            lin3 = torch.nn.Linear(390, hparams['num_y'])
            lin3.bias.data *= 0
            lin3.weight.data *= 0
            torch.nn.init.xavier_uniform_(lin3.weight)
            torch.nn.init.zeros_(lin3.bias)
            
        for lin in [lin1, lin2]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        net = torch.nn.Sequential(
                OrderedDict([
                    ('lin1', lin1),
                    ('relu1', torch.nn.ReLU(True)),
                    ('lin2', lin2),
                    ('relu2', torch.nn.ReLU(True)),
                    ('fc', lin3),
                    ])
                )

    elif hparams['net_type'] == 'resnet':

        net = torchvision.models.resnet.resnet50(pretrained=True)
        if hparams['fc_type'] == 'standard':
            if hparams['algorithm_name'] in ['LA_Group', 'LA_Cond', 'Group_ERM']:
                fc = torch.nn.Linear(net.fc.in_features, hparams['num_y'] + hparams['num_m'])
            else:
                fc = torch.nn.Linear(net.fc.in_features, hparams['num_y'])            
            fc.bias.data *= 0
            fc.weight.data *= 0
        elif hparams['fc_type'] in ['add_energy']:
            fc = AdditiveEnergyModel(
                d_in= net.fc.in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        elif hparams['fc_type'] == 'add_energy_multi_attr':
            fc = AdditiveEnergyModelMultiAttr(
                d_in= net.fc.in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        if hparams['precompute_features']:
            net = fc
        else:
            net.fc = fc
        
    elif hparams['net_type'] == 'bert':

        net = BertWrapper(
            BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=hparams['num_y']))
        net.zero_grad()
        if hparams['fc_type'] == 'standard':
            if hparams['algorithm_name'] in ['LA_Group', 'LA_Cond', 'Group_ERM']:
                fc = torch.nn.Linear(net.net.classifier.in_features, hparams['num_y'] + hparams['num_m'])
            else:
                fc = torch.nn.Linear(net.net.classifier.in_features, hparams['num_y'])            
            fc.bias.data *= 0
            fc.weight.data *= 0
        elif hparams['fc_type'] in ['add_energy', 'add_energy2']:
            fc = AdditiveEnergyModel(
                d_in= net.net.classifier.in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        elif hparams['fc_type'] == 'add_energy_multi_attr':
            fc = AdditiveEnergyModelMultiAttr(
                d_in= net.net.classifier.in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
            
        if hparams['precompute_features']:
            net = fc
        else:
            net.net.classifier= fc

    elif hparams['net_type'] == 'linear':
        dim_in_features= 201
        if hparams['fc_type'] == 'standard':
            if hparams['algorithm_name'] in ['LA_Group', 'LA_Cond', 'Group_ERM']:
                net = torch.nn.Linear(dim_in_features, hparams['num_y'] + hparams['num_m'])
            else:
                net = torch.nn.Linear(dim_in_features, hparams['num_y'])            
            net.bias.data *= 0
            net.weight.data *= 0
        elif hparams['fc_type'] == 'add_energy':
            net = AdditiveEnergyModel(
                d_in= dim_in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )
        elif hparams['fc_type'] == 'add_energy_multi_attr':
            net = AdditiveEnergyModelMultiAttr(
                d_in= dim_in_features, 
                num_classes= hparams['num_y'],
                num_attr= hparams['num_m'],
            )

    return net


def get_optim(hparams, net):

    if hparams['net_type'] == 'mlp':

        opt = torch.optim.Adam(
            net.parameters(),
            hparams['lr'],
            weight_decay=hparams['weight_decay'])
        opt.lr_scheduler = None
        return opt

    if hparams['net_type'] == 'resnet' or hparams['net_type'] == 'linear':

        opt = torch.optim.SGD(
            net.parameters(),
            hparams['lr'],
            momentum=0.9,
            weight_decay=hparams['weight_decay'])
        opt.lr_scheduler = None
        return opt
    
    #For simple 2d experiments
    if hparams['net_type'] in ['linear', 'non-linear']:

        opt = torch.optim.Adam(
            net.parameters(),
            hparams['lr'],
            weight_decay=hparams['weight_decay'])
        opt.lr_scheduler = None
        return opt
    
    if hparams['net_type'] == 'bert':

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in net.named_parameters():
            if any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        opt_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": hparams['weight_decay'],
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(
            opt_grouped_parameters,
            lr=hparams['lr'],
            eps=1e-8)

        def lr_lambda(current_step):
            warmup = hparams["num_step"] // 3
            tot = warmup + hparams["num_step"]
            if current_step < warmup:
                return 1.0 - current_step / warmup
            else:
                return 1.0 - (current_step - warmup) / (tot - warmup)
        opt.lr_scheduler = LambdaLR(opt, lr_lambda)

        return opt


class BertWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


class TwinNets(torch.nn.Module):
    def __init__(self, hparams, net_a, net_b):
        super().__init__()
        self.device = hparams["device"]
        self.hparams = hparams
        self.net_a = net_a
        self.net_b = net_b

    def forward(self, x):
        return torch.cat([self.net_a(x)[..., None],
                          self.net_b(x)[..., None]], -1)


class AdditiveEnergyModel(nn.Module):
    def __init__(self,
                d_in: int, 
                num_classes: int,
                num_attr: int,
                num_layers: int= 1,
                d_model: int= 10
            ):        
        super(AdditiveEnergyModel, self).__init__()
        self.num_classes= num_classes
        self.num_attr= num_attr

        if num_layers == 1:
            self.w_y=[nn.Linear(d_in, self.num_classes, bias=False)]
            self.w_a= [nn.Linear(d_in, self.num_attr, bias=False)]
        else:
            self.w_y=[nn.Linear(d_in, d_model, bias=False)]
            self.w_a= [nn.Linear(d_in, d_model, bias=False)]
            for idx in range(num_layers-1):
                self.w_y.append( nn.Linear( d_model, d_model) )
                self.w_y.append( nn.LeakyReLU() )
                self.w_a.append( nn.Linear( d_model, d_model) )
                self.w_a.append( nn.LeakyReLU() )
            self.w_y.append( nn.Linear( d_model, self.num_classes ) )
            self.w_a.append( nn.Linear( d_model, self.num_attr ) )

        self.w_y= nn.Sequential(*self.w_y)
        self.w_a= nn.Sequential(*self.w_a)
        self.offset= nn.Parameter( torch.zeros( (self.num_classes, self.num_attr) ), requires_grad=True )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """" 
        Returns the energy matrix E(x). Output is of the shape (batch size, num classes,  num attributes)
        """
        out= self.w_y(x).unsqueeze(-1) + self.w_a(x)[:, None]
        return out

class AdditiveEnergyModelMultiAttr(nn.Module):
    def __init__(self,
                d_in: int, 
                num_classes: int,
                num_attr: int,
                num_layers: int= 1,
                d_model: int= 64
            ):        
        super(AdditiveEnergyModelMultiAttr, self).__init__()
        self.num_classes= num_classes
        self.num_attr= num_attr
        self.total_spur_attr= int(np.log2(self.num_attr))
        self.feat_shape= (self.num_classes,) + self.total_spur_attr * (2,)
        
        #List of linear layers for each attribute
        self.net= []
        self.net.append( nn.Linear(d_in, self.num_classes, bias=False) )
        for idx in range(self.total_spur_attr):
            self.net.append( nn.Linear(d_in, 2, bias=False) )
        self.net= nn.ModuleList(self.net)

        self.offset= nn.Parameter( torch.zeros(self.feat_shape), requires_grad=True )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """" 
        Returns the energy matrix E(x). Output is of the shape (batch size, num classes, 2, ..., 2)
        """
        out= []
        for idx in range(1+ self.total_spur_attr):
            curr_out= self.net[idx](x)
            #Reshape for broadcast addition: Unsqueeze 1 along attribtues != idx
            curr_out= curr_out.view( (curr_out.shape[0],) +  (1,) * idx + (curr_out.shape[1],) + (self.total_spur_attr - idx) * (1,) )
            out.append(curr_out)
        return sum(out)
