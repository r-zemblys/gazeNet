#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:35:06 2017

@author: raimondas
"""

from collections import OrderedDict
import os
from distutils.dir_util import mkpath

import torch
import torch.nn as nn

from tensorboard import summary

def checkpoint(model, step=None, epoch=None):
    package = {
        'epoch': epoch if epoch else 'N/A',
        'step': step if step else 'N/A',
        'state_dict': model.state_dict(),
    }
    return package

def anneal_learning_rate(optimizer, lr):
    optim_state = optimizer.state_dict()
    optim_state['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict(optim_state)
    print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def load(model, model_dir, config, model_name=None):
    if len(config["model_name"]) or (model_name is not None):
        model_name = config["model_name"][-1] if model_name is None else model_name
    else:
        model_name = None
    logdir = "logdir/%s/models" % model_dir

    fpath_model = "%s/%s" % (logdir, model_name)
    print (fpath_model)
    if os.path.exists(fpath_model) and (model_name is not None):
        print "Loading model: %s" % fpath_model

        package = torch.load(fpath_model, map_location=lambda storage, loc: storage)
        epoch = package['epoch']+1 if not(package['epoch'] == 'N/A') else 1
        #edit variable names for loading in cpu
        #if not(config["cuda"]):
        for k in package['state_dict'].keys():
            package['state_dict'][k.replace('module.', '', 1)] = package['state_dict'].pop(k)

        state_dict = dict()
        for k in model.state_dict().keys():
            if package['state_dict'].has_key(k):
                state_dict[k] = package['state_dict'][k]
        model_state = model.state_dict()
        model_state.update(state_dict)
        model.load_state_dict(model_state)
        print ("done.")
    else:
        epoch = 1
        print "Pretrained model not found"
    return model_name, epoch

def save(model, model_dir, epoch, step,config):
    logdir = "logdir/%s/models" % model_dir
    mkpath(logdir)
    fname_model = 'gazeNET_%04d_%08d.pth.tar' %(epoch, step)
    file_path = '%s/%s' % (logdir, fname_model)

    torch.save(checkpoint(model, step, epoch), file_path)
    config["model_name"].append(fname_model)
    model_list = config["model_name"][-config['max_to_keep']:]
    remove_list = config["model_name"][:-config['max_to_keep']:]
    for _rm in remove_list:
        fpath_rm = '%s/%s' % (logdir, _rm)
        if os.path.exists(fpath_rm):
            os.remove(fpath_rm)
    config["model_name"] = model_list
    return config

def calc_params(model):
    all_params = OrderedDict()
    params = model.state_dict()

    for _p in params.keys():
        #if not('ih_l0_reverse' in _p):
        all_params[_p] = params[_p].nelement()
    return all_params

def param_summary(model, writer, step):
    state = model.state_dict()
    for _p in state.keys():
        param = state[_p].cpu().numpy()

        s = summary.histogram(_p, param.flatten())
        writer.add_summary(s, global_step = step)

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True, keep_prob=0.5):
        super(BatchRNN, self).__init__()
        self.batch_norm = batch_norm
        self.bidirectional = bidirectional

        rnn_bias = False if batch_norm else True
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          batch_first=True,
                          bias=rnn_bias)
        self.batch_norm_op = SequenceWise(nn.BatchNorm1d(hidden_size))

        self.dropout_op = nn.Dropout(1-keep_prob)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.contiguous()
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            x = x.contiguous()
        if self.batch_norm:
            x = self.batch_norm_op(x)
        x = self.dropout_op(x)
        return x


#%%
class gazeNET(nn.Module):
    def __init__(self, config, num_classes, seed=220617):
        super(gazeNET, self).__init__()
        torch.manual_seed(seed)
        if (torch.cuda.device_count()>0):
            torch.cuda.manual_seed(seed)

        if config['architecture'].has_key('conv_stack'):
            ## convolutional stack
            conv_config = config['architecture']['conv_stack']
            conv_stack = []
            #feat_dim = int(math.floor((config['sample_rate'] * 2*config['window_stride']) / 2) + 1)
            feat_dim = 2
            in_channels = 1
            for _conv in conv_config:
                name, out_channels, kernel_size, stride = _conv
                padding = map(lambda x: x/2, kernel_size)
                _conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=tuple(kernel_size), stride=tuple(stride),
                              padding = tuple(padding),
                              bias = False
                              )
                #init_vars.xavier_uniform(conv_op.weight, gain=np.sqrt(2))
                _conv = nn.Sequential(
                    _conv,
                    nn.BatchNorm2d(out_channels),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Dropout(1-config['keep_prob']),
                )
                conv_stack.append((name, _conv))
                in_channels = out_channels
                feat_dim = feat_dim/stride[0]+1
            self.conv_stack = nn.Sequential(OrderedDict(conv_stack))
            rnn_input_size = feat_dim * out_channels
        else:
            self.conv_stack = None
            rnn_input_size = 2

        ## RNN stack
        rnn_config = config['architecture']['rnn_stack']
        rnn_stack = []
        for _rnn in rnn_config:
            name, hidden_size, batch_norm, bidirectional = _rnn
            _rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, batch_norm=batch_norm,
                            keep_prob = config['keep_prob'])
            rnn_stack.append((name, _rnn))
            rnn_input_size = hidden_size
        self.rnn_stack = nn.Sequential(OrderedDict(rnn_stack))

        ## FC stack
        self.fc = nn.Sequential(
            SequenceWise(nn.Linear(hidden_size, num_classes, bias=False)),
        )
    ### forward
    def forward(self, x):
        if self.conv_stack is not None:
            x = self.conv_stack(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # TxNxH

        x = self.rnn_stack(x)

        x = self.fc(x)
        return x