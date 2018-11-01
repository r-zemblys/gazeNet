#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:40:20 2017

@author: raimondas
"""

#%% imports
import os, sys, glob, time
import itertools
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
#import matplotlib.pyplot as plt
#plt.ioff()

#import seaborn as sns
#sns.set_style("ticks")

###
import json
import argparse
import pickle
import random

import pandas as pd

import torch
from torch.autograd import Variable

from tensorboard import summary
from tensorboard import FileWriter

from utils_lib import utils
from utils_lib.data_loader import EMDataset, GazeDataLoader, load_npy_files
from utils_lib.ETeval import run_infer

from model import gazeNET as gazeNET
import model as model_func

import copy

#%% functions

def get_arguments():
    parser = argparse.ArgumentParser(description='gazeNet')
    parser.add_argument('--model_dir', type=str, default='model_dev',
                        help='Directory in which to store the logging ')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_epochs', default=500, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=220617, help='seed')

    return parser.parse_args()

#%% init variables
args = get_arguments()

logdir =  os.path.join('logdir', args.model_dir)
fname_config = os.path.join(logdir, 'config.json')
if os.path.exists(fname_config):
    configuration = utils.Config(fname_config)
    config = configuration.params
else:
    print("No config file found in %s" % args.model_dir)
    sys.exit()


if (torch.cuda.device_count()>0) & config['cuda']:
    batch_size = config["batch_size"]
    batch_size*=torch.cuda.device_count()
    cuda = True
else:
    batch_size = 100
    cuda = False

#%% load data
print ("Preparing data")
dir_data = 'logdir/%s/data'% (args.model_dir)

##train dataset
log_writer_train = FileWriter('%s/TB/train' % logdir)
fname_train = '%s/%s'% (dir_data, config['data_train'][0])
if os.path.exists('%s'%fname_train):
    if os.path.splitext(fname_train)[-1] == '.pkl':
        #if training data is pickle file
        with open(fname_train, 'rb') as f:
            X_train = pickle.load(f)
    else:
        #!!! EXPERIMENTAL !!!
        X_train = load_npy_files('%s/*.npy'%(fname_train))
else:
    print("Train data does not exist. Run data preparation. Exiting")
    sys.exit()

##val dataset
log_writer_val = FileWriter('%s/TB/val' % logdir)
fname_val = '%s/%s'% (dir_data, config['data_val'][0])
if os.path.exists('%s'%fname_val):
    if os.path.splitext(fname_val)[-1] == '.pkl':
        #if validation data is pickle file
        with open(fname_val, 'rb') as f:
            X_val = pickle.load(f)
            X_val = [_d for _t, _d in X_val]
    else:
        #!!! EXPERIMENTAL !!!
        X_val = load_npy_files('%s/*.npy'%(fname_val))
else:
    print("Validation data does not exist. Run data preparation. Exiting")
    sys.exit()

##data used to train generative model
log_writer_train_gen = FileWriter('%s/TB/train_gen' % logdir)
fname_train_gen = '%s/%s'% (dir_data, config['data_train_gen'][0])
if os.path.exists('%s'%fname_train_gen):
    if os.path.splitext(fname_train_gen)[-1] == '.pkl':
        #if train_gen data is pickle file
        with open(fname_train_gen, 'rb') as f:
            X_train_gen = pickle.load(f)
            X_train_gen = [_d for _t, _d in X_train_gen]
    else:
        #!!! EXPERIMENTAL !!!
        X_train_gen = load_npy_files('%s/*.npy'%(fname_train_gen))
else:
    print("Train (gen) data does not exist. Run data preparation. Exiting")
    sys.exit()

#%%prepare data loaders
kwargs = {'seed': args.seed}
#train data
dataset_train = EMDataset(config = config, gaze_data = X_train)
loader_train = GazeDataLoader(dataset_train, batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True, **kwargs)

#val data
config_val = copy.deepcopy(config)
config_val['split_seqs']=False
config_val['batch_size']=1
config_val['augment']=False

dataset_val = EMDataset(config = config_val, gaze_data = X_val)
loader_val = GazeDataLoader(dataset_val, batch_size=1,
                            num_workers=args.num_workers,
                            shuffle=False, **kwargs)

dataset_train_gen = EMDataset(config = config_val, gaze_data = X_train_gen)
loader_train_gen = GazeDataLoader(dataset_train_gen, batch_size=1,
                                 num_workers=args.num_workers,
                                 shuffle=False, **kwargs)

#%% prepare model
num_classes = len(config['events'])
model = gazeNET(config, num_classes, args.seed)
n_params = model_func.calc_params(model)
print("Number of parameters: %s" % utils.human_format(sum(n_params.values())))

#note that epoch increases every run
_, epoch_start = model_func.load(model, args.model_dir, config)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

parameters = model.parameters()
optimizer = torch.optim.RMSprop(parameters, lr=config["learning_rate"])

#event weights
event_stats = np.array([_e for _x in X_train_gen
                           for _e in _x['evt'].tolist()
                           if not(len(_x)<config['seq_len']+1)])
event_stats = utils.convertToOneHot(event_stats-1, len(np.unique(event_stats)))
event_weights = event_stats.sum(0)[:3]
event_weights = event_weights.astype(np.float32)/event_weights.sum()
weights = torch.FloatTensor(1-event_weights[:3])

if cuda:
    weights_cuda = weights.cuda()
    criterion = torch.nn.CrossEntropyLoss(weights_cuda)
else:
    criterion = torch.nn.CrossEntropyLoss(weights)

#%%training
model.train()
val_score_best = 0
for epoch in range(epoch_start, args.num_epochs+1): #because we start from 1
    iterator = tqdm(loader_train)
    end = time.time()
    for step, data in enumerate(iterator):
        global_step = len(loader_train)*(epoch-1) + step

        ##Prepare data
        inputs, targets, input_percentages, target_sizes, _ = data
        t_data = time.time() - end

        t_model_s = time.time()
        inputs = Variable(inputs)
        y_ = Variable(targets)
        if cuda:
            inputs = inputs.cuda()
            y_ = y_.cuda()

        ##Forward Pass
        y = model(inputs)
        yt, yn = y.size()[:2]
        y = y.view(yt * yn, -1)
        #WARNING: only works for split_seqs=True;
        #i.e. all sequences need to be same exact length
        loss = criterion(y, y_)

        ##Backward pass
        if np.isfinite(loss.data[0]):
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), config["gradclip"])
            optimizer.step()
        end = time.time()

        iterator.set_description('Epoch: %d, Loss: %.3f, t_data = %.3f, t_model:%.3f' % (epoch, loss.data[0], t_data, end-t_model_s))

        #%%model persistence
        if not(config['save_every']==0) and (global_step%config['save_every'] == 0):
            global_step = len(loader_train)*(epoch-1) + step
            model_func.save(model, args.model_dir, epoch, global_step, config)

        ##validation
        if config['val_every'] > 0 and \
           global_step%config['val_every']==0 and \
           not(global_step==0):
            print ("Running validation...")
            model.eval()
            global_step = len(loader_train)*(epoch-1) + step

            val_s = time.time()
            kwargs = {
                'cuda': cuda
            }

            #train data
            n_samples = len(dataset_val)
            WER_train, CER_train, KE_train, KS_train, _ = \
            run_infer(model, n_samples, loader_train, **kwargs)

            #val data
            n_samples = len(dataset_val)
            WER_val, CER_val, KE_val, KS_val, _ = \
            run_infer(model, n_samples, loader_val, **kwargs)

            val_score = np.array([KE_val[-1], KS_val[-1], WER_val, CER_val])
            val_score[2:] = 1-val_score[2:]
            val_score=np.linalg.norm(val_score)

            n_samples = len(dataset_train_gen)
            WER_train_gen, CER_train_gen, KE_train_gen, KS_train_gen , _= \
            run_infer(model, n_samples, loader_train_gen, **kwargs)
            val_e = time.time()

            #save summary
            #model_func.param_summary(model, log_writer_train, global_step)
            summary_infos = [
                {'writer':log_writer_train, 'name': 'train',
                 'measures': {'wer': WER_train,
                              'cer': CER_train,
                              'KE':KE_train[-1],
                              'KS':KS_train[-1],
                              'KE_all':KE_train,
                              'KS_all':KS_train}},

                {'writer':log_writer_val, 'name': 'val',
                 'measures': {'wer': WER_val,
                              'cer': CER_val,
                              'KE':KE_val[-1],
                              'KS':KS_val[-1],
                              'KE_all':KE_val,
                              'KS_all':KS_val,
                              'val_score': val_score}},

                {'writer':log_writer_train_gen, 'name': 'unpaired',
                 'measures': {'wer': WER_train_gen,
                              'cer': CER_train_gen,
                              'KE':KE_train_gen[-1],
                              'KS':KS_train_gen[-1],
                              'KE_all':KE_train_gen,
                              'KS_all':KS_train_gen}},

            ]
            for summary_info in summary_infos:
                info = (summary_info['name'], summary_info['measures']['wer'], summary_info['measures']['cer'],
                    ', '.join(['%.2f' % (_m*100) for _m in np.array(summary_info['measures']['KS_all'])]),
                    ', '.join(['%.2f' % (_m*100) for _m in np.array(summary_info['measures']['KE_all'])]),
                )
                print ('[%s]. WER: %.3f, CER: %.3f, KS: %s, KE: %s' % info)

                writer = summary_info['writer']
                for key, value in summary_info['measures'].iteritems():
                    if not 'all' in key:
                        s = summary.scalar(key, value)
                        writer.add_summary(s, global_step = global_step)

            #save best model
            if val_score > val_score_best:
                val_score_best = val_score
                global_step = len(loader_train)*(epoch-1) + step
                sdir = "logdir/%s/models/best" % args.model_dir
                mkpath(sdir)
                fname_model = 'gazeNET_%04d_%08d_K%.4f.pth.tar' %(epoch, global_step, val_score)
                file_path = '%s/%s' % (sdir, fname_model)

                torch.save(model_func.checkpoint(model, step, epoch), file_path)

            #switch back to train mode
            model.train()

    #%% on epoch done
    #save model
    #global_step = len(train_loader)*(epoch-1) + step
    #config = model_func.save(model, args.model_dir, epoch, global_step, config)
    #config['learning_rate'] = config['learning_rate']/config['learning_rate_anneal']
    #model_func.anneal_learning_rate(optimizer, config['learning_rate'])
    #configuration.save_params(config)