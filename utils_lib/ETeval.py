#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: rz
@email:
"""
#%% imports
import itertools, time, copy

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import Levenshtein as Lev
from sklearn import metrics


from .etdata import ETData
from .utils import convertToOneHot

#%% setup parameters


#%% code

def calc_k(gt, pr):
    '''
    Handles error if all samples are from the same class
    '''
    k = 1. if (gt == pr).all() else metrics.cohen_kappa_score(gt, pr)
    return k

def calc_f1(gt, pr):
    f1 = 1. if (gt == pr).all() else metrics.f1_score(gt, pr)
    return f1

def calc_KE(etdata_gt, etdata_pr):

    #calculate event level matches
    gt_evt_index = [ind for i, n in enumerate(np.diff(etdata_gt.evt[['s', 'e']]).squeeze()) for ind in itertools.repeat(i, n)]
    pr_evt_index = [ind for i, n in enumerate(np.diff(etdata_pr.evt[['s', 'e']]).squeeze()) for ind in itertools.repeat(i, n)]

    overlap = np.vstack((gt_evt_index, pr_evt_index)).T

    overlap_matrix = [_k + [len(list(_g)), False, False] for _k, _g in itertools.groupby(overlap.tolist())]
    overlap_matrix = pd.DataFrame(overlap_matrix, columns=['gt', 'pr', 'l', 'matched', 'selected'])
    overlap_matrix['gt_evt'] = etdata_gt.evt.loc[overlap_matrix['gt'], 'evt'].values
    overlap_matrix['pr_evt'] = etdata_pr.evt.loc[overlap_matrix['pr'], 'evt'].values

    while not(overlap_matrix['matched'].all()):
        #select longest overlap
        ind = overlap_matrix.loc[~overlap_matrix['matched'], 'l'].argmax()
        overlap_matrix.loc[ind, ['selected']]=True
        mask_matched = (overlap_matrix['gt']==overlap_matrix.loc[ind, 'gt']).values |\
                       (overlap_matrix['pr']==overlap_matrix.loc[ind, 'pr']).values
        overlap_matrix.loc[mask_matched, 'matched'] = True
    overlap_events = overlap_matrix.loc[overlap_matrix['selected'], ['gt', 'pr', 'gt_evt', 'pr_evt']]

    #sanity check
    evt_gt = etdata_gt.evt.loc[overlap_events['gt'], 'evt']
    evt_pr = etdata_pr.evt.loc[overlap_events['pr'], 'evt']
    #assert (evt_gt.values == evt_pr.values).all()

    #add not matched events
    set_gt = set(etdata_gt.evt.index.values) - set(evt_gt.index.values)
    set_pr = set(etdata_pr.evt.index.values) - set(evt_pr.index.values)

    evt_gt = pd.concat((evt_gt, etdata_gt.evt.loc[set_gt, 'evt']))
    evt_pr = pd.concat((evt_pr, pd.DataFrame(np.zeros(len(set_gt)))))

    evt_gt = pd.concat((evt_gt, pd.DataFrame(np.zeros(len(set_pr)))))
    evt_pr = pd.concat((evt_pr, etdata_pr.evt.loc[set_pr, 'evt']))

    return overlap_events.values, np.squeeze(evt_gt.values.astype(np.int32)), np.squeeze(evt_pr.values.astype(np.int32))


def eval_evt(etdata_gt, etdata_pr, n_events):

    t = time.time()
    if etdata_gt.evt is None:
        etdata_gt.calc_evt(fast=True)
    if etdata_pr.evt is None:
        etdata_pr.calc_evt(fast=True)

    #levenshtein distance
    evt_gt = etdata_gt.evt['evt']
    evt_gt = evt_gt[~(evt_gt==0)]
    evt_pr = etdata_pr.evt['evt']
    evt_pr = evt_pr[~(evt_pr==0)]
    wer = Lev.distance(''.join(map(str, evt_gt)),
                       ''.join(map(str, evt_pr)))/\
                       float(len(evt_gt))

    _cer = map(lambda _a, _b: Lev.distance(_a, _b),
               ''.join(map(str, etdata_gt.data['evt'])).split('0'),
               ''.join(map(str, etdata_pr.data['evt'])).split('0'))
    mask=etdata_gt.data['evt']==0
    evt_len = float(sum(~mask))
    cer = sum(_cer)/evt_len

    #sample level K
    t = time.time()
    evts_gt_oh = convertToOneHot(etdata_gt.data['evt'], n_events)
    evts_pr_oh = convertToOneHot(etdata_pr.data['evt'], n_events)
    ks = [calc_k(evts_gt_oh[:,i], evts_pr_oh[:,i]) for i in range(1, n_events)]

    evt_gt = etdata_gt.data['evt']
    evt_gt = evt_gt[~(evt_gt==0)]
    evt_pr = etdata_pr.data['evt']
    evt_pr = evt_pr[~(evt_pr==0)]
    ks_all = metrics.cohen_kappa_score(evt_gt, evt_pr)

    ks.extend([ks_all])

    #event level K and F1
    try:
        t = time.time()

        ke_ = []
        f1e_ = []
        for evt in range(1, 4):
            #evt=1
            _etdata_gt = copy.deepcopy(etdata_gt)
            mask_ext = _etdata_gt.data['evt']==0
            mask = _etdata_gt.data['evt']==evt
            _etdata_gt.data['evt'][mask]=1
            _etdata_gt.data['evt'][~mask]=0
            _etdata_gt.data['evt'][mask_ext]=255
            _etdata_gt.calc_evt(fast=True)

            _etdata_pr = copy.deepcopy(etdata_pr)
            mask_ext = _etdata_pr.data['evt']==0
            mask = _etdata_pr.data['evt']==evt
            _etdata_pr.data['evt'][mask]=1
            _etdata_pr.data['evt'][~mask]=0
            _etdata_pr.data['evt'][mask_ext]=255
            _etdata_pr.calc_evt(fast=True)

            evt_overlap, evt_gt, evt_pr = calc_KE(_etdata_gt, _etdata_pr)
            mask = (evt_gt==255) & (evt_pr==255)
            evt_gt = evt_gt[~mask]
            evt_pr = evt_pr[~mask]
            ke_.append(calc_k(evt_gt, evt_pr))
            f1e_.append(calc_f1(evt_gt, evt_pr))


        evt_overlap, evt_gt, evt_pr = calc_KE(etdata_gt, etdata_pr)
        mask = (evt_gt==0) & (evt_pr==0)
        evt_gt = evt_gt[~mask]
        evt_pr = evt_pr[~mask]
        #print ('[overlap], dur %.2f' % (time.time()-t))
        evt_gt_oh = convertToOneHot(evt_gt, n_events)
        evt_pr_oh = convertToOneHot(evt_pr, n_events)
        ke = [calc_k(evt_gt_oh[:,i], evt_pr_oh[:,i]) for i in range(1, n_events)]
        f1e = [calc_f1(evt_gt_oh[:,i], evt_pr_oh[:,i]) for i in range(1, n_events)]

        ke_all = metrics.cohen_kappa_score(evt_gt, evt_pr)
        f1_all = metrics.f1_score(evt_gt, evt_pr, average='weighted')
        ke.extend([ke_all])
        ke_.extend([ke_all])
        f1e.extend([f1_all])
        f1e_.extend([f1_all])
        #print ('[KE], dur %.2f' % (time.time()-t))
    except:
        #TODO: Debug
        print ("Could not calculate event level k")
        ks = [0.,]*(n_events+1)
        ke = [0.,]*(n_events+1)
        f1e = [0.,]*(n_events+1)


    return wer, cer, ke_, ks, f1e_, (evt_overlap, evt_gt, evt_pr)

def run_infer(model, n_samples, data_loader, **kwargs):
    fs = 500.
    cuda = False if not(kwargs.has_key("cuda")) else kwargs["cuda"]
    use_tqdm = False if not(kwargs.has_key("use_tqdm")) else kwargs["use_tqdm"]
    perform_eval = True if not(kwargs.has_key("eval")) else kwargs["eval"]
    #save_dir = None if not(kwargs.has_key("save_dir")) else kwargs["save_dir"]

    etdata_pr = ETData()
    etdata_gt = ETData()
    _etdata_pr = []
    _etdata_gt = []
    _pr_raw=[]

    sample_accum = 0
    t = time.time()
    iterator = tqdm(data_loader) if use_tqdm else data_loader
    for data in iterator:
        inputs, targets, input_percentages, target_sizes, aux = data

        #do forward pass
        inputs = Variable(inputs, volatile=True).contiguous()
        if cuda:
            inputs = inputs.cuda()
        y = model(inputs)
        seq_length = y.size(1)
        sizes = Variable(input_percentages.mul(int(seq_length)).int())

        if cuda:
            inputs = inputs.cpu()
            y = y.cpu()
            sizes = sizes.cpu()

            targets = targets.cpu()

        #decode output
        outputs_split = [_y[:_l] for _y, _l in zip(y.data, target_sizes)]

        events_decoded = [torch.max(_o, 1)[1].numpy().flatten() for _o in outputs_split]
        events_target= np.array_split(targets.numpy(), np.cumsum(sizes.data.numpy())[:-1])

        trials = [np.cumsum(_y[0, :, :_l], axis=1).T for _y, _l in zip(inputs.data.numpy(), target_sizes)]

        for ind, (gt, pr, pr_raw, tr) in enumerate(zip(events_target, events_decoded, outputs_split, trials)):
            #TODO:
            #check why sizes do not match sometimes

            minl = min(len(gt), len(pr))
            gt = gt[:minl]
            pr = pr[:minl]
            _pr_raw.append(pr_raw.numpy())
            #pr = np.hstack((pr[0], pr[:-1]))
            _etdata_pr.extend(zip(np.arange(len(gt))/fs,
                          tr[:,0],
                          tr[:,1],
                          itertools.repeat(True),
                          pr+1
                       ))
            _etdata_pr.append((0, )*5)
            _etdata_gt.extend(zip(np.arange(len(gt))/fs,
                          tr[:,0],
                          tr[:,1],
                          itertools.repeat(True),
                          gt+1
                       ))
            _etdata_gt.append((0, )*5)

            sample_accum+=1

        if sample_accum >= n_samples:
            break
    print ('[FP], n_samples: %d, dur: %.2f' % (sample_accum, time.time()-t))

    if perform_eval:
        #run evaluation
        etdata_pr.load(np.array(_etdata_pr), **{'source':'np_array'})
        etdata_gt.load(np.array(_etdata_gt), **{'source':'np_array'})
        wer, cer, ke, ks, _, (evt_overlap, _, _) = eval_evt(etdata_gt, etdata_pr, 4)
        return wer, cer, ke, ks, (_etdata_gt, _etdata_pr, _pr_raw)
    else:
        return _etdata_gt, _etdata_pr, _pr_raw