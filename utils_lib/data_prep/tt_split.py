#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:24:07 2017

@author: raimondas
"""
#%% imports
import os, sys, glob
import itertools
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
#import matplotlib.pyplot as plt
#plt.ion()

sys.path.append('..')

#import seaborn as sns
#sns.set_style("ticks")

###
import random
import copy
from collections import OrderedDict

import pandas as pd
import argparse
#from utils import ETData, training_params
import pickle

from sklearn.model_selection import train_test_split
import scipy.io as scio

#from utils import eval_evt
from etdata import ETData, get_px2deg



#%% functions
def augment_with_saccades(data, seq_len, histogram_eq = True):
    #debug
#    stop
#    data = data['unpaired_clean']
#    seq_len=100
    #augment by centering on saccades
    etdata = ETData()
    sacc = []
    data_clean = [_d for (_i, _d) in data if len(_d) > seq_len]
    for i, _data in enumerate(data_clean):
        etdata.load(_data, **{'source':'array'})
        etdata.calc_evt()
        evt = etdata.evt.loc[etdata.evt['evt']==2, :]
        evt = evt.assign(ind=i)
        sacc.append(evt)

    sacc_df = pd.concat(sacc).reset_index()
    seeds = range(len(sacc_df))
    train_data_pick = []
    for (_, sacc), seed in zip(sacc_df.iterrows(), seeds):
        np.random.seed(seed)
        i = np.random.randint(args.seq_len)
        trial_ind = int(sacc['ind'])

        s = np.maximum(0, sacc['s'] - i).astype(int)
        e = s + args.seq_len + 2 #because we will differentiate and predict next sample
        if e < len(data_clean[trial_ind]):
            #plt.plot(_train_data[trial_ind][s:e]['x'])
            #plt.plot(_train_data[trial_ind][s:e]['y'])
            train_data_pick.append(data_clean[trial_ind][s:e])

    if histogram_eq:
        #augment with large saccades
        sacc=[]
        for i, _data in enumerate(train_data_pick):
            etdata.load(_data, **{'source':'array'})
            etdata.calc_evt()
            evt = etdata.evt.loc[etdata.evt['evt']==2, :]
            evt = evt.assign(ind=i)
            sacc.append(evt)

        sacc_pick_df = pd.concat(sacc).reset_index()

        h, edges = np.histogram(sacc_pick_df['ampl'], bins='auto')
        p = (h.max()-h)
        sacc = []
        seeds = range(len(h))
        for _p, _es, _ee, seed in zip(p, edges[:-1], edges[1:], seeds):
            mask = (sacc_df['ampl'] > _es) & (sacc_df['ampl'] < _ee)
            if (len(sacc_df.loc[mask, :]) > 0) and (_p > 0):
                sacc.append(sacc_df.loc[mask, :].sample(n = _p, replace = True, random_state=seed))
        sacc_ampl_df = pd.concat(sacc).reset_index()

        train_data_ampl = []
        seeds = range(len(sacc_ampl_df))
        for (_, sacc), seed in zip(sacc_ampl_df.iterrows(), seeds):
            np.random.seed(seed)
            i = np.random.randint(args.seq_len)
            trial_ind = int(sacc['ind'])

            s = np.maximum(0, sacc['s'] - i).astype(int)
            e = s + args.seq_len + 2 #because we will differentiate and predict next sample
            if e < len(data_clean[trial_ind]):
                #plt.plot(_train_data[trial_ind][s:e]['x'])
                #plt.plot(_train_data[trial_ind][s:e]['y'])
                train_data_ampl.append(data_clean[trial_ind][s:e])
        train_data_pick +=train_data_ampl
    return train_data_pick

#%% set parameters

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../../etdata',
                   help='data root')
parser.add_argument('--dataset', type=str, default='lund2013_npy',
                   help='dataset')
parser.add_argument('--seq_len', type=int, default=100,
                   help='number of samples in data iterator')
parser.add_argument('--events', default=[1, 2, 3],
                   help='events')
args = parser.parse_args()

etdata = ETData()

#%%data reader
print ("Reading data")
ddir = '%s/%s'%(args.root, args.dataset)
if not os.path.exists(ddir):
    mkpath(ddir)

    #try to convert from mat
    fdir_mat = 'EyeMovementDetectorEvaluation/annotated_data/originally uploaded data/images'
    FILES_MAT = glob.glob('%s/%s/*.mat'% (args.root, fdir_mat))

    for fpath in tqdm(FILES_MAT):
        fdir, fname = os.path.split(os.path.splitext(fpath)[0])

        mat = scio.loadmat(fpath)
        fs = mat['ETdata']['sampFreq'][0][0][0][0]
        geom = {
            'screen_width' :mat['ETdata']['screenDim'][0][0][0][0],
            'screen_height': mat['ETdata']['screenDim'][0][0][0][1],
            'display_width_pix' : mat['ETdata']['screenRes'][0][0][0][0],
            'display_height_pix' :mat['ETdata']['screenRes'][0][0][0][1],
            'eye_distance' : mat['ETdata']['viewDist'][0][0][0][0],
        }
        px2deg = get_px2deg(geom)

        data = mat['ETdata']['pos'][0][0]
        t = np.arange(0, len(data)).astype(np.float64)/fs
        status = (data[:,3] == 0) | (data[:,4] == 0)

        data = np.vstack((t, data[:,3], data[:,4], ~status, data[:,5])).T
        etdata.load(data, **{'source': 'np_array'})

        etdata.data['x'] = (etdata.data['x'] - geom['display_width_pix']/2) / px2deg
        etdata.data['y'] = (etdata.data['y'] - geom['display_height_pix']/2) / px2deg
        etdata.data['x'][status] = np.nan
        etdata.data['y'][status] = np.nan

        #fix
        if 'UH29_img_Europe_labelled_MN' in fname:
            etdata.data['evt'][3197:3272] = 1

        #set status one more time
        status = np.isnan(etdata.data['x']) | np.isnan(etdata.data['y']) |\
                 ~np.in1d(etdata.data['evt'], args.events) | ~etdata.data['status']
        etdata.data['status'] = ~status

        etdata.save('%s/%s' % (ddir, fname))

FILES = glob.glob('%s/*.npy' % ddir)
#for replication use following code
with open('datalist', 'r') as f:
    FILES = ['%s/%s'%(ddir, _fname.strip()) for _fname in f.readlines()]

#%%split based on trial
print ("Train/test split")

exp = [(fpath,) + tuple(os.path.split(os.path.splitext(fpath)[0])[-1].split('_labelled_')) +\
       (os.path.split(os.path.splitext(fpath)[0])[-1].split('_')[0][:2], )+\
       (os.path.split(os.path.splitext(fpath)[0])[-1].split('_')[2], ) for fpath in FILES]
exp_df = pd.DataFrame(exp, columns=['fpath', 'flabel', 'coder', 'sub', 'img'])
exp_gr = exp_df.groupby('flabel')
exp_df['pair'] = False
for _e, _d in exp_gr:
    if len(_d) >1:
        exp_df.loc[_d.index, 'pair'] = True

print ('Number of trials: %d' %len(exp_df['flabel'].unique()))
print ('Number of subjects: %d' %len(exp_df['sub'].unique()))
print ('Number of images: %d' %len(exp_df['img'].unique()))

#split on pairs
exp_gr_pair = exp_df.groupby('pair')
X_unpaired, X_paired = [_d for _, _d in exp_gr_pair]


#%%load data
print ("Cleaning data")
data = OrderedDict()
data_lens = []
#iterates through data and marks status as false for events other than [1, 2, 3]
for df, part in zip([X_unpaired, X_paired, X_paired],
                    [['unpaired'], ['paired', 'RA'], ['paired', 'MN']]):
    if len(part)==1:
        part = part[0]
        data[part] = []
        for n, d in df.iterrows():
            _data = np.load(d['fpath'])
            data_lens.append(len(_data))
            mask = np.in1d(_data['evt'], args.events)
            _data['status'][~mask] = False
            data[part].append((d, _data))
    else:
        coder = part[-1]
        part = part[-1]
        data[part] = []
        for n, d in df.loc[df['coder']==coder,:].iterrows():
            _data = np.load(d['fpath'])
            data_lens.append(len(_data))
            mask = np.in1d(_data['evt'], args.events)
            _data['status'][~mask] = False
            data[part].append((d, _data))


#sort according to flabel
labels_mn = pd.DataFrame([l['flabel'] for l, _ in data['MN']])
labels_ra = pd.DataFrame([l['flabel'] for l, _ in data['RA']])
inds = [np.where(labels_ra[0].values==_mn)[0][0] for _mn in labels_mn[0].values]
data['RA']=[data['RA'][_i] for _i in inds]

#PAIR CLEAN: retain only samples, where both coders tag fix, sacc or pso
pair_clean = False
###comment to estimate accurate event percentages of lund2013 dataset
for (pair1, data1), (pair2, data2) in zip(data['MN'], data['RA']):
    #sanity check
    assert (pair1['flabel']==pair2['flabel']) & (pair1['fpath']!=pair2['fpath'])
    mask = data1['status'] & data2['status']

    data1['status'][~mask] = False
    data2['status'][~mask] = False
    data1['evt'][~mask] = 0
    data2['evt'][~mask] = 0
    pair_clean = True
###

data['paired'] = data['MN']+data['RA']
data['all'] = data['unpaired']+data['paired']
labels_all, data_all = zip(*data['all'])
labels_all = pd.DataFrame(list(labels_all))

#train/test split by trial
paired_info = pd.DataFrame([_i for _i, _d in data['paired']])
exp_gr_flabel = list(paired_info.groupby('flabel'))
labels_val, labels_test = train_test_split(exp_gr_flabel, train_size=0.25, random_state=220617)
labels_val = pd.concat([_d for _, _d in labels_val])
labels_test = pd.concat([_d for _, _d in labels_test])

mask = labels_all.isin(labels_val)
data['val'] = [(_i, _d) for _m, (_i, _d) in zip(mask['sub'].values, data['all']) if _m]
data['val_MN'] = [(_i, _d) for _m, (_i, _d) in zip(mask['coder'].values, data['all']) if _m & (_i['coder']=='MN')]
data['val_RA'] = [(_i, _d) for _m, (_i, _d) in zip(mask['coder'].values, data['all']) if _m & (_i['coder']=='RA')]
mask = labels_all.isin(labels_test)
data['test'] = [(_i, _d) for _m, (_i, _d) in zip(mask['sub'].values, data['all']) if _m]
data['test_MN'] = [(_i, _d) for _m, (_i, _d) in zip(mask['coder'].values, data['all']) if _m & (_i['coder']=='MN')]
data['test_RA'] = [(_i, _d) for _m, (_i, _d) in zip(mask['coder'].values, data['all']) if _m & (_i['coder']=='RA')]


#clean data; #splits data by removing dataloss
for part in data.keys():
    data['%s_clean'%part] = []
    for trid, (i, d) in enumerate(data[part]): #iterates over files
        dd = np.split(d, np.where(np.diff(d['status'].astype(np.int0)) != 0)[0]+1)
        dd = [(i, _d) for _d in dd if _d['status'].all()]
        data['%s_clean'%part].extend(dd)


#%% augment
print ("Augmenting train data")
data_augment = dict()
for part in ['unpaired_clean']:
    data['%s_augment'%part] = augment_with_saccades(data[part], args.seq_len)


#%% save
print ("Saving")
data_export = [
    'unpaired_clean_augment',
    'unpaired_clean',
    'val_RA_clean',
    'val_MN_clean',
    'val_clean',
    'test_RA_clean',
    'test_MN_clean',
]

sdir = '%s/gazeNet_data'%args.root
mkpath(sdir)
for _de in data_export:
    with open('%s/data.%s.pkl'% (sdir, _de), 'wb') as f:
        pickle.dump(data[_de], f)


sys.exit()



