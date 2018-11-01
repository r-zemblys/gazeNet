#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rz
@email: r.zemblys@tf.su.lt
"""
import itertools

import numpy as np
import pandas as pd
import scipy.signal as sg

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rc("axes.spines", top=False, right=False)
plt.ion()
#import seaborn as sns

#TODO: fix importing
from utils import round_up_to_odd, rolling_window


def get_px2deg(geom):
    """Calculates pix2deg values, based on simple geometry.
    Parameters:
        geom    --  dictionary with following parameters of setup geometry:
                    screen_width
                    screen_height
                    eye_distance
                    display_width_pix
                    display_height_pix
    Returns:
        px2deg  --  pixels per degree value
    """
    px2deg = np.mean(
        (1/
         (np.degrees(2*np.arctan(geom['screen_width']/
                    (2*geom['eye_distance'])))/
         geom['display_width_pix']),
         1/
         (np.degrees(2*np.arctan(geom['screen_height']/
                    (2*geom['eye_distance'])))/
         geom['display_height_pix']))
    )
    return px2deg

def aggr_events(events_raw):
    """Aggregates event vector to the list of compact event vectors.
    Parameters:
        events_raw  --  vector of raw events
    Returns:
        events_aggr --  list of compact event vectors ([onset, offset, event])
    """

    events_aggr = []
    s = 0
    for bit, group in itertools.groupby(events_raw):
        event_length = len(list(group))
        e = s+event_length
        events_aggr.append([s, e, bit])
        s = e
    return events_aggr

def calc_event_data(etdata, evt,
                    w = {255:1,
                         0: 1,
                         1: 50,
                         2: 1,
                         3: 1,
                         4: 1,
                         5: 1,
                         6: 1,
                         'vel': 18,
                         'etdq': 200}, ):
    """Calculates event parameters.
    Parameters:
        etdata  --  an instance of ETData
        evt     --  compact event vector
        w       --  dictionary of context to take into account
                    for each event type; in ms
    Returns:
        posx_s      --  onset position, horizontal
        posx_e      --  offset position, horizontal
        posy_s      --  onset position, vertical
        posy_e      --  offset position, vertical
        posx_mean   --  mean postion, horizontal
        posy_mean   --  mean postion, vertical
        posx_med    --  median postion, horizontal
        posy_med    --  median postion, vertical
        pv          --  peak velocity
        pv_index    --  index for peak velocity
        rms         --  precision, 2D rms
        std         --  precision, 2D std
    """

    #init params
    data = etdata.data
    fs = etdata.fs
    e = {k:v for k, v in zip(['s', 'e', 'evt'], evt)}
    ws = w[e['evt']]
    ws = 1 if not(ws > 1) else  round_up_to_odd(ws/1000.0*fs, min_val=3)
    ws_vel = round_up_to_odd(w['vel']/1000.0*fs, min_val=3)
    w_etdq = int(w['etdq']/1000.*fs)

    #calculate velocity using Savitzky-Golay filter
    vel = np.hypot(sg.savgol_filter(data['x'], ws_vel, 2, 1),
                   sg.savgol_filter(data['y'], ws_vel, 2, 1))*fs

    ind_s = e['s']+ws
    ind_s = ind_s if ind_s < e['e'] else e['e']
    ind_e = e['e']-ws
    ind_e = ind_e if ind_e > e['s'] else e['s']

    posx_s = np.nanmean(data[e['s']:ind_s]['x'])
    posy_s = np.nanmean(data[e['s']:ind_s]['y'])
    posx_e = np.nanmean(data[ind_e:e['e']]['x'])
    posy_e = np.nanmean(data[ind_e:e['e']]['y'])

    posx_mean = np.nanmean(data[e['s']:e['e']]['x'])
    posy_mean = np.nanmean(data[e['s']:e['e']]['y'])
    posx_med = np.nanmedian(data[e['s']:e['e']]['x'])
    posy_med = np.nanmedian(data[e['s']:e['e']]['y'])

    pv = np.max(vel[e['s']:e['e']])
    pv_index = e['s']+ np.argmax(vel[e['s']:e['e']])

    if e['e']-e['s']>w_etdq:
        x_ = rolling_window(data[e['s']:e['e']]['x'], w_etdq)
        y_ = rolling_window(data[e['s']:e['e']]['y'], w_etdq)

        std = np.median(np.hypot(np.std(x_, axis=1), np.std(y_, axis=1)))
        rms = np.median(np.hypot(np.sqrt(np.mean(np.diff(x_)**2, axis=1)),
                                 np.sqrt(np.mean(np.diff(y_)**2, axis=1))))
    else:
        std = 0
        rms = 0

    return posx_s, posx_e, posy_s, posy_e, posx_mean, posy_mean, posx_med, posy_med, pv, pv_index, rms, std

class ETData():
    #Data types and constants
    dtype = np.dtype([
        ('t', np.float64),
        ('x', np.float32),
        ('y', np.float32),
        ('status', np.bool),
        ('evt', np.uint8)
    ])
    evt_color_map = dict({
        0: 'gray',  #0. Undefined
        1: 'b',     #1. Fixation
        2: 'r',     #2. Saccade
        3: 'y',     #3. Post-saccadic oscillation
        4: 'm',     #4. Smooth pursuit
        5: 'k',     #5. Blink
        9: 'k',     #9. Other
    })

    def __init__(self):
        self.data = np.array([], dtype=ETData.dtype)
        self.fs = None
        self.evt = None

    def load(self, fpath, **kwargs):
        """Loads data.
        Parameters:
            fpath   --  file path
            kwargs:
                'source'. Available values:
                          'etdata'    --  numpy array with ETData.dtype
                          function    --  function, which parses custom
                                          data format and returns numpy array,
                                          which can be converted to have data
                                          type of ETData.dtype
        """

        if not(kwargs.has_key('source')):
            try:
                self.data = np.load(fpath)
            except:
                print("ERROR loading %s" % fpath)
        else:
            if kwargs['source']=='etdata':
                self.data = np.load(fpath)

            if kwargs['source']=='array':
                if not fpath.dtype == ETData.dtype:
                    print "Error. Data types do not match"
                    return False
                self.data = fpath

            if kwargs['source']=='np_array':
                self.data = np.core.records.fromarrays(fpath.T,
                                                       dtype=ETData.dtype)

            if callable(kwargs['source']):
                self.data = kwargs['source'](fpath, ETData.dtype)

        #estimate sampling rate
        self.fs = float(self.find_nearest_fs(self.data['t']))
        self.evt = None
        return self.data

    def save(self, spath):
        """Saves data as numpy array with ETData.dtype data type.
        Parameters:
            spath   --  save path
        """
        np.save(spath, self.data)

    def find_nearest_fs(self, t):
        """Estimates data sampling frequency.
        Parameters:
            t   --  timestamp vector
        Returns:
            Estimated sampling frequency
        """
        fs = np.array([2000, 1250, 1000, 600, 500,  #high end
                       300, 250, 240, 200,          #middle end
                       120, 75, 60, 50, 30, 25])    #low end
        ##debug
        #if (np.diff(t) == 0).any():
        #    stop
        t = np.median(1/np.diff(t))
        return fs.flat[np.abs(fs - t).argmin()]

    def calc_evt(self, fast=False):
        '''Calculated event data
        '''
        evt_compact = aggr_events(self.data['evt'])
        evt = pd.DataFrame(evt_compact,
                           columns = ['s', 'e', 'evt'])
        evt['dur_s'] = np.diff(evt[['s', 'e']], axis=1).squeeze()
        evt['dur'] = evt['dur_s']/self.fs

        if not(fast):
            evt['posx_s'], evt['posx_e'], evt['posy_s'], evt['posy_e'],\
            evt['posx_mean'], evt['posy_mean'], evt['posx_med'], evt['posy_med'],\
            evt['pv'], evt['pv_index'], evt['rms'], evt['std']   = \
               zip(*map(lambda x: calc_event_data(self, x), evt_compact))
            evt['ampl_x'] = np.diff(evt[['posx_s', 'posx_e']])
            evt['ampl_y'] = np.diff(evt[['posy_s', 'posy_e']])
            evt['ampl'] = np.hypot(evt['ampl_x'], evt['ampl_y'])
        #TODO:
        #   calculate fix-to-fix saccade amplitude
        self.evt = evt
        return self.evt

    def plot(self, spath = None, save=False, show=True, title=None):
        '''Plots trial
        '''
        if show:
            plt.ion()
        else:
            plt.ioff()

        fig = plt.figure(figsize=(10,6))
        ax00 = plt.subplot2grid((2, 2), (0, 0))
        ax10 = plt.subplot2grid((2, 2), (1, 0), sharex=ax00)
        ax01 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        ax00.plot(self.data['t'], self.data['x'], '-')
        ax10.plot(self.data['t'], self.data['y'], '-')
        ax01.plot(self.data['x'], self.data['y'], '-')
        for e, c in ETData.evt_color_map.iteritems():
            mask = self.data['evt'] == e
            ax00.plot(self.data['t'][mask], self.data['x'][mask], '.', color = c)
            ax10.plot(self.data['t'][mask], self.data['y'][mask], '.', color = c)
            ax01.plot(self.data['x'][mask], self.data['y'][mask], '.', color = c)

        etdata_extent = np.nanmax([np.abs(self.data['x']), np.abs(self.data['y'])])+1

        ax00.axis([self.data['t'].min(), self.data['t'].max(), -etdata_extent, etdata_extent])
        ax10.axis([self.data['t'].min(), self.data['t'].max(), -etdata_extent, etdata_extent])
        ax01.axis([-etdata_extent, etdata_extent, -etdata_extent, etdata_extent])

#        sns.despine()
        if title is not None:
            plt.suptitle(title)
        plt.tight_layout()

        if save and not(spath is None):
            plt.savefig('%s.png' % (spath))
            plt.close()
