import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from .utils import BoxMuller_gaussian

def load_npy_files(glob_mask, config):
    FILES = glob.glob(glob_mask)
    data_list = []
    for _fpath in FILES:
        _data = np.load(_fpath)
        _mask = np.in1d(_data['evt'], config['events'])
        _data['status'][~_mask] = False

        data_list.append(_data)
    return data_list

class EventParser(object):
    def __init__(self, config):
        """

        """
        super(EventParser, self).__init__()
        self.config = config


    def parse_data(self, sample):
        config = self.config
        augment = config['augment']
        rms_noise_levels = np.arange(*config["augment_noise"])

        inpt_dir = ['x', 'y']

        gaze_x = np.copy(sample[inpt_dir[0]])
        gaze_y = np.copy(sample[inpt_dir[1]])

        if augment:
            u1, u2 = np.random.uniform(0,1, (2, len(sample)))
            noise_x, noise_y = BoxMuller_gaussian(u1,u2)
            rms_noise_level = np.random.choice(rms_noise_levels)
            noise_x*=rms_noise_level/2
            noise_y*=rms_noise_level/2
            #rms = np.sqrt(np.mean(np.hypot(np.diff(noise_x), np.diff(noise_y))**2))
            gaze_x+=noise_x
            gaze_y+=noise_y

        inpt_x, inpt_y = [np.diff(gaze_x),
                          np.diff(gaze_y)]

        X = [(_coords) for _coords in zip(inpt_x, inpt_y)]
        X = np.array(X, dtype=np.float32)

        return X


class EMDataset(Dataset, EventParser):
    def __init__(self, config, gaze_data):
        """
        Dataset that loads tensors
        """

        split_seqs = config['split_seqs']
        #mode = config['mode']

        #input is in fact diff(input), therefore we want +1 sample
        seq_len = config['seq_len']+1
        #seq_step = seq_len/2 if mode == 'train' else seq_len
        seq_step = seq_len

        data = []
        #seqid = -1
        for d in gaze_data: #iterates over files
            dd = np.split(d, np.where(np.diff(d['status'].astype(np.int0)) != 0)[0]+1)
            dd = [_d for _d in dd if (_d['status'].all() and not(len(_d) < seq_len))]

            for seq in dd: #iterates over chunks of valid data
                #seqid +=1
                if split_seqs and not(len(seq) < seq_len):
                    #this
                    #1. overlaps the last piece of data and
                    #2. allows for overlaping sequences in general; not tested
                    seqs = [seq[pos:pos + seq_len] if (pos + seq_len) < len(seq) else
                            seq[len(seq)-seq_len:len(seq)] for pos in range(0, len(seq), seq_step)]
                else:
                    seqs = [seq]

                data.extend(seqs)

        self.data = data
        self.size = len(data)
        self.config = config

        super(EMDataset, self).__init__(config)

    def __getitem__(self, index):
        sample = self.data[index]
        gaze_data = self.parse_data(sample)
        evt = self.parse_evt(sample['evt'])

        return torch.FloatTensor(gaze_data.T), evt, ()

    def parse_evt(self, evt):
        return evt[1:]-1

    def __len__(self):
        return self.size

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    #return batch
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = [] #torch.IntTensor(minibatch_size, 3)
    targets = []

    for x in range(minibatch_size):
        sample = batch[x]

        tensor, target, (_) = sample
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes.append(len(target))
        targets.extend(target.tolist())
    targets = torch.LongTensor(targets)
    return inputs, targets, input_percentages, target_sizes, (_)



class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        torch.manual_seed(220617)
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)

class GazeDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader
        """
        seed = kwargs.pop('seed', 220617)
        super(GazeDataLoader, self).__init__(*args, **kwargs)
        np.random.seed(seed)
        self.collate_fn = _collate_fn
        #self.sampler = RandomSampler(*args)
