r"""Data source management."""

import os
import torch
import numpy as np
import pandas as pd
import random
from scipy import signal

from scipy.stats import zscore
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

Hz = 128 // 2
SEG_SEC = 30
IN_CHAN = 1
NUM_CLASSES = 2     # number of sleep stages (2, 3, 4)

FILE_TYPE = [
    'Pleth.txt',    # PPG signal
    'ECG1.txt',
    'Hypno.txt'     # Sleep labels
]

SUBJECTS = [
    'AHI11.2-1',
    'AHI16.2-1',
    'AHI17.3-1',
    'AHI18.4-1',
    'AHI19.2-1',
    'AHI20.8-1',
    'AHI22.1-1',
    'AHI28.3-1',
    'AHI52.5-1',
    'AHI68.7-1',
]

r"""Map categorical sleep stages to numerical."""
if NUM_CLASSES == 4:
    LABELS_MAP = {
        'W': 0,
        '1': 1,
        '2': 1,
        '3': 2,
        '4': 2,
        'R': 3,
    }
elif NUM_CLASSES == 3:
    LABELS_MAP = {
        'W': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        'R': 2,
    }
elif NUM_CLASSES == 2:
    LABELS_MAP = {
        'W': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        'R': 1,
    }


def get_classes(
    input_directory
):
    r"""Retrieve sleep-scoring categories/labels."""
    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if f.endswith(FILE_TYPE[-1]):
            header_files.append(g)

    classes = set()
    for filename in header_files:
        with open(filename, 'r') as f:
            for lbl in f:
                classes.add(lbl.strip().upper())
    return classes


def load_data(
    header_file
):
    r"""Read PPG signal for given annotation file."""
    with open(header_file, 'r') as f:
        header = f.readlines()
    csv_file = header_file.replace('Hypno', 'Pleth')
    recording = pd.read_csv(csv_file).values

    r"Resample from 128 to 64Hz."
    print(f"load-data sz:{recording.shape}, len:{len(recording)}")
    recording = signal.resample(recording, len(recording)//2)
    print(f"load-data resamp sz:{recording.shape}, len:{len(recording)}")
    return recording, header


class PPGDataset(Dataset):
    r"""ECG dataset class."""

    def __init__(
        self, input_directory, hz, seg_sec=30, n_chan=1,
        seg_slide_sec=30, n_skip_segments=0, data_augment=False, log=print
    ):
        r"""Instantiate EcgDataset."""
        self.input_directory = input_directory
        self.data_augment = data_augment
        self.log = log
        self.record_names = []
        self.record_wise_segments = {}
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_sz = self.seg_sec * self.hz
        self.n_chan = n_chan
        self.seg_slide_sec = seg_slide_sec
        self.segments = []
        self.seg_labels = []
        self.n_skip_segments = n_skip_segments

        self.initialise()

        '''Indexes of data, another level of abstraction. Flexible to shuffle.'''
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def initialise(
        self
    ):
        r"""Initialise dataset."""
        self.header_files = []
        for f in os.listdir(self.input_directory):
            g = os.path.join(self.input_directory, f)
            if f.endswith(FILE_TYPE[-1])\
                    and sum([1 for i in range(len(SUBJECTS)) if f.find(SUBJECTS[i]) > -1]) > 0:
                '''EEG file'''
                self.header_files.append(g)
        self.debug(
            f"Input: {self.input_directory}, "
            f"header files: {len(self.header_files)}, seg_size:{self.seg_sz}")

        classes = get_classes(self.input_directory)
        self.debug(f'{len(classes)} found, [{classes}]')

        self.debug('Loading data...')

        records = []
        labels = []

        for i in range(len(self.header_files)):
            recording, header = load_data(self.header_files[i])
            records.append(recording)

            r"Assign index to labels"
            labels_act = list()
            for lbl in header:
                labels_act.append(LABELS_MAP.get(lbl.strip()))
            labels.append(labels_act)

        self.debug(f'Segmenting {len(records)} records ...')

        for i_rec in range(len(records)):
            rec = records[i_rec]

            n_samples = rec.shape[0]
            n_windows = n_samples // (self.hz * self.seg_slide_sec)
            if self.header_files is not None:
                r"Extract recordname from header file path."
                h = self.header_files[i_rec]
                self.record_names.append(
                    (lambda x: x[:x.rindex('.')])(h.split('/')[-1]))
                r"Initialise record-wise segment index storage."
                self.record_wise_segments[self.record_names[-1]] = []
                self.debug(
                    f"[{h}] Calculating... {n_windows} segments out of "
                    f"{n_samples} samples, {len(labels[i_rec])} labels."
                )

            i_rec_seg_start = len(self.segments)
            for i_window in range(self.n_skip_segments, n_windows):
                start = i_window * self.hz * self.seg_slide_sec
                seg_ = rec[start:start+self.seg_sz, :self.n_chan]

                r"Sec-by-sec norm to reduce outlier effect."
                seg_z_ = []
                for i_sec_ in range(0, self.seg_sec):
                    sec_seg_ = seg_[i_sec_*self.hz:(i_sec_+1)*self.hz]
                    seg_z_.extend(zscore(sec_seg_))

                self.segments.append(
                    np.column_stack(
                        np.array(seg_z_).T
                    )
                )
                self.seg_labels.append(labels[i_rec][i_window])
                '''store segment index to record-wise store'''
                self.record_wise_segments[self.record_names[-1]].append(
                    len(self.segments)-1
                )

                r"Data Augmentation."
                if self.data_augment and i_window > self.n_skip_segments:
                    r"Generate new segments based on current and previous \
                    segments."
                    if NUM_CLASSES == 2:
                        r"2-class, already enough samples. Take only previous\
                        segment has the same label."
                        if self.seg_labels[-1] == self.seg_labels[-2]:
                            self.seg_augment(
                                i_prev_seg_start_sec=20, step_sec=1,
                                i_prev_seg_stop_sec=29
                            )

                    else:
                        if self.seg_labels[-1] == self.seg_labels[-2]:
                            self.seg_augment(
                                i_prev_seg_start_sec=5, step_sec=2,
                                i_prev_seg_stop_sec=30
                            )
                        else:
                            r"Prev segment label different, start taking from 75%."
                            self.seg_augment(
                                i_prev_seg_start_sec=23, step_sec=2,
                                i_prev_seg_stop_sec=30
                            )
                pass
            self.debug(
                f"... [{self.record_names[i_rec]}] orig window:{n_windows}, "
                f"new created: {len(self.segments)-i_rec_seg_start}, "
                f"labels: {len(self.seg_labels)-i_rec_seg_start}, "
                f"class-dist: {np.unique(self.seg_labels[i_rec_seg_start:], return_counts=True)}, "
                f"seg_shape:{self.segments[-1].shape}"
            )
        self.debug(f'Segmentation done, total {len(self.segments)}.')

    def seg_augment(
        self, i_prev_seg_start_sec=20, step_sec=1, i_prev_seg_stop_sec=1
    ):
        r"""Augment segments using cropped-overlapping strategy.

        Spawn segment from previous and current segment with specified second
        slide.
        """
        prev_seg = self.segments[-2]
        cur_seg = self.segments[-1]
        cur_label = self.seg_labels[-1]
        i_cur_seg_stop_sec = self.seg_sec - \
            (self.seg_sec - i_prev_seg_start_sec)
        count_new_seg = 0
        for i_prev_start_sec in range(i_prev_seg_start_sec, i_prev_seg_stop_sec, step_sec):
            count_new_seg += 1
            r"Form new seg = 0.5*prev_seg + 0.5*cur_seg"
            new_seg = []
            new_seg = np.concatenate(
                (
                    prev_seg[i_prev_start_sec*self.hz:, :],
                    cur_seg[:i_cur_seg_stop_sec*self.hz, :]
                ), axis=0
            )

            assert new_seg.shape[0] == self.seg_sz

            r"Right shift prev and cur segment index by sampling hz."
            i_cur_seg_stop_sec += step_sec

            r"Add to segment and label global list."
            self.segments.insert(-2, new_seg)
            self.seg_labels.insert(-2, cur_label)

            r"store segment index to record-wise store."
            self.record_wise_segments[self.record_names[-1]].append(
                len(self.segments)-1
            )

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return None, None

    def debug(self, msg):
        self.log(msg)


class PartialDataset(PPGDataset):
    r"""Generate dataset from a parent dataset and indexes."""

    def __init__(
        self, dataset, seg_index, test=False, shuffle=False,
        balance_labels=False
    ):
        r"""Instantiate dataset from parent dataset and indexes."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        if balance_labels:
            self.__balance_labels__()

    def __balance_labels__(self):
        r"""Undersample majority class samples to balance dataset."""
        data_idx, data_dist = np.unique(
            [self.memory_ds.seg_labels[i] for i in self.indexes], return_counts=True
        )

        # np.unique returns tuple of two arrays i. index-array, and ii. data-array.
        self.memory_ds.debug(f"partial-ds data-dist:{data_idx}, {data_dist}, ")
        min_lbl, min_lbl_freq = np.argmin(
            data_dist), data_dist[np.argmin(data_dist)]
        self.memory_ds.debug(f"min_lbl/freq: {min_lbl}/{min_lbl_freq}")
        new_indexes = []
        for i_cur_data in data_idx:
            if i_cur_data == min_lbl:
                continue
            n_reduce = data_dist[i_cur_data] - min_lbl_freq

            label_idx_idx = [
                i for i in range(len(self.indexes))
                if self.memory_ds.seg_labels[self.indexes[i]] == i_cur_data]
            self.memory_ds.debug(
                f"Label:{i_cur_data}, reduce: {n_reduce} from {len(label_idx_idx)}"
            )
            for _ in range(n_reduce):
                i_del = random.randint(0, len(label_idx_idx)-1)
                label_idx_idx.pop(i_del)
            new_indexes.extend([self.indexes[i] for i in label_idx_idx])
            self.memory_ds.debug(
                f"Label:{i_cur_data}, reduced to {len(label_idx_idx)}, "
                f"detail:{new_indexes[len(new_indexes)-10:]}"
            )
        r"Replace self.indexes with new_indexes."
        self.indexes = new_indexes[:]

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = trainY
        return X_tensor, Y_tensor

    def distort_seg(self, seg):
        r"""Add noise to segment."""
        noise = np.random.normal(0, 1, seg.shape[-1]).reshape(1, -1)
        seg += noise

    def debug(self, msg):
        r"""Output log message."""
        self.memory_ds.debug(msg)
