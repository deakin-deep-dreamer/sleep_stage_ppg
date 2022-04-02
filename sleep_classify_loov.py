import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse
import random
from scipy import signal
import logging
import traceback

from scipy.stats import zscore
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# from models import model_zoo_classif as model_zoo_classif
from models import model_zoo_classif2 as model_zoo_classif
from lib import model_training

logger = logging.getLogger(__name__)

FILE_TYPE = ["Pleth.txt", "ECG1.txt", "Hypno.txt"]

SUBJECTS = [
    "AHI11.2-1",
    "AHI16.2-1",
    "AHI17.3-1",
    "AHI18.4-1",
    "AHI19.2-1",
    "AHI20.8-1",
    "AHI22.1-1",
    "AHI28.3-1",
    "AHI52.5-1",
    "AHI68.7-1",
]


def log(msg):
    r"""Log message."""
    logger.debug(msg)


def config_logger():
    r"""Config logger."""
    global logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    logger.addHandler(ch)
    # logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    # logger = logging.getLogger(__name__)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"{LOG_PATH}/{LOG_FILE}")
    fh.setFormatter(format)
    fh.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)


def get_classes(input_directory):
    header_files = []
    for f in os.listdir(input_directory):
        g = os.path.join(input_directory, f)
        if f.endswith(FILE_TYPE[-1]):
            header_files.append(g)

    classes = set()
    for filename in header_files:
        with open(filename, "r") as f:
            for lbl in f:
                classes.add(lbl.strip().upper())
    return classes


def load_data(header_file):
    with open(header_file, "r") as f:
        header = f.readlines()
    csv_file = header_file.replace("Hypno", "Pleth")
    recording = pd.read_csv(csv_file).values

    r"Resample from 128 to 64Hz."
    print(f"load-data sz:{recording.shape}, len:{len(recording)}")
    recording = signal.resample(recording, len(recording) // 2)
    print(f"load-data resamp sz:{recording.shape}, len:{len(recording)}")
    return recording, header


def create_model(conv_kernel=15):
    r"""Create model."""
    _low_conv_cfg = [32, 32, "M"]
    _low_conv_kernels = [21, 21]
    _low_conv_strides = [5, 1]
    _low_conv_pooling_kernels = [2]
    for ef in range(SEG_LARGE_FACTOR - 2):
        _low_conv_cfg.extend([32, 32, "M"])
        _low_conv_kernels.extend([21, 21])
        _low_conv_strides.extend([1, 1])
        _low_conv_pooling_kernels.extend([2])

    model = model_zoo_classif.DenseNet(
        input_size=Hz * SEG_SEC * SEG_LARGE_FACTOR,
        in_channels=IN_CHAN,
        n_classes=NUM_CLASSES,
        log=log,
        # add_unit_kernel=False,
        skip_final_transition_blk=True,
        kernels=[[5] for _ in range(N_DENSE_BLOCK)],
        layer_dilations=[
            # [1],
            # [1],
            # [1],
            # [1],
            # [1],
            # [1]
            [1]
            for _ in range(N_DENSE_BLOCK)
        ],
        channel_per_kernel=[
            32
            for _ in range(N_DENSE_BLOCK)
            # 64, 128, 256
        ],
        n_blocks=[2 for _ in range(N_DENSE_BLOCK)],
        # low_conv_cfg=[32, 32, 'M'], low_conv_kernels=[21, 21],
        # low_conv_strides=[5, 1], low_conv_pooling_kernels=[2, 2],
        low_conv_cfg=_low_conv_cfg,
        low_conv_kernels=_low_conv_kernels,
        low_conv_strides=_low_conv_strides,
        low_conv_pooling_kernels=_low_conv_pooling_kernels,
        transition_pooling=[2 for _ in range(N_DENSE_BLOCK)],
    )
    return model


class EcgDataset(Dataset):
    r"""ECG dataset class."""

    def __init__(
        self,
        input_directory,
        hz,
        seg_sec=30,
        n_chan=1,
        seg_slide_sec=30,
        n_skip_segments=0,
        data_augment=False,
        data_aug_cfg=None,
        log=log,
    ):
        r"""Instantiate EcgDataset."""
        self.input_directory = input_directory
        self.data_augment = data_augment
        self.data_aug_cfg = data_aug_cfg
        self.log = log
        # self.records = records
        # self.labels = labels
        # self.header_files = header_files
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

        """Indexes of data, another level of abstraction. Flexible to shuffle."""
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def initialise(self):
        r"""Initialise dataset."""
        self.header_files = []
        for f in os.listdir(self.input_directory):
            g = os.path.join(self.input_directory, f)
            if (
                f.endswith(FILE_TYPE[-1])
                and sum([1 for i in range(len(SUBJECTS)) if f.find(SUBJECTS[i]) > -1])
                > 0
            ):
                """PPG file"""
                self.header_files.append(g)
        self.debug(
            f"Input: {self.input_directory}, "
            f"header files: {len(self.header_files)}, seg_size:{self.seg_sz}"
        )

        classes = get_classes(self.input_directory)
        self.debug(f"{len(classes)} found, [{classes}]")

        self.debug("Loading data...")

        records = []
        labels = []

        for i in range(len(self.header_files)):
            recording, header = load_data(self.header_files[i])
            records.append(recording)

            """assign index to labels"""
            labels_act = list()
            # labels_act = np.zeros(NUM_CLASSES)
            for lbl in header:
                labels_act.append(LABELS_MAP.get(lbl.strip()))
                # labels_act[LABELS_MAP.get(l.strip())] = 1
            labels.append(labels_act)

        self.debug(f"Segmenting {len(records)} records ...")

        for i_rec in range(len(records)):
            rec = records[i_rec]

            # r"Bandpass filter."
            # rec = bandpass(
            #     rec, hz=self.hz
            # )

            r"Derivative of signal. Flatten np.ndarray before pass to fn."
            # rec_diff = derivative(rec[:, :self.n_chan].reshape(-1))

            """Normalise"""
            # rec = zscore(rec, axis=0)

            n_samples = rec.shape[0]
            n_windows = n_samples // (self.hz * self.seg_slide_sec)
            if self.header_files is not None:
                """Extract recordname from header file path"""
                h = self.header_files[i_rec]
                self.record_names.append(
                    (lambda x: x[: x.rindex(".")])(h.split("/")[-1])
                )
                """initialise record-wise segment index storage"""
                self.record_wise_segments[self.record_names[-1]] = []
                self.debug(
                    f"[{h}] Calculating... {n_windows} segments out of {n_samples} samples, {len(labels[i_rec])} labels."
                )

            i_rec_seg_start = len(self.segments)
            for i_window in range(self.n_skip_segments, n_windows):
                start = i_window * self.hz * self.seg_slide_sec
                seg_ = rec[start : start + self.seg_sz, : self.n_chan]

                # print(f"seg_ : {seg_.shape}")

                r"Convert np ndarray to array."
                # seg_diff_ = rec_diff[start:start + self.seg_sz]
                # seg_ = self.bandpass(seg_)

                r"Sec-by-sec norm to reduce outlier effect."
                seg_z_ = []
                seg_sub_mean_ = []
                for i_sec_ in range(0, self.seg_sec):
                    sec_seg_ = seg_[i_sec_ * self.hz : (i_sec_ + 1) * self.hz]
                    seg_z_.extend(zscore(sec_seg_))
                    # seg_sub_mean_.extend(
                    #     np.array(sec_seg_) - np.mean(sec_seg_))

                self.segments.append(
                    # np.column_stack(
                    #     [zscore(seg_sub_mean_), zscore(seg_diff_)]
                    # )
                    np.column_stack(
                        # zscore(seg_sub_mean_).T
                        np.array(seg_z_).T
                    )
                )
                # print(f"seg_z_ : {len(seg_z_)}, segment: {self.segments[-1].shape}")

                self.seg_labels.append(labels[i_rec][i_window])
                """store segment index to record-wise store"""
                self.record_wise_segments[self.record_names[-1]].append(
                    len(self.segments) - 1
                )

                r"Data Augmentation."
                if self.data_augment and i_window > self.n_skip_segments:
                    r"Generate new segments based on current and previous \
                    segments."
                    if NUM_CLASSES == 2:
                        r"2-class, already enough samples. Take only previous\
                        segment has the same label."
                        if self.seg_labels[-1] == self.seg_labels[-2]:
                            aug_start_sec, aug_stop_sec, aug_step = 10, 25, 1
                            if self.data_aug_cfg == 1:
                                # minimal aug
                                # aug_start_sec, aug_stop_sec, aug_step = 13, 20, 2
                                aug_step = 2

                            self.seg_augment(
                                i_prev_seg_start_sec=aug_start_sec,
                                step_sec=aug_step,
                                i_prev_seg_stop_sec=aug_stop_sec,
                            )

                    else:
                        if self.seg_labels[-1] == self.seg_labels[-2]:
                            aug_start_sec, aug_stop_sec, aug_step = 10, 29, 2
                            if self.data_aug_cfg == 1:
                                # minimal aug
                                aug_step = 4
                            self.seg_augment(
                                i_prev_seg_start_sec=aug_start_sec,
                                step_sec=aug_step,
                                i_prev_seg_stop_sec=aug_stop_sec,
                            )
                        else:
                            r"Prev segment label different, start taking from 75%."
                            aug_start_sec, aug_stop_sec, aug_step = 23, 29, 2
                            if self.data_aug_cfg == 1:
                                # minimal aug
                                aug_step = 4
                            self.seg_augment(
                                i_prev_seg_start_sec=aug_start_sec,
                                step_sec=aug_step,
                                i_prev_seg_stop_sec=aug_stop_sec,
                            )

            if SEG_LARGE_FACTOR > 1:
                # self.resegment(i_rec_seg_start)
                self.resegment_historical(i_rec_seg_start)

            self.debug(
                f"... [{self.record_names[i_rec]}] orig window:{n_windows}, "
                f"new created: {len(self.segments)-i_rec_seg_start}, "
                f"labels: {len(self.seg_labels)-i_rec_seg_start}, "
                f"class-dist: {np.unique(self.seg_labels[i_rec_seg_start:], return_counts=True)}, "
                f"seg_shape:{self.segments[-1].shape}"
            )
        self.debug(f"Segmentation done, total {len(self.segments)}.")

    def seg_augment(self, i_prev_seg_start_sec=20, step_sec=1, i_prev_seg_stop_sec=1):
        r"""Spawn segment from previous and current segment with specified\
        second slide."""
        prev_seg = self.segments[-2]
        cur_seg = self.segments[-1]
        cur_label = self.seg_labels[-1]
        i_cur_seg_stop_sec = self.seg_sec - (self.seg_sec - i_prev_seg_start_sec)
        count_new_seg = 0
        n_segs_before_aug = len(self.segments)
        # for i_prev_start_sec in range(i_prev_seg_start_sec, self.seg_sec):
        for i_prev_start_sec in range(
            i_prev_seg_start_sec, i_prev_seg_stop_sec, step_sec
        ):
            count_new_seg += 1
            r"Form new seg = 0.5*prev_seg + 0.5*cur_seg"
            new_seg = []
            # new_seg.extend(prev_seg[i_prev_start_sec*self.hz:, :])
            # new_seg.extend(cur_seg[:i_cur_seg_stop_sec*self.hz, :])
            new_seg = np.concatenate(
                (
                    prev_seg[i_prev_start_sec * self.hz :, :],
                    cur_seg[: i_cur_seg_stop_sec * self.hz, :],
                ),
                axis=0,
            )

            # print(f"new-seg-len:{len(new_seg)}, shape:{new_seg.shape}")
            assert new_seg.shape[0] == self.seg_sz

            r"Right shift prev and cur segment index by sampling hz."
            i_cur_seg_stop_sec += step_sec

            r"Add to segment and label global list."
            # self.segments.append(new_seg)
            # self.seg_labels.append(cur_label)
            self.segments.insert(-2, new_seg)
            self.seg_labels.insert(-2, cur_label)

            r"store segment index to record-wise store."
            self.record_wise_segments[self.record_names[-1]].append(
                len(self.segments) - 1
            )

        # self.debug(
        #     f"[Augment:{cur_label}] rec: {self.record_names[-1]}, "
        #     f"new_segs:{count_new_seg}, before:{n_segs_before_aug}, "
        #     f"after:{len(self.segments)}"
        # )

    def resegment_historical(self, i_rec_seg_start):
        r"""Enlarge current segment filling from historical segments."""
        seg_len_orig = Hz * SEG_SEC
        for i_cur_rec_seg in range(i_rec_seg_start, len(self.segments)):
            seg = self.segments[i_cur_rec_seg]
            new_seg_buf = np.zeros((seg_len_orig * SEG_LARGE_FACTOR, IN_CHAN))
            r"Place current segment to far right of new segment."
            # print(f"seg:{seg.shape}, new_seg_buf:{new_seg_buf.shape}")
            new_seg_buf[seg_len_orig * (SEG_LARGE_FACTOR - 1) :, :] = seg

            if i_cur_rec_seg - 1 < i_rec_seg_start:
                r"Not enough historical segments of current record."
                self.segments[i_cur_rec_seg] = new_seg_buf
                continue
            r"Copy historical segment's 1sec deducted portion."
            prev_seg = self.segments[i_cur_rec_seg - 1]
            # print(f"prev_seg:{seg.shape}, new_seg_buf:{new_seg_buf.shape}")
            new_seg_buf[: seg_len_orig * (SEG_LARGE_FACTOR - 1), :] = prev_seg[
                seg_len_orig:, :
            ]
            r"Update current segment."
            self.segments[i_cur_rec_seg] = new_seg_buf

        self.debug(
            f"[{self.record_names[-1]}] After add extended segs, "
            f"seg_created:{len(self.segments)-i_rec_seg_start}, "
            f"n_seg:{len(self.segments)}, seg_shape:{self.segments[-1].shape}"
        )

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return None, None

    def debug(self, msg):
        if DEBUG:
            self.log(msg)


class PartialDataset(EcgDataset):
    r"""Generate dataset from a parent dataset and indexes."""

    def __init__(
        self,
        dataset,
        seg_index,
        test=False,
        shuffle=False,
        balance_labels=False,
        balance_minority_idx=0,
        as_np=False,
    ):
        r"""Instantiate dataset from parent dataset and indexes."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.as_np = as_np
        if balance_labels:
            self.__balance_labels__(balance_minority_idx)

    def __balance_labels__(self, idx_minority):
        r"""Undersample majority class samples to balance dataset."""
        data_idx, data_dist = np.unique(
            [self.memory_ds.seg_labels[i] for i in self.indexes], return_counts=True
        )
        # np.unique returns tuple of two arrays i. index-array, and ii. data-array.
        log(
            f"partial-ds data-dist:{data_idx}, {data_dist}, idx_minority:{idx_minority}"
        )
        min_lbl, min_lbl_freq = np.argmin(data_dist), data_dist[np.argmin(data_dist)]
        if idx_minority:
            r"User defined minority class."
            min_lbl_freq = np.sort(data_dist)[idx_minority]
            min_lbl = np.where(data_dist == min_lbl_freq)[0][0]
        log(f"min_lbl/freq: {min_lbl}/{min_lbl_freq}")

        new_indexes = []
        for i_cur_data in data_idx:
            # if i_cur_data == min_lbl:
            #     continue
            n_reduce = data_dist[i_cur_data] - min_lbl_freq

            label_idx_idx = [
                i
                for i in range(len(self.indexes))
                if self.memory_ds.seg_labels[self.indexes[i]] == i_cur_data
            ]
            log(f"Label:{i_cur_data}, reduce: {n_reduce} from {len(label_idx_idx)}")
            for _ in range(n_reduce):
                i_del = random.randint(0, len(label_idx_idx) - 1)
                label_idx_idx.pop(i_del)
            new_indexes.extend([self.indexes[i] for i in label_idx_idx])
            log(
                f"Label:{i_cur_data}, reduced to {len(label_idx_idx)}, "
                f"detail:{new_indexes[len(new_indexes)-10:]}"
            )
        r"Replace self.indexes with new_indexes."
        self.indexes = new_indexes[:]

        r"After balancing"
        data_idx, data_dist = np.unique(
            [self.memory_ds.seg_labels[i] for i in self.indexes], return_counts=True
        )
        log(f"After balance, partial-ds data-dist:{data_idx}, {data_dist}")

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
        # trainX = np.array(self.memory_ds.segments[ID])
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY
        if torch.any(torch.isnan(X_tensor)):
            X_tensor = torch.nan_to_num(X_tensor)
        return X_tensor, Y_tensor

    def distort_seg(self, seg):
        r"""Add noise to segment."""
        noise = np.random.normal(0, 1, seg.shape[-1]).reshape(1, -1)
        seg += noise

    def debug(self, msg):
        r"""Output log message."""
        if DEBUG:
            self.memory_ds.log(msg)


def train_loov(input_dir, k_fold=5):
    r"""Train for leave-one-out testing.

    Subject-wise model testing which was trained with rest subjects using
    k-fold.
    """
    dataset = EcgDataset(
        input_directory=input_dir, hz=Hz, data_augment=False, n_skip_segments=1,
    )
    if DATA_AUG:
        augmented_dataset = EcgDataset(
            input_directory=input_dir,
            hz=Hz,
            data_augment=True,
            n_skip_segments=1,
            data_aug_cfg=DATA_AUG_CFG,
        )
    else:
        augmented_dataset = dataset

    avg_global = []

    for i_test_rec_name, test_rec_name in enumerate(dataset.record_names):
        test_idx = []
        test_idx.extend(dataset.record_wise_segments[test_rec_name])
        test_dataset = PartialDataset(dataset, seg_index=test_idx, test=True)

        preds = []
        labels = []
        fold_scores = {"acc": [], "prec": [], "recl": [], "f1": []}

        r"Fold strategy: balanced labels in train/validation."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

        r"Training sample database."
        train_idx = []
        for train_rec_name in dataset.record_names:
            if train_rec_name in [test_rec_name]:
                continue
            train_idx.extend(augmented_dataset.record_wise_segments[train_rec_name])

        for i_fold in range(1, k_fold + 1):
            r"Next split test/validation."
            train_index, val_index = next(
                skf.split(
                    np.zeros((len(train_idx), 1)),  # dummy signal
                    [augmented_dataset.seg_labels[i] for i in train_idx],
                )
            )
            train_dataset = PartialDataset(
                augmented_dataset,
                seg_index=train_index,
                shuffle=True,
                balance_labels=BALANCED_CLASS,
                balance_minority_idx=IDX_MINORITY_BALANCE,
            )
            val_dataset = PartialDataset(augmented_dataset, seg_index=val_index)

            preds_fold = []
            labels_fold = []

            model = create_model()
            log(model)

            r"Training."
            model_file = (
                f"{MODEL_PATH}/{LOG_FILE.replace('.log', '')}"
                f"_test{test_rec_name}_fold{i_fold}.pt"
            )

            class_weights = None
            if WEIGHTED_CLASS:
                labels_ = None
                labels_ = [
                    train_dataset.memory_ds.seg_labels[i] for i in train_dataset.indexes
                ]
                class_weights = torch.from_numpy(
                    model_training.get_class_weights(
                        labels_, n_class=NUM_CLASSES, log=logger.debug
                    )[-1]
                ).type(torch.FloatTensor)
                class_weights = class_weights.to(DEVICE)

            criterion = nn.CrossEntropyLoss(weight=class_weights)

            val_scores, train_stat = model_training.fit(
                model,
                train_dataset,
                val_dataset,
                test_dataset=test_dataset,
                device=DEVICE,
                model_file=model_file,
                max_epoch=MAX_EPOCH,
                early_stop_patience=EARLY_STOP_PATIENCE,
                early_stop_delta=EARLY_STOP_DELTA,
                weight_decay=WEIGHT_DECAY,
                lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
                batch_size=BATCH_SIZE,
                init_lr=INIT_LR,
                criterion=criterion,
                log=logger.debug,
            )

            fold_scores["acc"].append(val_scores["acc"])
            fold_scores["prec"].append(val_scores["prec"])
            fold_scores["recl"].append(val_scores["recl"])
            fold_scores["f1"].append(val_scores["f1"])

            preds_fold.extend(val_scores.get("preds"))
            labels_fold.extend(val_scores.get("labels"))

            r"Append fold preds/labels to global preds/labels list."
            preds.extend(preds_fold)
            labels.extend(labels_fold)

            train_dist = np.unique(
                [train_dataset.memory_ds.seg_labels[i] for i in train_dataset.indexes],
                return_counts=True,
            )
            val_dist = np.unique(
                [val_dataset.memory_ds.seg_labels[i] for i in val_dataset.indexes],
                return_counts=True,
            )
            test_dist = np.unique(
                [test_dataset.memory_ds.seg_labels[i] for i in test_dataset.indexes],
                return_counts=True,
            )
            log(
                f"@Fold:{i_fold}|rec:{test_rec_name}, "
                f"fold_acc:{val_scores['acc']:.05f}, "
                f"fold_prec:{val_scores['prec']:.05f}, "
                f"fold_recl:{val_scores['recl']:.05f}, "
                f"fold_f1:{val_scores['f1']:.05f}, "
                f"seg-train:{len(train_dataset)}, "
                f"train-dist:{train_dist}, "
                # f"seg-val:{val_rec_name}/{len(val_dataset)}, "
                f"val-dist:{val_dist}, "
                f"seg-test:{test_rec_name}/{len(test_dataset)}, "
                f"test-dist:{test_dist}, "
                f'Validation report:{val_scores.get("report")}, '
                f"confusion_matrix:{confusion_matrix(labels_fold, preds_fold)}"
            )

            r"Persist test preds and labels"
            if PERSIST_PREDS:
                pred_path = f"{LOG_PATH}/sleep_preds/" f"{LOG_FILE.replace('.log', '')}"
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                pred_file = f"{pred_path}/test{test_rec_name}_fold{i_fold}_.preds.csv"
                df = pd.DataFrame({"preds": preds_fold, "labels": labels_fold})
                df.to_csv(pred_file, index=True)

            r"Plot training statistics."
            model_training.plot_training_stat(
                0,
                train_stat,
                log_image_path=f"{BASE_PATH}/training_stat",
                image_file_suffix=f"test_{test_rec_name}",
            )

        _prec, _recl, _f1, _acc, report_dict = model_training.score(labels, preds)
        confusion_mat = confusion_matrix(labels, preds)
        log(
            f"@KFold summary|rec:{test_rec_name}, "
            f"acc:{np.average(fold_scores['acc']):.05f}, "
            f"prec:{np.average(fold_scores['prec']):.05f}, "
            f"recl:{np.average(fold_scores['recl']):.05f}, "
            f"f1:{np.average(fold_scores['f1']):.05f}, "
            f"report: {report_dict}, all-fold-confusion_matrix:{confusion_mat}"
        )

        avg_global.append(np.average(fold_scores["acc"]))

    log(f"@@EndOfSim, Global-avg:{np.average(avg_global)}")


def run():
    r"""Run simulation."""
    train_loov(input_dir=DATA_DIR, k_fold=5)


if __name__ == "__main__":
    model_training.fix_randomness()

    parser = argparse.ArgumentParser(
        description="Simulation: Sampling frequency & CNN architecture."
    )
    parser.add_argument("--i_cuda", required=False, default=0, help="CUDA device ID.")
    parser.add_argument(
        "--class_w", required=False, default=1, help="Class weight (0/1)."
    )
    # parser.add_argument("--aug", required=False, default=1, help="Data augment (0/1).")
    # parser.add_argument(
    #     "--i_model", required=False, default=2, help="Model index (0-3)."
    # )
    parser.add_argument("--n_classes", choices=["2", "3", "4", "5", "6"], required=True)
    parser.add_argument("--blk", choices=["2", "4"], default=4, required=True)
    parser.add_argument("--augCfg", choices=["0", "1", "2"], default=2, required=True)
    parser.add_argument(
        "--clzBalance",
        choices=["-1", "0", "1", "2", "3", "4", "5"],
        default=-1,
        required=False,
        help="0-5: idx-minority-class",  # index in array, sorted in fq ascending order,
    )
    args = parser.parse_args()

    DEBUG = True
    PERSIST_PREDS = True

    r"Train parameters."
    MAX_EPOCH = 100
    EARLY_STOP_PATIENCE = 17
    EARLY_STOP_DELTA = 0
    LR_SCHEDULER_PATIENCE = 5
    INIT_LR = 0.001
    WEIGHT_DECAY = 1e-6

    SEG_LARGE_FACTOR = 1
    N_DENSE_BLOCK = int(args.blk)
    Hz = 128 // 2
    SEG_SEC = 30
    IN_CHAN = 1
    START_TIME_TAG = f"{datetime.now():%Y%m%d%H%M%S}"
    # GRU = False
    BATCH_SIZE = 16
    # VAL_BATCH_SIZE = BATCH_SIZE
    CUDA = int(args.i_cuda)
    # IDX_MODEL = int(args.i_model)
    DATA_AUG_CFG = int(args.augCfg)
    # DATA_AUG = True if int(args.aug) == 1 else False
    DATA_AUG = DATA_AUG_CFG != 0
    WEIGHTED_CLASS = True if int(args.class_w) == 1 else False
    NUM_CLASSES = int(args.n_classes)
    BALANCED_CLASS = int(args.clzBalance) > -1
    IDX_MINORITY_BALANCE = int(args.clzBalance)
    if CUDA > 0:
        DEVICE = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEVICE == "cpu":
            DEVICE = torch.device(
                f"cuda:{CUDA}" if torch.cuda.is_available() else "cpu"
            )

    HOME = os.path.expanduser("~")
    DATA_DIR = f"{HOME}/data/SleepStageStudies/InputData"
    BASE_PATH = (
        f"{HOME}/py_code/runs/{sys.argv[0]}_{Hz}Hz_nClz{NUM_CLASSES}_"
        f"w{WEIGHTED_CLASS}_aug{DATA_AUG}_"
        f"{create_model().name()}_denseBlk{N_DENSE_BLOCK}_"
        f"{'balancedClz_' if BALANCED_CLASS else ''}"
        f"{'idxMinority'+str(IDX_MINORITY_BALANCE)+'_' if BALANCED_CLASS else ''}"
        f"cuda{CUDA}_"
        f"{START_TIME_TAG}"
    )
    LOG_PATH = BASE_PATH
    MODEL_PATH = f"{BASE_PATH}/models"
    LOG_FILE = f"{sys.argv[0]}_{Hz}hz.log"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if NUM_CLASSES == 6:
        LABELS_MAP = {
            "W": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "R": 5,
        }
    elif NUM_CLASSES == 5:
        LABELS_MAP = {
            "W": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 3,
            "R": 4,
        }
    elif NUM_CLASSES == 4:
        LABELS_MAP = {
            "W": 0,
            "1": 1,
            "2": 1,
            "3": 2,
            "4": 2,
            "R": 3,
        }
    elif NUM_CLASSES == 3:
        LABELS_MAP = {
            "W": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "R": 2,
        }
    elif NUM_CLASSES == 2:
        LABELS_MAP = {
            "W": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
            "R": 1,
        }

    config_logger()

    try:
        run()
    except:
        logger.error(f"Error occured, {traceback.format_exc()}")
