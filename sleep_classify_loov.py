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
# from models import model_zoo_classif2 as model_zoo_classif
from lib import model_training
import model
import datasource

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

    _model = model.DenseNet(
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
    return _model


def train_loov(input_dir, k_fold=5):
    r"""Train for leave-one-out testing.

    Subject-wise model testing which was trained with rest subjects using
    k-fold.
    """
    dataset = datasource.PPGDataset(
        input_directory=input_dir, hz=Hz, data_augment=False, n_skip_segments=1,
    )
    if DATA_AUG:
        augmented_dataset = dataset.PPGDataset(
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
        test_dataset = datasource.PartialDataset(dataset, seg_index=test_idx, test=True)

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
            train_dataset = datasource.PartialDataset(
                augmented_dataset,
                seg_index=train_index,
                shuffle=True,
                balance_labels=BALANCED_CLASS,
                balance_minority_idx=IDX_MINORITY_BALANCE,
            )
            val_dataset = datasource.PartialDataset(
                augmented_dataset, seg_index=val_index
            )

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
