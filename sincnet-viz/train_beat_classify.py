
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import logging
import traceback

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from models import sincnet
from data import datasource_physionet as ds
from lib import model_training, signal_process


logger = logging.getLogger(__name__)


LABEL_MAP = None


def beat_type_map(reload=False):
    r"""Map beat types to AAMI heartbeat classes."""
    global LABEL_MAP
    if LABEL_MAP and not reload:
        return LABEL_MAP
    LABEL_GROUP = ['NLRej', 'AaJS', 'VE', 'F', '/fQ']
    LABEL_MAP = {}
    all_types = ''
    for i_lg, lg in enumerate(LABEL_GROUP):
        all_types += lg
        for lbl in lg:
            LABEL_MAP[lbl] = i_lg
    return LABEL_MAP, all_types


def config_logger():
    global logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def create_model(
    log=print
):
    low_hz = np.linspace(2.0, 60.0, CONV0_OUT_CHAN+1)
    # band_hz = np.diff(low_hz)
    band_hz = np.tile(4, CONV0_OUT_CHAN)    # narrow band of 2
    # logger.debug(f"low_hz:{low_hz}, band:{band_hz}")
    model = sincnet.SincNet({
        "cfg": [
            CONV0_OUT_CHAN, 'A', 'M', 4*CONV0_OUT_CHAN, 'A', 'M',
            # 8*CONV0_OUT_CHAN, 8*CONV0_OUT_CHAN, 'A', 'M',
            128, 128, 'A', 'M',
        ],
        # "cfg": [
        #     8*CONV0_OUT_CHAN, 8*CONV0_OUT_CHAN, 'M', 'A',
        #     # 8*CONV0_OUT_CHAN, 8*CONV0_OUT_CHAN, 'M',
        #     # 16*CONV0_OUT_CHAN, 16*CONV0_OUT_CHAN, 'M',
        #     # CONV0_OUT_CHAN, CONV0_OUT_CHAN, 'M',
        #     32*CONV0_OUT_CHAN, 32*CONV0_OUT_CHAN
        # ],
        # "cfg": [CONV0_OUT_CHAN if i == 0 else CONV0_OUT_CHAN*1 for i in range(0, N_LAYER)],
        # "cfg": [OUT_CHAN for i in range(1, N_LAYER+1)],
        "kernel": [KERNEL_SZ for _ in range(N_LAYER)],
        # "kernel_group": [1 for i in range(N_LAYER)],
        "kernel_group": [
            1 if i == 0 else CONV0_OUT_CHAN for i in range(N_LAYER)
        ],
        # ([1 if i == 0 else CONV0_OUT_CHAN \
        #                   for i in range(N_LAYER)] if SINC_CONV0 \
        #                  else [1 for i in range(N_LAYER)]),
        "n_class": NUM_CLASSES,
        "input_sz": SEG_LEN,
        'hz': Hz,
        'sinc_conv0': SINC_CONV0,
        'fq_band_lists': (
            # ([20], [15]) \
            # if CONV0_OUT_CHAN == 1 else (low_hz[:-1], band_hz)),
            FQ_BAND_LISTS if FQ_BAND_LISTS else (low_hz[:-1], band_hz)
        ),
        # 'fq_band_lists': ([20], [15]),
        'gap_layer': True,
        'flatten': True,
    }, log=log)

    if FQ_BAND_FIXED:
        model.conv[0].requires_grad = False    # optional
        model.conv[0].low_hz_.requires_grad = False
        model.conv[0].band_hz_.requires_grad = False
    return model


def debug_data(
    epoch, model=None, inputs=None, labels=None, preds=None,
    extra_out=None
):
    global sinc_training_token
    debug_path = f"{LOG_PATH}/debug"
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    filename = LOG_FILE.replace('.log', f"_epoch{epoch}.debug")
    # debug_file = (
    #     f"{debug_path}/{filename}")
    # data_dict = {
    #     # 'signal': inputs[0][0],
    #     'label': labels[0],
    #     'pred': preds.argmax(axis=1),
    # }
    # logger.debug(
    #     f"[debug_data] label:{labels.shape}, pred:{len(preds.argmax(axis=1))}")

    if model.options.get('sinc_conv0'):
        # out_layers = extra_out

        # for i in range(len(out_layers.get('conv'))):
        #     # dim: batch X chan X seg.
        #     for k in range(out_layers.get('conv')[i].shape[1]):
        #         data_dict.update({
        #             f"conv{i}_k{k}": out_layers.get('conv')[i][0][k]
        #         })

        # final_conv_layer = out_layers.get('conv')[-1][0]
        # logger.debug(f"final_conv_layer: {final_conv_layer.shape}")
        # data_dict.update({
        #     f"conv{len(out_layers.get('conv'))-1}_kr_argmax": np.argmax(final_conv_layer, axis=0)
        # })

        r"Store band-pass filter optimisation."
        low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy()[:, 0]
        band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy()[:, 0]
        fband_line = f"test:{sinc_training_token}, epoch:{epoch-1}, "
        for i_sinc_chan in range(CONV0_OUT_CHAN):
            fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
        r"Remove trailing ,"
        fband_line = fband_line[:fband_line.rfind(',')]
        filename = LOG_FILE.replace('.log', ".fband")
        with open(f"{debug_path}/{filename}", 'a') as f:
            f.write(fband_line+'\n')

        # debug_model_param(epoch, model)

    # logger.debug(data_dict)

    # df = pd.DataFrame(data_dict)
    # df.to_csv(debug_file, index=True)


def debug_model_param(
    epoch, model
):
    low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy()
    # logger.debug(f"--- low-hz: {low_hz.shape}, 0:{low_hz[0]}")
    low_hz = ', '.join(
        ['%.2f' % num for num in low_hz.flatten()])
    band_hz = ', '.join(
        ['%.2f' % num for num in model.conv[0].band_hz_.data.detach().cpu().numpy().flatten()])
    logger.debug(
        f"--|EpochDebug:{epoch}, "
        f"\n\tlow_hz:{low_hz}, "
        f"\n\tband_hz:{band_hz}")


sinc_training_token = None


def train_split_dataset(
    dataset, aug_dataset=None, idx_chan=-1
):
    r"""Split dataset into two subsets.
    According to Chazal et. al. - DS1 and DS2 in 'Automatic classification of
    heartbeats using ECG morphology and heartbeat interval features.'
    """
    DS_SPLITS = [
        ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
         '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
         '223', '230'],
        ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
         '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
         '233', '234']
    ]

    global sinc_training_token

    r"Two splits in turn used as training and testing."
    # for i_ds_split in range(2):
    for i_ds_split in range(1, 2):
        train_record_names = DS_SPLITS[i_ds_split]
        test_record_names = DS_SPLITS[(i_ds_split+1) % 2]

        sinc_training_token = f"ds_split_{i_ds_split}"

        r"Fold strategy: balanced labels in train/validation."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

        r"Training sample database."
        train_dataset, val_dataset = None, None
        train_idx = []
        for train_rec_name in train_record_names:
            train_parent_ds = aug_dataset if aug_dataset else dataset
            train_idx.extend(
                train_parent_ds.record_wise_segments[train_rec_name])
        # Training/validation 80/20 split.
        train_index, val_index = next(
            skf.split(
                np.zeros((len(train_idx), 1)),  # dummy train sample
                [train_parent_ds.seg_labels[i] for i in train_idx]
            )
        )
        train_dataset = ds.PartialEcgBeatDataset(
            train_parent_ds, seg_index=train_index, shuffle=True,
            i_chan=idx_chan)
        val_dataset = ds.PartialEcgBeatDataset(
            train_parent_ds, seg_index=val_index, i_chan=idx_chan)

        model = create_model(
            log=logger.debug)
        logger.debug(model)

        train_dist = np.unique(
            [train_dataset.memory_ds.seg_labels[i]
                for i in train_dataset.indexes], return_counts=True)
        val_dist = np.unique(
            [val_dataset.memory_ds.seg_labels[i]
                for i in val_dataset.indexes], return_counts=True)
        logger.debug(
            f"Training starts. Train:DS{i_ds_split}, channel:{idx_chan}, "
            f"#params:{model_training.count_parameters(model)}, "
            f"train_dist:{train_dist}, val_dist:{val_dist}, ")

        r"Training."
        model_file_suffix = LOG_FILE.replace('.log', f"_trainDs{i_ds_split}")
        model_file = (
            f"{MODEL_PATH}/{model_file_suffix}"
            f"_chan{idx_chan}.pt")

        r"Loss function with/without weighted class."
        _labels = [train_dataset.memory_ds.seg_labels[i]
                   for i in train_dataset.indexes]
        _clz_weights = model_training.get_class_weights(
            _labels, n_class=NUM_CLASSES)
        class_weights = torch.from_numpy(
            _clz_weights[-1]
        ).type(torch.FloatTensor)
        logger.debug(
            f"class-dist:{_clz_weights}, "
            f"weights: {class_weights}")
        class_weights = class_weights.to(DEVICE)

        r"Criteria with weighted loss for imabalanced class."
        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        val_scores, train_stat = model_training.fit(
            model, train_dataset, val_dataset,
            test_dataset=None,
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
            epoch_wise_debug_fn=debug_data)

        r"Plot training statistics."
        i_fold = 0
        image_path = LOG_FILE.replace('.log', "_train_stat")
        log_image_path = (
            f"{LOG_PATH}/training_stat/{image_path}")
        model_training.plot_training_stat(
            i_fold, train_stat,
            log_image_path=log_image_path,
            image_file_suffix=f"trainDs_{i_ds_split}_chan{idx_chan}")

        if SINC_CONV0:
            logger.info(
                f"Training Ds{i_ds_split} is done. "
                f"\n\tlow_hz_:{model.conv[0].low_hz_.data}, "
                f"\n\tband_hz_:{model.conv[0].band_hz_.data}")

        r"Model training done. Now test."
        for test_rec_name in test_record_names:
            test_idx = []
            test_idx.extend(
                dataset.record_wise_segments[test_rec_name])
            test_dataset = ds.PartialEcgBeatDataset(
                dataset, seg_index=test_idx, i_chan=idx_chan)
            preds = []
            labels = []

            r"Test record score."
            val_scores = model_training.validate(
                model, test_dataset, device=DEVICE, evaluate=True)

            r"Append fold preds/labels to global preds/labels list."
            preds.extend(val_scores.get('preds'))
            labels.extend(val_scores.get('labels'))

            logger.info(
                f"@Fold:{i_fold}|rec:{test_rec_name}_chan{idx_chan}, "
                f"fold_acc:{val_scores['acc']:.05f}, "
                f"fold_prec:{val_scores['prec']:.05f}, "
                f"fold_recl:{val_scores['recl']:.05f}, "
                f"fold_f1:{val_scores['f1']:.05f}, "
                f'Validation report:{val_scores.get("report")}, '
                f"confusion_matrix:\n{confusion_matrix(labels, preds)}")

            r"Persist preds and labels."
            if PERSIST_PREDS:
                pred_path = (
                    f"{LOG_PATH}/preds/"
                    f"{LOG_FILE.replace('.log', '')}")
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                pred_file = (
                    f"{pred_path}/trainDs{i_ds_split}_test{test_rec_name}_"
                    f"chan{idx_chan}_"
                    f"fold{i_fold}_.preds.csv"
                    # f"val{val_rec_name}.preds.csv"
                )
                df = pd.DataFrame({
                    'preds': preds,
                    'labels': labels
                })
                df.to_csv(pred_file, index=True)


def train_loov(
    dataset, k_fold=5, idx_chan=-1
):
    r"""Train for leave-one-out testing."""
    global sinc_training_token
    avg_global = []
    for i_test_rec_name, test_rec_name in enumerate(dataset.record_names):
        sinc_training_token = test_rec_name
        test_idx = []
        test_idx.extend(
            dataset.record_wise_segments[test_rec_name])
        test_dataset = ds.PartialEcgBeatDataset(
            dataset, seg_index=test_idx, i_chan=idx_chan)

        preds = []
        labels = []
        fold_scores = {
            'acc': [],
            'prec': [],
            'recl': [],
            'f1': []
        }

        r"Fold strategy: balanced labels in train/validation."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

        r"Training sample database."
        train_idx = []
        for train_rec_name in dataset.record_names:
            if train_rec_name in [test_rec_name]:
                continue
            train_idx.extend(
                dataset.record_wise_segments[train_rec_name])

        for i_fold in range(1, k_fold+1):
            train_index, val_index = next(
                skf.split(
                    np.zeros((len(train_idx), 1)),  # dummy train sample
                    [dataset.seg_labels[i] for i in train_idx]
                )
            )
            train_dataset = ds.PartialEcgBeatDataset(
                dataset, seg_index=train_index, shuffle=True, i_chan=idx_chan)
            val_dataset = ds.PartialEcgBeatDataset(
                dataset, seg_index=val_index, i_chan=idx_chan)

            preds_fold = []
            labels_fold = []

            model = create_model()
            if i_test_rec_name == 0 and i_fold == 1:
                logger.debug(model)

            train_dist = np.unique(
                [train_dataset.memory_ds.seg_labels[i]
                    for i in train_dataset.indexes], return_counts=True
            )
            val_dist = np.unique(
                [val_dataset.memory_ds.seg_labels[i]
                    for i in val_dataset.indexes], return_counts=True
            )
            test_dist = np.unique(
                [test_dataset.memory_ds.seg_labels[i]
                    for i in test_dataset.indexes], return_counts=True
            )

            logger.debug(
                f"Training starts. Test:{test_rec_name}, channel:{idx_chan}, "
                f"#params:{model_training.count_parameters(model)}, "
                f"train_dist:{train_dist}, val_dist:{val_dist}, "
                f"test_dist:{test_dist}")

            r"Training."
            model_file_suffix = LOG_FILE.replace('.log', f"_fold{i_fold}")
            model_file = (
                f"{MODEL_PATH}/{model_file_suffix}"
                f"_test{test_rec_name}_chan{idx_chan}.pt")

            r"Loss function with/without weighted class."
            _labels = [train_dataset.memory_ds.seg_labels[i]
                       for i in train_dataset.indexes]
            _clz_weights = model_training.get_class_weights(
                _labels, n_class=NUM_CLASSES)
            class_weights = torch.from_numpy(
                _clz_weights[-1]
            ).type(torch.FloatTensor)
            logger.debug(
                f"class-dist:{_clz_weights}, "
                f"weights: {class_weights}")
            class_weights = class_weights.to(DEVICE)

            r"Criteria with weighted loss for imabalanced class."
            criterion = nn.CrossEntropyLoss(
                weight=class_weights
            )

            val_scores, train_stat = model_training.fit(
                model, train_dataset, val_dataset,
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
                epoch_wise_debug_fn=debug_data)

            r"Fold metrics."
            fold_scores['acc'].append(val_scores['acc'])
            fold_scores['prec'].append(val_scores['prec'])
            fold_scores['recl'].append(val_scores['recl'])
            fold_scores['f1'].append(val_scores['f1'])
            preds_fold.extend(val_scores.get('preds'))
            labels_fold.extend(val_scores.get('labels'))

            r"Append fold preds/labels to global preds/labels list."
            preds.extend(preds_fold)
            labels.extend(labels_fold)

            logger.info(
                f"@Fold:{i_fold}|rec:{test_rec_name}_chan{idx_chan}, "
                f"fold_acc:{val_scores['acc']:.05f}, "
                f"fold_prec:{val_scores['prec']:.05f}, "
                f"fold_recl:{val_scores['recl']:.05f}, "
                f"fold_f1:{val_scores['f1']:.05f}, "
                f'Validation report:{val_scores.get("report")}, '
                f"confusion_matrix:{confusion_matrix(labels_fold, preds_fold)}")

            r"Plot training statistics."
            image_path = LOG_FILE.replace('.log', f"_train_stat")
            log_image_path = (
                f"{LOG_PATH}/training_stat/{image_path}")
            model_training.plot_training_stat(
                i_fold, train_stat,
                log_image_path=log_image_path,
                image_file_suffix=f"{test_rec_name}_chan{idx_chan}")

            r"Persist preds and labels."
            if PERSIST_PREDS:
                pred_path = (
                    f"{LOG_PATH}/preds/"
                    f"{LOG_FILE.replace('.log', '')}")
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                pred_file = (
                    f"{pred_path}/test{test_rec_name}_chan{idx_chan}_"
                    f"fold{i_fold}_.preds.csv"
                    # f"val{val_rec_name}.preds.csv"
                )
                df = pd.DataFrame({
                    'preds': preds_fold,
                    'labels': labels_fold
                })
                df.to_csv(pred_file, index=True)

        _prec, _recl, _f1, _acc, report_dict = model_training.score(
            labels, preds)
        confusion_mat = confusion_matrix(labels, preds)
        logger.info(
            f"@KFold summary|rec:{test_rec_name}_chan{idx_chan}, "
            f"acc:{np.average(fold_scores['acc']):.05f}, "
            f"prec:{np.average(fold_scores['prec']):.05f}, "
            f"recl:{np.average(fold_scores['recl']):.05f}, "
            f"f1:{np.average(fold_scores['f1']):.05f}, "
            f"report: {report_dict}, all-fold-confusion_matrix:{confusion_mat}"
        )

        avg_global.append(np.average(fold_scores['acc']))

    logger.info(
        f'@@EndOfSim, Global-avg:{np.average(avg_global)}, detail:{avg_global}')


def viz_beats():
    r"""Visualise beats."""
    type_map, beat_types = beat_type_map()
    dataset = ds.EcgBeatDataset(
        dbname=DB_NAME, n_samp_around=128, log=print,
        valid_beats=beat_types, label_map=type_map, augment=True)

    for rec_name in dataset.record_names:
        indexes = dataset.record_wise_segments.get(rec_name)
        labels = [dataset.seg_labels[i] for i in indexes]
        print(
            f"[{rec_name}]: {np.unique(labels, return_counts=True)}")

    partialDataset = ds.PartialEcgBeatDataset(
        dataset,
        seg_index=dataset.record_wise_segments[dataset.record_names[0]],
        log=logger.debug, as_np=True)
    partialDataset.on_epoch_end()

    for i in range(100):
        seg, lbl = partialDataset[i]
        print(seg.shape)
        fig, ax = plt.subplots()
        ax.plot(range(len(seg.T[0])), seg.T[0])
        # ax.plot(range(len(seg)), seg)
        ax.set_title(lbl)
        plt.show()
        # break
        time.sleep(0.5)


def run():
    r"""Run simulation."""
    model = create_model()
    logger.debug(model)

    # viz_beats()

    type_map, beat_types = beat_type_map()
    dataset = ds.EcgBeatDataset(
        dbname=DB_NAME, hz=Hz, valid_beats=beat_types,
        label_map=type_map, n_samp_around=N_SAMP_AROUND,
        log=logger.debug)
    aug_dataset = ds.EcgBeatDataset(
        dbname=DB_NAME, hz=Hz, valid_beats=beat_types,
        label_map=type_map, n_samp_around=N_SAMP_AROUND, augment=True,
        log=logger.debug)
    # train_loov(
    #     dataset, k_fold=1, idx_chan=0
    # )
    train_split_dataset(
        dataset, aug_dataset=aug_dataset, idx_chan=0
    )


if __name__ == '__main__':
    model_training.fix_randomness()

    # DB_NAME = ds.DB_NAMES[0]
    # viz_beats()

    parser = argparse.ArgumentParser(description='ECG beat classification.')
    parser.add_argument("--i_cuda", required=False, default=0)
    parser.add_argument("--i_db", required=False, default=0)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    DB_NAME = ds.DB_NAMES[int(args.i_db)]
    BATCH_SIZE = 16
    # Hz = 360 // 2
    Hz = 100  # 250
    N_SAMP_AROUND = Hz // 3
    SEG_LEN = 1 + 2 * N_SAMP_AROUND
    CUDA = int(args.i_cuda)
    if CUDA > 0:
        DEVICE = torch.device(
            f"cuda:{CUDA}" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEVICE == 'cpu':
            DEVICE = torch.device(
                f"cuda:{CUDA}" if torch.cuda.is_available() else "cpu")

    r"Train parameters."
    MAX_EPOCH = 200
    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_DELTA = 0
    LR_SCHEDULER_PATIENCE = 5
    INIT_LR = 0.001
    # SIG_NORM = 3
    NUM_CLASSES = 5
    WEIGHT_DECAY = 1e-6
    PERSIST_PREDS = True

    OUT_CHAN_FACTOR = 4
    CONV0_OUT_CHAN = OUT_CHAN_FACTOR*2
    N_LAYER = 10
    KERNEL_SZ = 11
    SINC_CONV0 = True
    FQ_BAND_LISTS = None
    # FQ_BAND_LISTS = [[8, 16, 22, 29, 37, 45], [4, 3, 3, 3, 3, 2]]
    FQ_BAND_LISTS = [[2, 22, 37, 45], [4, 3, 3, 2]]
    FQ_BAND_FIXED = False
    if FQ_BAND_LISTS:
        CONV0_OUT_CHAN = len(FQ_BAND_LISTS[0])
        FQ_BAND_FIXED = True

    START_TIME_TAG = f"{datetime.now():%Y%m%d%H%M%S}"
    # HOME = '/home/582/ah9647'
    # HOME = "/home/mahabib"
    HOME = os.path.expanduser("~")
    BASE_PATH = (
        f"{HOME}/py_code/runs/{sys.argv[0]}_{Hz}Hz_"
        f"train{DB_NAME}_{create_model().name()}_"
        f"fxband{FQ_BAND_FIXED}_"
        f"{START_TIME_TAG}")
    LOG_PATH = BASE_PATH
    MODEL_PATH = f"{BASE_PATH}/models"
    LOG_FILE = (f"{sys.argv[0]}_train{DB_NAME}_{Hz}hz.log")
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    LOG_FILE = (f"{sys.argv[0]}_{Hz}Hz_{create_model().name()}_"
                f"{START_TIME_TAG}.log")

    config_logger()

    try:
        run()
    except:
        logger.error(f"Error occured, {traceback.format_exc()}")
