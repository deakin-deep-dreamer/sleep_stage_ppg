
import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import logging
import traceback

import torch.nn as nn
from sklearn.model_selection import train_test_split

from models import sincnet
# from data import datasource as ds
from data import datasource_physionet as ds
from lib import model_training


logger = logging.getLogger(__name__)


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
    # m_cfg_l2 = [CONV0_OUT_CHAN, 'A', CONV0_OUT_CHAN*1, 'A']
    # m_cfg = m_cfg_l2[:]
    # assert N_LAYER % 2 == 0
    # for n_layer in range(2, N_LAYER, 2):
    #     m_cfg.extend(m_cfg_l2)
    if N_LAYER == 2:
        m_cfg = [
            CONV0_OUT_CHAN, 'A', CONV_OUT_CHAN, 'A'
        ]
    elif N_LAYER == 4:
        m_cfg = [
            CONV0_OUT_CHAN, 'A', CONV_OUT_CHAN, 'A',
            CONV_OUT_CHAN, 'A', CONV_OUT_CHAN, 'A'
        ]
    else:
        raise NotImplementedError("N_LAYER > 4 not implemented yet.")
    model_cfg = {
        "cfg": m_cfg,
        "kernel": [
            KERNEL_SZ for _ in range(N_LAYER)
        ],
        "kernel_group": [
            1 if i == 0 else CONV0_OUT_CHAN for i in range(N_LAYER)
            # 1 for i in range(N_LAYER)
        ],
        'fq_band_lists': (
            FQ_BAND_LISTS if FQ_BAND_LISTS else (low_hz[:-1], band_hz)
        ),
        "n_class": 2,
        "input_sz": SEG_SIZE,
        'hz': Hz,
        'sinc_conv0': SINC_CONV0,
    }
    model = sincnet.SincNet(model_cfg, log=log)

    if FQ_BAND_FIXED:
        model.conv[0].requires_grad = False    # optional
        model.conv[0].low_hz_.requires_grad = False
        model.conv[0].band_hz_.requires_grad = False

    r"""    Correct way to freeze layers
    # we want to freeze the fc2 layer this time: only train fc1 and fc3
    net.fc2.weight.requires_grad = False
    net.fc2.bias.requires_grad = False

    # passing only those parameters that explicitly requires grad
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)

    # then do the normal execution of loss calculation and backward propagation

    #  unfreezing the fc2 layer for extra tuning if needed
    net.fc2.weight.requires_grad = True
    net.fc2.bias.requires_grad = True

    # add the unfrozen fc2 weight to the current optimizer
    optimizer.add_param_group({'params': net.fc2.parameters()})
    """

    return model


def debug_data(
    epoch, model=None, inputs=None, labels=None, preds=None,
    extra_out=None
):
    debug_path = f"{LOG_PATH}/debug"
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    filename = LOG_FILE.replace('.log', f"_epoch{epoch}.debug")
    debug_file = (
        f"{debug_path}/{filename}")
    data_dict = {
        'signal': inputs[0][0],
        'label': labels[0],
        'pred': preds.argmax(axis=1)[0],
    }

    if model.options.get('sinc_conv0'):
        out_layers = extra_out

        for i in range(len(out_layers.get('conv'))):
            # dim: batch X chan X seg.
            for k in range(out_layers.get('conv')[i].shape[1]):
                data_dict.update({
                    f"conv{i}_k{k}": out_layers.get('conv')[i][0][k]
                })

        final_conv_layer = out_layers.get('conv')[-1][0]
        data_dict.update({
            f"conv{len(out_layers.get('conv'))-1}_kr_argmax": np.argmax(final_conv_layer, axis=0)
        })

        r"Store band-pass filter optimisation."
        low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy()[:, 0]
        band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy()[:, 0]
        fband_line = f"{epoch-1}, "
        for i_sinc_chan in range(CONV0_OUT_CHAN):
            fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
        r"Remove trailing ,"
        fband_line = fband_line[:fband_line.rfind(',')]
        filename = LOG_FILE.replace('.log', ".fband")
        with open(f"{debug_path}/{filename}", 'a') as f:
            f.write(fband_line+'\n')

        # debug_model_param(epoch, model)

    df = pd.DataFrame(data_dict)
    df.to_csv(debug_file, index=True)


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


def run():
    r"""Run simulation."""
    model = create_model()

    r"Dataset. Random 80/20 train/validation split."
    dataset = ds.QrsDataset(
        DB_NAME, hz=Hz, seg_sec=SEG_SEC,
        seg_slide_sec=SEG_SLIDE_SEC, log=logger.debug
    )
    record_names = dataset.record_names
    rec_indexes = [i for i in range(len(record_names))]
    idx_rec_train, idx_rec_val = train_test_split(
        rec_indexes, test_size=0.2, random_state=1
    )
    train_records = [record_names[i] for i in idx_rec_train]
    val_records = [record_names[i] for i in idx_rec_val]

    train_dataset = ds.PartialQrsDataset(
        dataset, record_names=train_records, i_chan=0, log=logger.debug
    )
    val_dataset = ds.PartialQrsDataset(
        dataset, record_names=val_records, i_chan=0, log=logger.debug
    )

    logger.debug(model)

    model_file = (
        f"{MODEL_PATH}/{LOG_FILE.replace('.log', '.pt')}")

    r"Criteria with weighted loss for imabalanced class."
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=INIT_LR, weight_decay=0)

    _, train_stat = model_training.fit(
        model, train_dataset, val_dataset,
        test_dataset=None,
        device=DEVICE,
        model_file=model_file,
        max_epoch=MAX_EPOCH,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_delta=EARLY_STOP_DELTA,
        weight_decay=0,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
        batch_size=BATCH_SIZE,
        init_lr=INIT_LR,
        criterion=criterion,
        optimizer=optimizer,
        log=logger.debug,
        epoch_wise_debug_fn=debug_data)

    r"Plot training statistics."
    model_training.plot_training_stat(
        0, train_stat,
        log_image_path=f"{BASE_PATH}/training_stat",
        image_file_suffix=f"t{DB_NAME}")
    logger.info(
        f"Training is done. \n\tlow_hz_:{model.conv[0].low_hz_.data}, "
        f"\n\tband_hz_:{model.conv[0].band_hz_.data}")


if __name__ == '__main__':
    model_training.fix_randomness()

    parser = argparse.ArgumentParser(
        description='Simulation: Sampling frequency & CNN architecture.')
    parser.add_argument("--i_cuda", required=False, default=0)
    parser.add_argument("--i_traindb", required=False, default=0)
    args = parser.parse_args()

    DB_NAME = ds.DB_NAMES[int(args.i_traindb)]
    BATCH_SIZE = 16
    Hz = 100
    SEG_SEC = 3
    SEG_SIZE = Hz * SEG_SEC
    SEG_SLIDE_SEC = SEG_SEC
    # SEG_LEN = Hz * 3
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
    MAX_EPOCH = 100
    EARLY_STOP_PATIENCE = 15
    EARLY_STOP_DELTA = 0
    LR_SCHEDULER_PATIENCE = 5
    INIT_LR = 0.001
    # SIG_NORM = 3

    OUT_CHAN_FACTOR = 4
    CONV0_OUT_CHAN = OUT_CHAN_FACTOR*2
    CONV_OUT_CHAN = OUT_CHAN_FACTOR*2
    N_LAYER = 2
    KERNEL_SZ = 7
    SINC_CONV0 = True
    FQ_BAND_LISTS = None
    FQ_BAND_LISTS = [[4, 16, 18, 29], [2, 2, 3, 2]]
    # FQ_BAND_LISTS = [[18], [3]]
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
        f"train{DB_NAME}_L{N_LAYER}_{create_model().name()}_"
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

    LOG_FILE = (f"{sys.argv[0]}_{Hz}Hz_"
                f"{START_TIME_TAG}.log")

    config_logger()

    try:
        run()
    except:
        logger.error(f"Error occured, {traceback.format_exc()}")
