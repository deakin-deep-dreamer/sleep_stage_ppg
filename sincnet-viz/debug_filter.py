
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from models import sincnet
# from data import datasource
# from lib import model_training


class SincNet(nn.Module):
    def __init__(self, options, log=print):
        super(SincNet, self).__init__()

        self.iter = 0
        self.log = log
        self.options = options

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])

        # self.feature = self.create_encoder()
        self.create_encoder()
        self.classifier = nn.Conv1d(
            in_channels=self.hidden_dim, out_channels=options["n_class"],
            kernel_size=1, padding=0, dilation=1)

    def name(self):
        r"Format cfg meaningfully."
        return (
            f"{self.__class__.__name__}_"
            f"{sincnet.model_cfg_str(self.options)}")

    def forward(
        self, x, i_zero_kernels=None, fn_sinc_kernel_modify=None,
        modify_params=None
    ):
        self.debug(f"input: {x.shape}")

        # out = self.feature(x)
        # self.debug(f"features: {out.shape}")

        layer_out = {
            'conv': [],
            'bn': [],
            'act': [],
        }

        out = x

        i_conv, i_pool, i_act = 0, 0, 0

        for x in self.options["cfg"]:
            if x == 'M':
                out = self.pool[i_pool](out)
                i_pool += 1
            elif x == 'A':
                out = self.act[i_act](out)
                i_act += 1
            else:
                out = self.conv[i_conv](out)
                self.debug(f"conv{i_conv} out: {out.shape}")

                if i_conv == 0 and fn_sinc_kernel_modify:
                    fn_sinc_kernel_modify(out, modify_params)
                elif i_zero_kernels:
                    r"zero kernel if index is non-negative."
                    i_zero_kernels = i_zero_kernels if i_zero_kernels else []
                    for i_zero_kernel in i_zero_kernels:
                        if i_zero_kernel < 0:
                            continue
                        _b, _c, _d = out.shape
                        out[0, i_zero_kernel, :] = torch.zeros(_d)

                r"Layer output debug."
                layer_out.get('conv').append(
                    out.detach().cpu().numpy()
                )

                out = self.bn[i_conv](out)

                # layer_out.get('bn').append(
                #     out.detach().cpu().numpy()
                # )

                # out = self.act[i_conv](out)

                # layer_out.get('act').append(
                #     out.detach().cpu().numpy()
                # )

                i_conv += 1

        out = self.classifier(out)
        self.debug(f"classif: {out.shape}")

        self.iter += 1
        return out, layer_out

    def create_encoder(self):
        # layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = 1
        input_sz = self.options["input_sz"]
        for x in self.options["cfg"]:
            if x == 'M':
                count_pooling += 1
                # layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.pool.append(nn.MaxPool1d(kernel_size=2, stride=2))
                input_sz /= 2
            elif x == 'A':
                self.act.append(
                    nn.ELU(inplace=True)
                    # nn.ReLU(inplace=True)
                )
            else:
                if i_kernel == 0 and self.options['sinc_conv0']:
                    conv = sincnet.SincConv1d_fast(
                        in_channels=in_chan, out_channels=x,
                        kernel_dim=self.options["kernel"][i_kernel],
                        hz=self.options["hz"],
                        log=self.log,
                        padding=sincnet.padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1))
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan, out_channels=x,
                        kernel_size=self.options["kernel"][i_kernel],
                        groups=self.options["kernel_group"][i_kernel],
                        padding=sincnet.padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1))
                i_kernel += 1
                # layers += [
                #     conv,
                #     nn.BatchNorm1d(x),
                #     nn.ELU(inplace=True)
                #     ]
                self.conv.append(conv)
                self.bn.append(nn.BatchNorm1d(x))
                # self.act.append(
                #     nn.ELU(inplace=True)
                #     # nn.ReLU(inplace=True)
                # )
                in_chan = self.hidden_dim = x
            pass    # for
        # return nn.Sequential(*layers)

    def debug(self, *args):
        if self.iter == 0 and self.log:
            self.log(args)


def log(msg):
    print(msg)


def create_model(
    model_cfg=None
):
    if model_cfg:
        return SincNet(model_cfg, log=None)
    return SincNet({
        # "cfg": [CONV0_OUT_CHAN*(2**i) for i in range(0, N_LAYER)],
        "cfg": [CONV0_OUT_CHAN if i == 0 else CONV0_OUT_CHAN*1 for i in range(0, N_LAYER)],
        # "cfg": [OUT_CHAN for i in range(1, N_LAYER+1)],
        "kernel": [KERNEL_SZ for _ in range(N_LAYER)],
        # "kernel_group": [1 if i == 0 else OUT_CHAN for i in range(N_LAYER)],
        "kernel_group": [1 if i == 0 else CONV0_OUT_CHAN for i in range(N_LAYER)],
        "n_class": 2,
        "input_sz": SEG_LEN,
        'hz': 100,
        'sinc_conv0': SINC_CONV0,
    }, log=None)


def load_model(
    trained_model=None, device='cpu', model_cfg=None
):
    r"""Load a model."""
    raw_model = create_model(model_cfg)

    if trained_model:
        tmp_model_file = "tmp_model.pt"
        torch.save(trained_model.state_dict(), tmp_model_file)
        raw_model.load_state_dict(
            torch.load(tmp_model_file, map_location=torch.device(device))
        )
        # log("Model loaded from parameter.")
        return raw_model

    r"Find model file."
    MODEL_PATH = f"{BASE_PATH}/models"
    for f in os.listdir(MODEL_PATH):
        if not f.endswith(".pt"):
            continue
        model_file = os.path.join(MODEL_PATH, f)

        # log(f"Load model: {model_file} ...")

        raw_model.load_state_dict(
            torch.load(model_file, map_location=torch.device(device))
        )
        log(f"Model loaded: {model_file}")
        return raw_model


def debug_data(
    model, signal, labels, preds, preds_raw=None, out_layers=None, epoch=0,
    rec_name=None, i_seg=0, debug_path=None, file_suffix=None, in_memory=False
):
    r"""Debug model layer and prediction."""
    # debug_path = f"{VALIDATATION_PATH}/debug/{rec_name}"
    if not in_memory:
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        debug_file = f"{debug_path}/{i_seg}_{file_suffix}.debug"

    data_dict = {
        'signal': signal,
        'label': labels.astype(int),
        'pred': preds,
    }

    for i in range(len(out_layers.get('conv'))):
        # first element in batch
        for k in range(out_layers.get('conv')[i].shape[1]):
            # print(
            #     f"out_layer:{i}, kernel:{k}, data:{out_layers.get('conv')[i][0][k].shape}")
            data_dict.update({
                f"conv{i}_k{k}": out_layers.get('conv')[i][0][k]
            })
            # df.to_csv(debug_file, mode="a")

    final_conv_layer = out_layers.get('conv')[-1][0]
    data_dict.update({
        f"conv{len(out_layers.get('conv'))-1}_kr_argmax": np.argmax(final_conv_layer, axis=0)
    })

    r"Preds-raw (2-class)."
    data_dict.update({
        "clz0": preds_raw[0][0],
        "clz1": preds_raw[0][1],
    })
    df = pd.DataFrame(data_dict)
    if not in_memory:
        df.to_csv(debug_file, index=True, mode="w")
        return

    return df


def seg_predict(
    model, segment, i_zero_kernels=None, modify_fn_params=None
):
    r"""Predict segment."""
    model.eval()
    with torch.no_grad():
        X_tensor = Variable(torch.from_numpy(
            segment)).type(torch.FloatTensor)
        X_tensor = X_tensor.view(1, 1, X_tensor.size()[0])
        inputs = X_tensor.to(DEVICE)
        model.to(DEVICE)

        if modify_fn_params:
            preds, out_layers = model(
                inputs, fn_sinc_kernel_modify=fn_sinc_kernel_modify,
                modify_params=modify_fn_params)
        else:
            preds, out_layers = model(
                inputs, i_zero_kernels=i_zero_kernels)

        preds = preds.detach().cpu().numpy()
        out_mask = preds.argmax(axis=1)
        out_mask = out_mask.flatten()

    return preds.reshape((2, -1)), out_mask, out_layers


def fn_sinc_kernel_modify(
    sinc_fmap, modify_params
):
    i_kr = modify_params.get('i_kr')
    i_sig = modify_params.get('i_sig')
    r"""Alter a single item of feature-map."""
    _b, _c, _d = sinc_fmap.shape
    # log(f"Reset [0, {i_kr}, {i_sig}]: {sinc_fmap.detach().cpu()[0,i_kr,:i_sig+3]}")
    sinc_fmap[0, i_kr, i_sig].fill_(0)
    # log(f"*Reset [0, {i_kr}, {i_sig}]: {sinc_fmap.detach().cpu()[0,i_kr,:i_sig+3]}")


r"""Global model, load once."""
debug_model = None


def alter_filter_items(
    model, model_cfg=None, signal=None, n_conv0_filter=1, pred_dim=2,
    hz=100, n_layers=2, kernel_sz=7, data_seg_sec=3, i_kernels=None
):
    r"""Reset each feature-map bit at a time and record score.

    This process calculates 'explanation-vector', G_ijk.
    """
    global BASE_PATH, CONV0_OUT_CHAN, N_LAYER, KERNEL_SZ, Hz, DATA_SEG_SEC, SEG_LEN, \
        IN_CHAN, DEVICE, SINC_CONV0, debug_model
    # BASE_PATH = base_path
    Hz = hz
    CONV0_OUT_CHAN = n_conv0_filter
    N_LAYER = n_layers
    KERNEL_SZ = kernel_sz
    SINC_CONV0 = True
    DATA_SEG_SEC = data_seg_sec
    SEG_LEN = Hz * DATA_SEG_SEC
    IN_CHAN = 1
    DEVICE = 'cpu'
    n_samp = SEG_LEN

    # df = pd.read_csv(f"{BASE_PATH}/{seg_file}", skiprows=0)
    # segment = df['signal']
    # log(
    #     f"Signal: {segment.shape}, path:{seg_file}, "
    #     # f"\nsig:{segment.values[:3]}"
    # )

    if debug_model is None:
        debug_model = load_model(model, model_cfg=model_cfg)
    model = debug_model

    r"Print sinc-conv f1 and band."
    low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy().reshape((-1))
    band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy().reshape((-1))
    # low_hz = ', '.join(['%.2f' % num for num in low_hz])
    # band_hz = ', '.join(['%.2f' % num for num in band_hz])
    # print(
    #     f"\tlow_hz:{', '.join(['%.2f' % num for num in low_hz])}, "
    #     f"\n\tband_hz:{', '.join(['%.2f' % num for num in band_hz])}")

    raw_preds, out_mask, _ = seg_predict(
        model, signal)

    explanation_vector = np.zeros((n_conv0_filter, n_samp))
    z_ref_scores = np.zeros((n_conv0_filter*pred_dim, n_samp))
    z_changed_scores = np.zeros((n_conv0_filter*pred_dim, n_samp))
    pred_mask_vec = np.zeros((n_conv0_filter, n_samp))
    pred_mask_changed_vec = np.zeros((n_conv0_filter, n_samp))

    for i_kr in range(n_conv0_filter):
        if i_kernels and i_kr not in i_kernels:
            continue
        for i_seg_samp in range(n_samp):
            # raw_preds, out_mask, _ = seg_predict(
            #     model, signal)
            r"Incremental segment prediction by tempering a single sample \
            at a time."
            raw_preds_modify, out_mask_modify, _ = seg_predict(
                model, signal, modify_fn_params={
                    'i_kr': i_kr,
                    'i_sig': i_seg_samp
                })
            # log(
            #     f"raw_preds: {raw_preds.shape}, out_mask:{out_mask.shape} "
            #     f"=> {out_mask}")
            for i_pred_dim in range(pred_dim):
                r"Track 2-out channels (original and updated) per kernel."
                score_ = raw_preds[i_pred_dim, i_seg_samp]
                z_ref_scores[2*i_kr+i_pred_dim,
                             i_seg_samp] = score_
                score_ = raw_preds_modify[i_pred_dim, i_seg_samp]
                z_changed_scores[2*i_kr+i_pred_dim,
                                 i_seg_samp] = score_

            r"Exp vector for sample is the difference of z-scores. \
            Only dominant class score change was recorded."
            i_out_dominant = out_mask[i_seg_samp]
            # log(f"i_out_dominant: {i_out_dominant}")
            delta_z = raw_preds[i_out_dominant, i_seg_samp] \
                - raw_preds_modify[i_out_dominant, i_seg_samp]
            # delta_z = round(delta_z, 2)
            # log(f"kr{i_kr}, samp:{i_seg_samp}, delta_z:{delta_z}")
            explanation_vector[i_kr, i_seg_samp] = delta_z

            r"Record pred-mask and modified pred-mask."
            pred_mask_vec[i_kr, i_seg_samp] = out_mask[i_seg_samp]
            pred_mask_changed_vec[i_kr,
                                  i_seg_samp] = out_mask_modify[i_seg_samp]
            # break
    r"Create result DataFrame."
    data_dict = {
        'signal': signal,
        'pred_mask': out_mask,
    }
    for i_kr in range(n_conv0_filter):
        if i_kernels and i_kr not in i_kernels:
            continue
        data_dict.update({
            f"exp_vec_kr{i_kr}": explanation_vector[i_kr],
            f"out_mask_kr{i_kr}": pred_mask_vec[i_kr],
            f"u_out_mask_kr{i_kr}": pred_mask_changed_vec[i_kr],
        })
        for i_pred_dim in range(pred_dim):
            data_dict.update({
                f"kr{i_kr}_out{i_pred_dim}": z_ref_scores[2*i_kr+i_pred_dim],
                f"u_kr{i_kr}_out{i_pred_dim}": z_changed_scores[2*i_kr+i_pred_dim],
            })
    return pd.DataFrame(data_dict), low_hz, band_hz


def _alter_filter_item(
    model=None, signal=None, i_kr=None, i_seg_samp=None, pred_dim=None,
    raw_preds=None, z_ref_scores=None, z_changed_scores=None, out_mask=None,
    explanation_vector=None, pred_mask_vec=None, pred_mask_changed_vec=None
):
    raw_preds_modify, out_mask_modify, _ = seg_predict(
        model, signal, modify_fn_params={
            'i_kr': i_kr,
            'i_sig': i_seg_samp
        })
    for i_pred_dim in range(pred_dim):
        r"Track 2-out channels (original and updated) per kernel."
        score_ = raw_preds[i_pred_dim, i_seg_samp]
        z_ref_scores[2*i_kr+i_pred_dim,
                     i_seg_samp] = score_  # round(score_, 2)
        score_ = raw_preds_modify[i_pred_dim, i_seg_samp]
        z_changed_scores[2*i_kr+i_pred_dim,
                         i_seg_samp] = score_  # round(score_, 2)

    r"Exp vector for sample is the difference of z-scores. \
    Only dominant class score change was recorded."
    i_out_dominant = out_mask[i_seg_samp]
    # log(f"i_out_dominant: {i_out_dominant}")
    delta_z = raw_preds[i_out_dominant, i_seg_samp] \
        - raw_preds_modify[i_out_dominant, i_seg_samp]
    # delta_z = round(delta_z, 2)
    # log(f"kr{i_kr}, samp:{i_seg_samp}, delta_z:{delta_z}")
    explanation_vector[i_kr, i_seg_samp] = delta_z

    r"Record pred-mask and modified pred-mask."
    pred_mask_vec[i_kr, i_seg_samp] = out_mask[i_seg_samp]
    pred_mask_changed_vec[i_kr,
                          i_seg_samp] = out_mask_modify[i_seg_samp]


def alter_filter_items_file(
    base_path=None, seg_file=None, n_conv0_filter=1, pred_dim=2, hz=100, n_layers=2,
    kernel_sz=7, data_seg_sec=3
):
    r"""Reset each feature-map bit at a time and record score.

    This process calculates 'explanation-vector', G_ijk.
    """
    global BASE_PATH, CONV0_OUT_CHAN, N_LAYER, KERNEL_SZ, Hz, DATA_SEG_SEC, SEG_LEN, \
        IN_CHAN, DEVICE, SINC_CONV0
    BASE_PATH = base_path
    Hz = hz
    CONV0_OUT_CHAN = n_conv0_filter
    N_LAYER = n_layers
    KERNEL_SZ = kernel_sz
    SINC_CONV0 = True
    DATA_SEG_SEC = data_seg_sec
    SEG_LEN = Hz * DATA_SEG_SEC
    IN_CHAN = 1
    DEVICE = 'cpu'
    n_samp = SEG_LEN

    df = pd.read_csv(f"{BASE_PATH}/{seg_file}", skiprows=0)
    segment = df['signal']
    log(
        f"Signal: {segment.shape}, path:{seg_file}, "
        # f"\nsig:{segment.values[:3]}"
    )

    model = load_model()

    r"Print sinc-conv f1 and band."
    low_hz = model.conv[0].low_hz_.data.detach().cpu().numpy().reshape((-1))
    band_hz = model.conv[0].band_hz_.data.detach().cpu().numpy().reshape((-1))
    # low_hz = ', '.join(['%.2f' % num for num in low_hz])
    # band_hz = ', '.join(['%.2f' % num for num in band_hz])
    print(
        f"\tlow_hz:{', '.join(['%.2f' % num for num in low_hz])}, "
        f"\n\tband_hz:{', '.join(['%.2f' % num for num in band_hz])}")

    explanation_vector = np.zeros((n_conv0_filter, n_samp))
    z_ref_scores = np.zeros((n_conv0_filter*pred_dim, n_samp))
    z_changed_scores = np.zeros((n_conv0_filter*pred_dim, n_samp))
    pred_mask_vec = np.zeros((n_conv0_filter, n_samp))
    pred_mask_changed_vec = np.zeros((n_conv0_filter, n_samp))
    for i_kr in range(n_conv0_filter):
        for i_seg_samp in range(n_samp):
            raw_preds, out_mask, _ = seg_predict(
                model, segment.values)
            raw_preds_modify, out_mask_modify, _ = seg_predict(
                model, segment.values, modify_fn_params={
                    'i_kr': i_kr,
                    'i_sig': i_seg_samp
                })
            # log(
            #     f"raw_preds: {raw_preds.shape}, out_mask:{out_mask.shape} "
            #     f"=> {out_mask}")
            for i_pred_dim in range(pred_dim):
                r"Track 2-out channels (original and updated) per kernel."
                score_ = raw_preds[i_pred_dim, i_seg_samp]
                z_ref_scores[2*i_kr+i_pred_dim,
                             i_seg_samp] = score_  # round(score_, 2)
                score_ = raw_preds_modify[i_pred_dim, i_seg_samp]
                z_changed_scores[2*i_kr+i_pred_dim,
                                 i_seg_samp] = score_  # round(score_, 2)

            r"Exp vector for sample is the difference of z-scores. \
            Only dominant class score change was recorded."
            i_out_dominant = out_mask[i_seg_samp]
            # log(f"i_out_dominant: {i_out_dominant}")
            delta_z = raw_preds[i_out_dominant, i_seg_samp] \
                - raw_preds_modify[i_out_dominant, i_seg_samp]
            # delta_z = round(delta_z, 2)
            # log(f"kr{i_kr}, samp:{i_seg_samp}, delta_z:{delta_z}")
            explanation_vector[i_kr, i_seg_samp] = delta_z

            r"Record pred-mask and modified pred-mask."
            pred_mask_vec[i_kr, i_seg_samp] = out_mask[i_seg_samp]
            pred_mask_changed_vec[i_kr,
                                  i_seg_samp] = out_mask_modify[i_seg_samp]
            # break
    r"Create result DataFrame."
    data_dict = {
        'signal': segment.values,
        # 'exp_vec': explanation_vector
    }
    for i_kr in range(n_conv0_filter):
        data_dict.update({
            f"exp_vec_kr{i_kr}": explanation_vector[i_kr],
            f"out_mask_kr{i_kr}": pred_mask_vec[i_kr],
            f"u_out_mask_kr{i_kr}": pred_mask_changed_vec[i_kr],
        })
        for i_pred_dim in range(pred_dim):
            data_dict.update({
                f"kr{i_kr}_out{i_pred_dim}": z_ref_scores[2*i_kr+i_pred_dim],
                f"u_kr{i_kr}_out{i_pred_dim}": z_changed_scores[2*i_kr+i_pred_dim],
            })
    return pd.DataFrame(data_dict), low_hz, band_hz


def alter_filter(
    seg_file=None, subject=None, in_memory=False
):
    r"""Predict segment."""
    df = pd.read_csv(seg_file, skiprows=0)
    segment = df['signal']
    log(f"[{subject}] signal: {segment.shape}, path:{seg_file}")

    model = load_model()

    df_list = []
    for i_kr in range(-1, CONV0_OUT_CHAN):
        # if i_kr < 0:
        #     idx_zero_kernel = [i_kr]
        # else:
        #     idx_zero_kernel = [i for i in range(CONV0_OUT_CHAN)]
        #     idx_zero_kernel.remove(i_kr)
        idx_zero_kernel = [i_kr]
        raw_preds, out_mask, out_layers = seg_predict(
            model, segment.values, i_zero_kernels=idx_zero_kernel)

        idx_seg = (lambda x: x[:x.find('.debug')])(seg_file.split('/')[-1])
        seg_file_parent = seg_file[:seg_file.rfind('/')]
        ret_df = debug_data(
            model, df['signal'], df['label'], out_mask, preds_raw=raw_preds,
            out_layers=out_layers, rec_name=subject, i_seg=idx_seg,
            debug_path=f"{seg_file_parent}/debug_filter",
            file_suffix=f"ikzero{'-'.join(str(_) for _ in idx_zero_kernel)}",
            in_memory=in_memory)
        if in_memory:
            df_list.append(ret_df)
        # break
    return {
        'data': df_list,
        'low_hz': model.conv[0].low_hz_.data.detach().cpu().view(-1).numpy(),
        'band_hz': model.conv[0].band_hz_.data.detach().cpu().view(-1).numpy(),
    }


def run():
    r"""Run simulation."""
    sim_dir = f"{BASE_PATH}/debug"

    hz_info = None
    # sim_name = None
    for f in os.listdir(sim_dir):
        if f.endswith(".fband"):
            # sim_name = f[:f.find('.fband')]
            with open(os.path.join(sim_dir, f)) as f:
                lines = f.readlines()
                hz_info = lines[-1]
                log(f"f1, band: {hz_info}")
    r"Iterate validation folder segments."
    validation_debug = os.path.join(
        BASE_PATH, 'validate', VALIDATION_FOLDER, 'debug')
    for f in os.listdir(validation_debug):
        f_sub = os.path.join(validation_debug, f)
        if not os.path.isdir(f_sub):
            continue
        for ff in os.listdir(f_sub):
            f_sub_seg = os.path.join(f_sub, ff)
            if not os.path.isfile(f_sub_seg):
                continue
            log(f"seg_file: {f_sub_seg}")
            if (f_sub_seg.endswith(".debug")):
                log(f"Subject:{f}...")

                # alter_filter(subject=f, seg_file=f_sub_seg)

                data, _, _ = alter_filter_items_file(
                    base_path=BASE_PATH, seg_file=f_sub_seg[len(BASE_PATH):],
                    n_conv0_filter=CONV0_OUT_CHAN,
                    data_seg_sec=DATA_SEG_SEC)
                log(f"Ret exp. vector:{data}")
                debug_file = f"{f_sub_seg.replace('.debug', '.debug_exp')}"
                data.to_csv(debug_file, index=True, mode="w")

                break
        break


if __name__ == '__main__':
    # model_training.fix_randomness()

    CONV0_OUT_CHAN = 4*2
    N_LAYER = 2
    KERNEL_SZ = 7
    SINC_CONV0 = True

    START_TIME_TAG = f"{datetime.now():%Y%m%d%H%M%S}"
    Hz = 100
    DATA_SEG_SEC = 10
    SEG_LEN = Hz * DATA_SEG_SEC
    IN_CHAN = 1
    DEVICE = 'cpu'

    HOME = "/home/mahabib"
    BASE_PATH = f"{HOME}/py_code/runs/sincnet_20211127/train_sincnet.py_100Hz_trainincartdb_L2_SincNet_sinconv0True_cfg8xAx8xA_kr7x7_krgr1x8_20211225131609"
    LOG_PATH = (
        f"{BASE_PATH}/{sys.argv[0]}/{START_TIME_TAG}")
    VALIDATION_FOLDER = 'val_incartdb_20210922134359'
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    # LOG_FILE = f"debug_{create_model().name()}.log"

    run()

    # model = load_model()
    # print(model)
    # low_hz_ = model.conv[0].low_hz_.data.detach().cpu().view(-1).numpy()
    # print("low_hz_", low_hz_.shape, low_hz_)
    # print("low_hz_", model.conv[0].band_hz_.data.detach().cpu().numpy())
