
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math


def model_cfg_str(
    options
):
    r"""Config to string."""
    return (
        f"sinconv0{options.get('sinc_conv0')}_"
        f"cfg{'x'.join(str(x) for x in options.get('cfg'))}_"
        f"kr{'x'.join(str(x) for x in options.get('kernel'))}_"
        f"krgr{'x'.join(str(x) for x in options.get('kernel_group'))}")


def padding_same(input,  kernel, stride=1, dilation=1):
    """
        Calculates padding for applied dilation.
    """
    return int(0.5 * (stride * (input - 1) - input + kernel + (dilation - 1) * (kernel - 1)))


def flip(x, dim):
    r"""Flip function."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    print(f"[flip] x:{x.size()}, {x}")
    x = x.view(x.size(0), x.size(
        1), -1)[:getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band, t_right):
    r"""sinc function."""
    y_right = torch.sin(2*math.pi*band*t_right) / (2*math.pi*band*t_right)
    print(f"[sinc] y_right({y_right.shape}): {y_right}")
    y_left = flip(y_right, 0)
    # y_left = torch.flip(y_right, dims=[1])
    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
    return y


class SincConv1d_fast(nn.Module):
    r"""Sinc convolution layer."""

    def __init__(
        self, in_channels=None, out_channels=None, kernel_dim=None, hz=None,
        stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=2,
        min_band_hz=1, init_band_hz=2.0, fq_band_lists=None, log=print
    ):
        r"""Instantiate class."""
        super(SincConv1d_fast, self).__init__()

        if in_channels != 1:
            raise ValueError(
                f"SincConv1d supports one input channel, found {in_channels}.")
        if groups != 1:
            raise ValueError(
                f"SincConv1d supports single group, found {groups}.")
        if stride > 1 and padding > 0:
            raise ValueError(
                f"Padding should be 0 for >1 stride, found {padding}.")

        self.iter = 0
        self.log = log
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim

        r"Force filters to be odd."
        if kernel_dim % 2 == 0:
            self.kernel_dim += 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if bias:
            raise ValueError("SincConv1d does not support bias.")
        if groups > 1:
            raise ValueError("SincConv1d does not support groups.")

        self.fs = hz
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        # self.max_band_hz = max_band_hz

        r"Initialise filter-banks equally spaced in ECG frequency range."
        low_hz = self.min_low_hz

        if fq_band_lists:
            # print(
            #     f"[sinc-conv] low_hz:{fq_band_lists[0].shape}"
            #     f"band_hz:{fq_band_lists[-1].shape}")
            assert len(fq_band_lists[0]) == len(fq_band_lists[-1])
            self.low_hz_ = nn.Parameter(
                torch.Tensor(np.array(fq_band_lists[0])).view(-1, 1))
            self.band_hz_ = nn.Parameter(
                torch.Tensor(np.array(fq_band_lists[-1])).view(-1, 1))
            r"Adjust out channels based on supplied fq lists."
            self.out_channels = len(fq_band_lists[0])
        else:
            # high_hz = low_hz + self.hz - self.fs//4
            # high_hz = low_hz + self.fs - min_band_hz
            high_hz = low_hz + self.fs/2.0

            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)
            # hz = np.tile(min_low_hz, self.out_channels)
            # hz = np.random.uniform(low_hz, high_hz, self.out_channels+1)

            # hz = np.random.uniform(
            #     low_hz, high_hz, self.out_channels + 1)
            # hz = np.tile(np.array([self.min_low_hz]), self.out_channels + 1)

            r"Filter low frequency (out_channels + 1)"
            self.low_hz_ = nn.Parameter(
                torch.Tensor(np.sort(hz[:-1])).view(-1, 1))
            # self.low_hz_ = nn.Parameter(torch.Tensor(hz).view(-1, 1))
            # self.low_hz_ = nn.Parameter(torch.Tensor(hz).view(-1, 1))

            r"Filter frequency bank (out_channels, 1)"
            self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
            # self.band_hz_ = nn.Parameter(
            #     torch.Tensor(np.tile(high_hz, self.out_channels)).view(-1, 1))
            # self.band_hz_ = nn.Parameter(
            #     torch.Tensor(np.tile(init_band_hz, self.out_channels)).view(-1, 1))

        r"Hamming window"
        # self.window_ = torch.hamming_window(self.kernel_dim)
        n_lin = torch.linspace(
            0, (self.kernel_dim/2) - 1, steps=self.kernel_dim//2)  # half-window
        self.window_ = 0.54 - 0.46 * \
            torch.cos(2 * math.pi * n_lin/self.kernel_dim)

        # (1, kernel_dim/2)
        n = (self.kernel_dim - 1) / 2.0
        r"Due to symmetry, I only need half of the time axes"
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.fs

        # self.debug(
        #     f"n_:{self.n_.data.detach().cpu().view(-1)}, "
        #     f"low_hz_:{self.low_hz_.data.detach().cpu().view(-1)}, "
        #     f"band_hz_:{self.band_hz_.data.detach().cpu().view(-1)}")

    def name(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kernel_dim},), stride=({self.stride},), "
                f"padding=({self.padding},))"
                )

    def forward(self, waveforms):
        r"""
        Parameters
        ----------
        waveforms : 'torch.Tensor' (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : 'torch.Tensor' (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.debug(f"in: {waveforms.shape}")

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        # self.debug(f"low_hz_:{self.low_hz_},\nband_hz_:{self.band_hz_}")

        low = self.min_low_hz + torch.abs(self.low_hz_)
        # low = torch.clamp(
        #     low, self.min_low_hz, self.fs)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.fs/2.0)
        band = (high-low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        r"Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM \
        RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified \
        the terms. This way I avoid several useless computations."
        band_pass_left = ((torch.sin(f_times_t_high)
                           - torch.sin(f_times_t_low)) / (self.n_/2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([
            band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2*band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_dim)

        self.iter += 1

        return F.conv1d(
            waveforms, self.filters, stride=self.stride, padding=self.padding,
            dilation=self.dilation, bias=None, groups=self.groups
        )

    def debug(self, *args):
        if self.iter == 0 and self.log:
            self.log(f"[{self.name()}] {args}")


class SincConv1d(nn.Module):
    r"""Sinc convolution layer."""

    def __init__(
        self, in_channels=None, out_channels=None, kernel_dim=None, hz=None,
        stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=4.0,
        min_band_hz=2.0, max_band_hz=20.0, log=print
    ):
        r"""Instantiate class."""
        super(SincConv1d, self).__init__()

        if in_channels != 1:
            raise ValueError(
                f"SincConv1d supports one input channel, found {in_channels}")

        self.iter = 0
        self.log = log
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dim = kernel_dim

        r"Force filters to be odd."
        if kernel_dim % 2 == 0:
            self.kernel_dim += 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if bias:
            raise ValueError("SincConv1d does not support bias.")
        if groups > 1:
            raise ValueError("SincConv1d does not support groups.")

        self.fs = hz*1.0
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        r"Initialise filter-banks equally spaced in ECG frequency range."
        low_hz = self.min_low_hz
        high_hz = low_hz + self.fs/2.0

        hz = np.random.randint(low_hz, high_hz, self.out_channels)

        r"Filter low frequency (out_channels + 1)"
        # self.low_hz_ = nn.Parameter(torch.Tensor(hz).view(-1, 1))
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz/self.fs))

        r"Filter frequency bank (out_channels, 1)"
        # self.band_hz_ = nn.Parameter(
        #     torch.Tensor(np.tile(max_band_hz, self.out_channels)).view(-1, 1))
        self.band_hz_ = nn.Parameter(
            torch.from_numpy(np.tile(max_band_hz, self.out_channels)/self.fs))

        r"Hamming window"
        self.window_ = torch.hamming_window(self.kernel_dim)

        self.filters_ = Variable(torch.zeros(
            (self.out_channels, self.kernel_dim)))
        self.t_right_ = Variable(torch.linspace(
            1, (self.kernel_dim-1)/2, steps=int((self.kernel_dim-1)/2))/self.fs)

        self.debug(
            f"low_hz_:{self.low_hz_}, band_hz_:{self.band_hz_}")

    def name(self):
        return (f"{self.__class__.__name__}"
                f"_in{self.in_channels}_"
                f"out{self.out_channels}_"
                )

    def forward(self, waveforms):
        r"""
        Parameters
        ----------
        waveforms : 'torch.Tensor' (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : 'torch.Tensor' (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.debug(f"in: {waveforms.shape}")

        self.filters_ = self.filters_.to(waveforms.device)
        self.t_right_ = self.t_right_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        filt_beg_freq = torch.abs(self.low_hz_) + self.min_low_hz/self.fs
        filt_end_freq = filt_beg_freq \
            + (torch.abs(self.band_hz_)+self.min_band_hz/self.fs)

        for i in range(self.out_channels):
            low_pass1 = 2 *\
                filt_beg_freq[i].float()*sinc(
                    filt_beg_freq[i].float()*self.fs, self.t_right_)
            low_pass2 = 2 *\
                filt_end_freq[i].float()*sinc(
                    filt_end_freq[i].float()*self.fs, self.t_right)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / torch.max(band_pass)
            self.filters[i, :] = band_pass.to(waveforms.device)*self.window_

        self.iter += 1

        return F.conv1d(
            waveforms,
            self.filters.view(self.out_channels, 1, self.kernel_dim),
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, bias=None, groups=self.groups)

    def debug(self, *args):
        if self.iter == 0:
            self.log(f"[{self.name()}] {args}")


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
        self.gap = nn.AdaptiveAvgPool1d(1)
        if options.get('flatten'):
            r"Linear decision layer."
            self.classifier = nn.Linear(self.hidden_dim, options["n_class"])
        else:
            r"Point conv-layer as decision layer."
            self.classifier = nn.Conv1d(
                in_channels=self.hidden_dim, out_channels=options["n_class"],
                kernel_size=1, padding=0, dilation=1)

    def name(self):
        r"Format cfg meaningfully."
        return (
            f"{self.__class__.__name__}_"
            f"{model_cfg_str(self.options)}")

    def forward(self, x, i_zero_kernels=None):
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

                if i_conv == 0 and i_zero_kernels:
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

        if self.options.get('gap_layer'):
            out = self.gap(out)
            self.debug(f"  GAP out: {out.shape}")

        if self.options.get('flatten'):
            out = out.view(out.size(0), -1)
            self.debug(f'  flatten: {out.shape}')

        out = self.classifier(out)
        self.debug(f"classif: {out.shape}")

        self.iter += 1
        return out, layer_out

    def create_encoder(
        self
    ):
        # layers = []
        # count_pooling = 0
        i_kernel = 0
        in_chan = 1
        input_sz = self.options["input_sz"]
        for x in self.options["cfg"]:
            if x == 'M':
                # count_pooling += 1
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
                    conv = SincConv1d_fast(
                        in_channels=in_chan, out_channels=x,
                        kernel_dim=self.options["kernel"][i_kernel],
                        hz=self.options["hz"],
                        log=self.log,
                        fq_band_lists=self.options.get('fq_band_lists'),
                        padding=padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1))
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan, out_channels=x,
                        kernel_size=self.options["kernel"][i_kernel],
                        groups=self.options["kernel_group"][i_kernel],
                        padding=padding_same(
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


class ShortcutBlock(nn.Module):
    '''Pass a Sequence and add identity shortcut, following a ReLU.'''

    def __init__(
        self, layers=None, in_channels=None, out_channels=None,
        point_conv_group=1
    ):
        super(ShortcutBlock, self).__init__()
        self.iter = 0
        self.layers = layers
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, bias=False, groups=point_conv_group),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # self.debug(f'input: {x.shape}')

        out = self.layers(x)
        # self.debug(f'layers out: {out.shape}')

        out += self.shortcut(x)
        # self.debug(f'shortcut out: {out.shape}')

        out = F.relu(out)

        self.iter += 1
        return out

    def debug(self, *args):
        if self.iter == 0:
            print(self.__class__.__name__, args)


class SincResNet(nn.Module):
    r"""Sinc convolution with optional residual connections."""

    def __init__(
        self, segment_sz, kernels=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        n_classes=None, low_conv_options=None, shortcut_conn=False, log=print
    ):
        r"""Instance of convnet."""
        super(SincResNet, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        self.make_layers_deep()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self._out_channels, n_classes)

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"sinc{self.low_conv_options.get('sinc_kernel')}_"
            f"sinckr{self.low_conv_options.get('kernel')[0] if self.low_conv_options.get('sinc_kernel') else 0}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"cg{'x'.join(str(x) for x in self.conv_groups)}"
            )
        # return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f'  input: {x.shape}')

        x = self.input_bn(x)

        out = self.low_conv(x)
        self.debug(f'  low_conv out: {out.shape}')

        # out = self.features(x)
        # self.debug(f'features out: {out.shape}')

        for i_blk in range(self.n_blocks):
            if self.shortcut_conn:
                out = self.shortcut[i_blk](out)
                self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
            else:
                for i_conv_blk in range(self.n_conv_layers_per_block):
                    idx_flat = 2*i_blk+i_conv_blk
                    self.debug(
                        f"  block({i_blk}) conv({i_conv_blk}) {self.conv[idx_flat]}")
                    out = self.conv[idx_flat](out)
                    out = self.bn[idx_flat](out)
                    out = self.act[idx_flat](out)
                self.debug(
                    f"  block({i_blk}) out:{out.shape}, data:{out.detach().cpu()[0, 0, :10]}")
            r"One less pooling layer."
            if i_blk < self.n_blocks - 1:
                out = self.pool[i_blk](out)
                self.debug(f"  block({i_blk}) pool-out:{out.shape}")

        out = self.gap(out)
        self.debug(f"  GAP out: {out.shape}")

        out = out.view(out.size(0), -1)
        self.debug(f'  flatten: {out.shape}')
        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        # layers = []
        # in_channels = self.in_channels
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.kernels[i],
                            stride=1,
                            dilation=1)
                    ))
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(nn.ReLU(inplace=True))
                if self.shortcut_conn:
                    layers_for_shortcut.extend([
                        self.conv[-1], self.bn[-1],
                        # self.act[-1]
                    ])
                # layers += [
                #     nn.Conv1d(
                #         in_channels,
                #         out_channels,
                #         kernel_size=self.kernels[i],
                #         groups=self.conv_groups[i],
                #         padding=commons.padding_same(
                #             input=self.input_sz,
                #             kernel=self.kernels[i],
                #             stride=1,
                #             dilation=1)
                #     ),
                #     nn.BatchNorm1d(out_channels),
                #     nn.ReLU(inplace=True)
                # ]
                in_channels = self._out_channels
            # layers += [
            #     nn.AvgPool1d(2, stride=2)
            # ]

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
                )
            if i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        # return nn.Sequential(*layers)
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

    def make_low_conv(self):
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        # input_sz = self.input_sz
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                padding_ = 0 if stride_ > 1 else padding_same(
                    input=self.input_sz,
                    kernel=self.low_conv_options["kernel"][i_kernel],
                    stride=stride_,
                    dilation=1)
                if i_kernel == 0 and self.low_conv_options.get("sinc_kernel"):
                    conv = SincConv1d_fast(
                        in_channels=in_chan, out_channels=x,
                        kernel_dim=self.low_conv_options["kernel"][i_kernel],
                        stride=stride_,
                        hz=self.low_conv_options["hz"],
                        log=self.log,
                        fq_band_lists=self.low_conv_options.get(
                            'fq_band_lists'),
                        padding=padding_
                        )
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan, out_channels=x,
                        kernel_size=self.low_conv_options["kernel"][i_kernel],
                        groups=self.low_conv_options["conv_groups"][i_kernel],
                        stride=stride_,
                        padding=padding_same(
                            input=self.input_sz,
                            kernel=self.low_conv_options["kernel"][i_kernel],
                            stride=stride_,
                            dilation=1))
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    nn.ReLU(inplace=True)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")
            # self.log(f"[{self.__str__()}] {args}")


class SincNetZero(nn.Module):
    r"""SincNet with zero kernel feature."""

    def __init__(self, options, log=print):
        super(SincNetZero, self).__init__()

        self.iter = 0
        self.log = log
        self.options = options

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])

        # self.feature = self.create_encoder()
        self.create_encoder()
        self.gap = nn.AdaptiveAvgPool1d(1)
        if options.get('flatten'):
            r"Linear decision layer."
            self.classifier = nn.Linear(self.hidden_dim, options["n_class"])
        else:
            r"Point conv-layer as decision layer."
            self.classifier = nn.Conv1d(
                in_channels=self.hidden_dim, out_channels=options["n_class"],
                kernel_size=1, padding=0, dilation=1)

    def name(self):
        r"Format cfg meaningfully."
        return (
            f"{self.__class__.__name__}_"
            f"{model_cfg_str(self.options)}")

    def forward(self, x, i_zero_kernels=None):
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

                if i_zero_kernels:
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

        if self.options.get('gap_layer'):
            out = self.gap(out)
            self.debug(f"  GAP out: {out.shape}")

        if self.options.get('flatten'):
            out = out.view(out.size(0), -1)
            self.debug(f'  flatten: {out.shape}')

        out = self.classifier(out)
        self.debug(f"classif: {out.shape}")

        self.iter += 1
        return out, layer_out

    def create_encoder(self):
        # layers = []
        # count_pooling = 0
        i_kernel = 0
        in_chan = 1
        input_sz = self.options["input_sz"]
        for x in self.options["cfg"]:
            if x == 'M':
                # count_pooling += 1
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
                    conv = SincConv1d_fast(
                        in_channels=in_chan, out_channels=x,
                        kernel_dim=self.options["kernel"][i_kernel],
                        hz=self.options["hz"],
                        log=self.log,
                        fq_band_lists=self.options.get('fq_band_lists'),
                        padding=padding_same(
                            input=input_sz,
                            kernel=self.options["kernel"][i_kernel],
                            stride=1,
                            dilation=1))
                else:
                    conv = nn.Conv1d(
                        in_channels=in_chan, out_channels=x,
                        kernel_size=self.options["kernel"][i_kernel],
                        groups=self.options["kernel_group"][i_kernel],
                        padding=padding_same(
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


def test_sincResnet(
    IN_CHAN=1, N_BLOCK=5, NUM_CLASSES=2, Hz=100, LOW_CONV_STRIDE=2,
    low_conv_chan=8
):
    r"Test sincResnet."
    low_hz = np.linspace(2.0, 60.0, low_conv_chan+1)
    band_hz = np.tile(4, low_conv_chan)    # narrow band of 2
    model = SincResNet(
        1000, shortcut_conn=True,
        in_channels=IN_CHAN,
        kernels=[21, 21, 11, 7, 5],
        out_channels=[low_conv_chan*(2**i) for i in range(1, N_BLOCK+1)],
        conv_groups=[1 for i in range(N_BLOCK)],
        n_conv_layers_per_block=2, n_blocks=N_BLOCK, n_classes=NUM_CLASSES,
        low_conv_options={
            "cfg": [low_conv_chan, low_conv_chan, 'M'],
            'stride': [LOW_CONV_STRIDE, 1],
            "kernel": [21, 21],
            'conv_groups': [IN_CHAN, IN_CHAN],
            "sinc_kernel": True,
            "hz": Hz,
            "fq_band_lists": (low_hz[:-1], band_hz),
            },
    )
    print(model)
    x = torch.randn(1, 1, 1000)
    out = model(x)
    print(f"out: {out.shape}")

    print(model.low_conv[0])


def test_sincnet():
    r"Test sincnet"
    options = {
        "cfg": [4, 4],
        "kernel": [5, 5],
        "n_class": 2,
        "input_sz": 300,
        }
    model = SincNet(options)
    print(model)

    x = torch.randn(1, 1, 300)
    out = model(x)
    print(f"out: {out.shape}")

    for p in model.parameters():
        print(p.name, p.data, p.requires_grad)

    for n, p in model.named_parameters():
        print(n, p.data, p.requires_grad)

    print("direct access:", model.feature[0].low_hz_.data)


if __name__ == '__main__':
    # test_sincnet()
    test_sincResnet()
