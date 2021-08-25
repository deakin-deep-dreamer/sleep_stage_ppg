
import math
import torch
from torch import nn
import torch.nn.functional as F


def padding_same(input,  kernel, stride=1, dilation=1):
    r"""Calculates padding for applied dilation."""
    print(f"[padding_same] input:{input}, kr:{kernel}, stride:{stride}")
    return int(0.5 * (stride * (input - 1) - input + kernel + (dilation - 1) * (kernel - 1)))


class PreconvConv1d(nn.Module):
    r"""
    Convolution layer with an additional channel-growth controlling
    pre-convolution layer.
    """

    def __init__(
        self, input_size=None, in_channels=3, conv_kernel=5, conv_stride=1,
        dilation_factor=1, out_channels=1, n_growth_rate=1
    ):
        super(PreconvConv1d, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.dilation_factor = dilation_factor
        self.out_channels = out_channels
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.n_growth_rate = n_growth_rate
        self.pre_conv_spatial_scaling = self.make_pre_conv_layer()
        self.conv_layer = self.make_conv_layer()

    def name(self):
        return f'{self.__class__.__name__}'

    def forward(self, x):
        out = self.pre_conv_spatial_scaling(x)
        out = self.conv_layer(out)
        return out

    def make_pre_conv_layer(self):
        layers = []
        self.pre_conv_out_channels = int(self.n_growth_rate*self.out_channels)
        layers += [
            nn.Conv1d(
                self.in_channels,
                self.pre_conv_out_channels,
                kernel_size=1, stride=1,
                padding=0, bias=False
            ),
            nn.BatchNorm1d(self.pre_conv_out_channels),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def make_conv_layer(self):
        conv1d = nn.Conv1d(
            self.pre_conv_out_channels,
            self.out_channels,
            kernel_size=self.conv_kernel,
            stride=self.conv_stride,
            padding=padding_same(
                input=self.input_size,
                kernel=self.conv_kernel,
                stride=self.conv_stride,
                dilation=self.dilation_factor),
            dilation=self.dilation_factor,
            bias=False)
        return conv1d


class DenseBlock(nn.Module):
    def __init__(
        self, input_size=None, in_channels=None, kernels=None,
        layer_dilations=None, channel_per_kernel=None, n_blocks=2,
        n_growth_rate=4, log=print
    ):
        super(DenseBlock, self).__init__()
        self.log = log
        self.iter = 0
        self.input_size = input_size
        self.in_channels = in_channels
        self.kernels = kernels
        self.layer_dilations = layer_dilations
        self.channel_per_kernel = channel_per_kernel

        self.conv_blocks = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.act = nn.ModuleList([])

        out_channels = self.in_channels
        for i_block in range(n_blocks):
            self.bns.append(nn.BatchNorm1d(out_channels))
            self.act.append(nn.ReLU(inplace=True))
            self.conv_blocks.append(
                PreconvConv1d(
                    input_size=input_size,
                    in_channels=in_channels,
                    conv_kernel=kernels[0],
                    dilation_factor=layer_dilations[0],
                    out_channels=channel_per_kernel,
                    n_growth_rate=n_growth_rate,
                ))
            out_channels += len(kernels) * \
                len(layer_dilations)*channel_per_kernel
            in_channels = out_channels

    def name(self):
        return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f'input: {x.shape}')
        out = x

        prev_features = [x]
        for i, conv_ in enumerate(self.conv_blocks):
            self.debug(f'conv_block {i}: {out.shape}')
            out = conv_(self.act[i](self.bns[i](out)))
            prev_features.append(out)
            out = torch.cat(prev_features, 1)

        self.debug(f'dense cat: { out.shape}')

        self.iter += 1
        return out

    def debug(self, *args):
        if self.iter == 0:
            self.log(f"[{self.name()}] {args}")


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, pool_kr=2):
        super(TransitionBlock, self).__init__()
        self.pool_kr = pool_kr
        self.bn = nn.BatchNorm1d(in_planes)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)

    def name(self):
        return f'{self.__class__.__name__}'

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.max_pool1d(out, kernel_size=self.pool_kr, stride=self.pool_kr)
        return out


class DenseNet(nn.Module):
    def __init__(
        self, input_size=None, in_channels=1, kernels=None,
        layer_dilations=None, channel_per_kernel=None,
        n_hidden=None, n_classes=2, low_conv_cfg=None,
        low_conv_kernels=None, low_conv_strides=None,
        low_conv_pooling_kernels=None, n_blocks=None, reduction=0.5,
        n_growth_rate=4, transition_pooling=None, log=print,
        skip_final_transition_blk=False
    ):
        super(DenseNet, self).__init__()
        self.log = log
        self.iter = 0
        self.out_channels = 0
        self.current_input_sz = self.input_size = input_size
        self.in_channels = in_channels
        self.kernels = kernels
        self.n_classes = n_classes
        self.skip_final_transition_blk = skip_final_transition_blk
        '''Dilation factors per block'''
        self.layer_dilations = layer_dilations
        '''A single channel volume per kernel per block'''
        self.channel_per_kernel = channel_per_kernel
        self.n_blocks = n_blocks

        self.low_conv_cfg = low_conv_cfg
        self.low_conv_kernel = low_conv_kernels
        self.low_conv_pooling_kernels = low_conv_pooling_kernels
        self.low_conv_strides = low_conv_strides
        if low_conv_strides is None:
            self.low_conv_strides = [1 for x in self.low_conv_cfg if x != 'M']
        if low_conv_pooling_kernels is None:
            self.low_conv_pooling_kernels = [
                2 for x in self.low_conv_cfg if x == 'M']

        self.transition_pooling = transition_pooling

        self.low_conv = self.make_low_conv()

        self.dense_blocks = nn.ModuleList([])
        self.transitions = nn.ModuleList([])
        in_channels = self.low_conv_cfg[-1] if isinstance(
            self.low_conv_cfg[-1], int) else self.low_conv_cfg[-2]
        for i_block in range(len(self.layer_dilations)):
            self.dense_blocks.append(
                DenseBlock(
                    input_size=input_size,
                    in_channels=in_channels,
                    kernels=kernels[i_block],
                    layer_dilations=layer_dilations[i_block],
                    channel_per_kernel=channel_per_kernel[i_block],
                    n_blocks=n_blocks[i_block],
                    n_growth_rate=n_growth_rate
                )
            )
            in_channels += (
                n_blocks[i_block] * len(layer_dilations[i_block])
                * len(kernels[i_block]) * channel_per_kernel[i_block])

            if self.skip_final_transition_blk \
                    and i_block == len(self.layer_dilations) - 1:
                r"Skip transition-block in final layer."
                out_channels = in_channels
                break

            out_channels = int(math.floor(in_channels*reduction))
            self.debug(f'Transition pooling: {self.transition_pooling}')
            self.transitions.append(
                TransitionBlock(
                    in_channels, out_channels,
                    pool_kr=self.transition_pooling[i_block]
                )
            )
            in_channels = out_channels

        self.classifier = nn.Linear(out_channels, n_classes)

        # n_hidden = self.current_input_sz * out_channels
        # self.classifier = nn.Sequential(
        #     nn.Linear(n_hidden, n_hidden//2),
        #     nn.BatchNorm1d(n_hidden//2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(n_hidden//2, n_classes)
        # )

        # self.dropout = nn.Dropout(0.4)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def name(self):
        return f'{self.__class__.__name__}'

    def forward(
        self, x
    ):
        self.debug(f'input: {x.shape}')

        out = self.low_conv(x)
        self.debug(f'low conv: {out.shape}')

        for i_block in range(len(self.dense_blocks)):
            out = self.dense_blocks[i_block](out)
            self.debug(f'dense-{i_block}: {out.shape}')
            if self.skip_final_transition_blk \
                    and i_block == len(self.dense_blocks) - 1:
                break
            out = self.transitions[i_block](out)
            self.debug(f'transition-{i_block}: {out.shape}')

        self.debug(f'transition out: {out.shape}')

        out = self.gap(out)
        self.debug(f"gap out: {out.shape}")

        # out = self.dropout(out)

        out = out.view(out.size(0), -1)
        self.debug(f'flat: {out.shape}')

        out = self.classifier(out)
        self.debug(f'classifier out: {out.shape}')

        self.iter += 1
        return out

    def make_low_conv(self):
        layers = []
        in_channels = self.in_channels
        i_conv_layer = 0
        i_pooling = 0
        for x in self.low_conv_cfg:
            if x == 'M':
                pooling_kr = self.low_conv_pooling_kernels[i_pooling]
                layers += [nn.MaxPool1d(kernel_size=pooling_kr,
                                        stride=pooling_kr)]
                i_pooling += 1
            else:
                conv_zero_padd = int(
                    (self.low_conv_kernel[i_conv_layer] - 1) // 2)
                layers += [
                    nn.Conv1d(
                        in_channels, x,
                        kernel_size=self.low_conv_kernel[i_conv_layer],
                        padding=conv_zero_padd,
                        stride=self.low_conv_strides[i_conv_layer]),
                    nn.BatchNorm1d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
                i_conv_layer += 1
        return nn.Sequential(*layers)

    def debug(self, *args):
        if self.iter == 0:
            self.log(self.__class__.__name__, args)


if __name__ == '__main__':
    Hz = 128 // 2
    SEG_SEC = 30
    NUM_CLASSES = 2
    IN_CHAN = 1
    N_DENSE_BLOCK = 4

    model = DenseNet(
        input_size=Hz*SEG_SEC, in_channels=IN_CHAN,
        n_classes=NUM_CLASSES, skip_final_transition_blk=True,
        kernels=[
            [5] for _ in range(N_DENSE_BLOCK)
        ],
        layer_dilations=[
            [1] for _ in range(N_DENSE_BLOCK)
        ],
        channel_per_kernel=[
            32 for _ in range(N_DENSE_BLOCK)
        ],
        n_blocks=[
            2 for _ in range(N_DENSE_BLOCK)
        ],
        low_conv_cfg=[32, 32, 'M'], low_conv_kernels=[21, 21],
        low_conv_strides=[5, 1], low_conv_pooling_kernels=[2],
        transition_pooling=[2 for _ in range(N_DENSE_BLOCK)]
    )

    print(model)

    x = torch.randn(1, 1, Hz*SEG_SEC)
    out = model(x)
    print(f"out: {out.shape}")
