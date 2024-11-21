import math
from collections import OrderedDict

import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, T = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(
            kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False
        ),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="prelu"):
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        # type of ReLU is an input option
        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise Exception("relu type not implemented")
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        relu_type="relu",
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = (
            downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        )

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.ones_(m.weight)
                # nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert (
                self.chomp_size % 2 == 0
            ), "If symmetric chomp, chomp size needs to be even"

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2 : -self.chomp_size // 2].contiguous()
        else:
            return x[:, :, : -self.chomp_size].contiguous()


class TemporalConvLayer(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type
    ):
        super(TemporalConvLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(n_outputs),
            Chomp1d(padding, True),
            nn.PReLU(num_parameters=n_outputs)
            if relu_type == "prelu"
            else Swish()
            if relu_type == "swish"
            else nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _ConvBatchChompRelu(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size_set,
        stride,
        dilation,
        dropout,
        relu_type,
        se_module=False,
    ):
        super(_ConvBatchChompRelu, self).__init__()

        self.num_kernels = len(kernel_size_set)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert (
            n_outputs % self.num_kernels == 0
        ), "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(kernel_size_set):
            if se_module:
                setattr(
                    self, "cbcr0_se_{}".format(k_idx), SELayer(n_inputs, reduction=16)
                )
            cbcr = TemporalConvLayer(
                n_inputs,
                self.n_outputs_branch,
                k,
                stride,
                dilation,
                (k - 1) * dilation,
                relu_type,
            )
            setattr(self, "cbcr0_{}".format(k_idx), cbcr)
        self.dropout0 = nn.Dropout(dropout)
        for k_idx, k in enumerate(kernel_size_set):
            cbcr = TemporalConvLayer(
                n_outputs,
                self.n_outputs_branch,
                k,
                stride,
                dilation,
                (k - 1) * dilation,
                relu_type,
            )
            setattr(self, "cbcr1_{}".format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        self.se_module = se_module
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        # final relu
        if relu_type == "relu":
            self.relu_final = nn.ReLU()
        elif relu_type == "prelu":
            self.relu_final = nn.PReLU(num_parameters=n_outputs)
        elif relu_type == "swish":
            self.relu_final = Swish()

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        x = torch.cat(inputs, 1)
        outputs = []
        for k_idx in range(self.num_kernels):
            if self.se_module:
                branch_se = getattr(self, "cbcr0_se_{}".format(k_idx))
            branch_convs = getattr(self, "cbcr0_{}".format(k_idx))
            if self.se_module:
                outputs.append(branch_convs(branch_se(x)))
            else:
                outputs.append(branch_convs(x))
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0(out0)
        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, "cbcr1_{}".format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)
        # downsample?
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        return bottleneck_output


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers,
        num_input_features,
        growth_rate,
        kernel_size_set,
        dilation_size_set,
        dropout,
        relu_type,
        squeeze_excitation,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            dilation_size = dilation_size_set[i % len(dilation_size_set)]
            layer = _ConvBatchChompRelu(
                n_inputs=num_input_features + i * growth_rate,
                n_outputs=growth_rate,
                kernel_size_set=kernel_size_set,
                stride=1,
                dilation=dilation_size,
                dropout=dropout,
                relu_type=relu_type,
                se_module=squeeze_excitation,
            )

            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, relu_type):
        super(_Transition, self).__init__()
        self.add_module(
            "conv",
            nn.Conv1d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.BatchNorm1d(num_output_features))
        if relu_type == "relu":
            self.add_module("relu", nn.ReLU())
        elif relu_type == "prelu":
            self.add_module("prelu", nn.PReLU(num_parameters=num_output_features))
        elif relu_type == "swish":
            self.add_module("swish", Swish())


class DenseTemporalConvNet(nn.Module):
    def __init__(
        self,
        block_config,
        growth_rate_set,
        input_size,
        reduced_size,
        kernel_size_set,
        dilation_size_set,
        dropout=0.2,
        relu_type="prelu",
        squeeze_excitation=False,
    ):
        super(DenseTemporalConvNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))

        trans = _Transition(
            num_input_features=input_size,
            num_output_features=reduced_size,
            relu_type="prelu",
        )
        self.features.add_module("transition%d" % (0), trans)
        num_features = reduced_size

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate_set[i],
                kernel_size_set=kernel_size_set,
                dilation_size_set=dilation_size_set,
                dropout=dropout,
                relu_type=relu_type,
                squeeze_excitation=squeeze_excitation,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate_set[i]

            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=reduced_size,
                    relu_type=relu_type,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = reduced_size

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm1d(num_features))

    def forward(self, x):
        features = self.features(x)
        return features


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack(
        [torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0
    )


class DenseTCN(nn.Module):
    def __init__(
        self,
        block_config,
        growth_rate_set,
        input_size,
        reduced_size,
        num_classes,
        kernel_size_set,
        dilation_size_set,
        dropout,
        relu_type,
        squeeze_excitation=False,
    ):
        super(DenseTCN, self).__init__()
        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(
            block_config,
            growth_rate_set,
            input_size,
            reduced_size,
            kernel_size_set,
            dilation_size_set,
            dropout=dropout,
            relu_type=relu_type,
            squeeze_excitation=squeeze_excitation,
        )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class Lipreading(nn.Module):
    def __init__(self, densetcn_options={}):
        super(Lipreading, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type="swish")
        frontend_relu = Swish()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.tcn = DenseTCN(
            block_config=densetcn_options["block_config"],
            growth_rate_set=densetcn_options["growth_rate_set"],
            input_size=self.backend_out,
            reduced_size=densetcn_options["reduced_size"],
            num_classes=500,
            kernel_size_set=densetcn_options["kernel_size_set"],
            dilation_size_set=densetcn_options["dilation_size_set"],
            dropout=densetcn_options["dropout"],
            relu_type="swish",
            squeeze_excitation=densetcn_options["squeeze_excitation"],
        )

    def forward(self, x, lengths, boundaries=None):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        return x
