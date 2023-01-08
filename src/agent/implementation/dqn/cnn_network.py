# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス(CNN)

import math

import torch.nn as nn


# DQN NNクラス(CNN)
class CnnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size, structure):
        super().__init__()

        # ネットワークの作成
        self.network = self._create_network(structure, in_size, out_size)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        return self.network(x)

    # NN構造定義と入出力サイズに基づいてネットワークを作成する
    def _create_network(self, structure, in_size, out_size):
        actfunc_from_name = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
        }

        network = nn.Sequential()

        in_dim = len(in_size)
        # 入力が行列(HW)であっても、強制的に3階テンソル(CHW)にする
        if in_dim == 2:
            in_size = (1,) + in_size

        # 畳み込み層、全結合層の総数
        conv_layer_names = [name for name in structure if name.startswith('conv')]
        fc_layer_names = [name for name in structure if name.startswith('fc')]
        n_conv_layers = len(conv_layer_names)
        n_fc_layers = len(fc_layer_names)

        # 畳み込み層の作成
        conv_out_shape = in_size
        for i in range(1, n_conv_layers + 1):
            out_channel, kernel_size, padding, stride, bias, actfunc_name = structure[f'conv_{i}']
            in_channel, in_height, in_width = conv_out_shape

            conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
            actfunc = actfunc_from_name[actfunc_name]()

            out_height = math.floor(((in_height + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0]) + 1)
            out_width = math.floor(((in_width + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1]) + 1)
            conv_out_shape = (out_channel, out_height, out_width)

            network.add_module(f'conv{i}', conv)
            network.add_module(f'conv_actfunc{i}', actfunc)

        # 畳み込み層の出力を全結合層に入力するため、特徴マップを1次元化
        network.add_module('flatten', nn.Flatten())
        fc_in_units = math.prod(conv_out_shape)

        # 全結合層の作成
        for i in range(1, n_fc_layers + 1):
            n_units, bias, actfunc_name = structure[f'fc_{i}']
            in_units = fc_in_units if i == 1 else structure[f'fc_{i - 1}'][0]
            out_units = n_units

            fc = nn.Linear(in_features=in_units, out_features=out_units, bias=bias)
            actfunc = actfunc_from_name[actfunc_name]()

            network.add_module(f'fc{i}', fc)
            network.add_module(f'fc_actfunc{i}', actfunc)

        # 最後の全結合層
        final_fc = nn.Linear(
            in_features=structure[f'fc_{n_fc_layers}'][0],
            out_features=out_size,
            bias=structure[f'fc_{n_fc_layers}'][1])
        network.add_module(f'fc{n_fc_layers + 1}', final_fc)

        return network
