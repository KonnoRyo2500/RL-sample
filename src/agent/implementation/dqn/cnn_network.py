# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス(CNN)

import math

import torch.nn as nn


# DQN NNクラス(CNN)
class CnnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size):
        super().__init__()

        # ネットワークの作成
        self.network = self._create_network(in_size, out_size)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        return self.network(x)

    # ネットワークを作成する
    def _create_network(self, in_size, out_size):
        network = nn.Sequential()

        # 入力が行列(HW)の場合は入力チャンネルは1とする
        if len(in_size) == 2:
            in_channel = 1
            in_width = in_size[0]
            in_height = in_size[1]
        else:
            in_channel = in_size[0]
            in_width = in_size[1]
            in_height = in_size[2]

        # 畳み込み層
        kernel_size = 3
        out_channels = 4
        conv1 = nn.Conv2d(
            kernel_size=kernel_size,
            in_channels=in_channel,
            out_channels=out_channels
        )
        network.add_module("Conv1", conv1)
        network.add_module("conv_relu1", nn.ReLU())

        # 全結合層
        fc_in_features = (in_width - kernel_size + 1) * (in_height - kernel_size + 1) * out_channels
        network.add_module("flatten", nn.Flatten())
        network.add_module("fc1", nn.Linear(in_features=fc_in_features, out_features=30))
        network.add_module("fc_relu1", nn.ReLU())
        network.add_module("fc2", nn.Linear(in_features=30, out_features=out_size))

        return network
