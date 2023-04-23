# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス

import os.path as op

import torch.nn as nn
import yaml

from agent.implementation.dqn.fc_network import FcNetwork
from agent.implementation.dqn.cnn_network import CnnNetwork


# DQN NNクラス
class DqnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size, config):
        super().__init__()

        # 設定に応じたNNの作成
        nn_class = {
            'FC': FcNetwork,
            'CNN': CnnNetwork,
        }[config.nn_type]
        self.nn = nn_class(in_size, out_size)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        return self.nn(x)
