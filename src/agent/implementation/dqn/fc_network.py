# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス(全結合型NN)

import torch.nn as nn


# DQN NNクラス(全結合型NN)
class FcNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size):
        super().__init__()

        # 全結合型NNに入力される状態はベクトル(1階テンソル)なので、入力サイズの型をtuple -> intにする
        in_size = in_size[0]

        # ネットワークの作成
        self.network = self._create_network(in_size, out_size)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        return self.network(x)

    # ネットワークを作成する
    def _create_network(self, in_size, out_size):
        network = nn.Sequential()

        network.add_module("fc1", nn.Linear(in_size, 100, bias=True))
        network.add_module("relu1", nn.ReLU())
        network.add_module("fc2", nn.Linear(100, 100, bias=True))
        network.add_module("relu2", nn.ReLU())
        network.add_module("fc3", nn.Linear(100, 100, bias=True))
        network.add_module("relu3", nn.ReLU())
        network.add_module("fc4", nn.Linear(100, out_size, bias=True))
        network.add_module("relu4", nn.ReLU())

        return network
