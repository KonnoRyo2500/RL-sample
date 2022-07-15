# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス

import torch.nn as nn

# DQN NNクラス
class DqnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size):
        super().__init__()

        self.network = self._create_network(in_size, out_size)

    # 推論の実行
    def forward(self, x):
        x = self.network(x)

        return x

    # ネットワークの構築
    def _create_network(self, in_size, out_size):
        mid1_size = in_size * 4
        mid2_size = in_size * 4

        # 3層の全結合ネットワーク
        # 活性化関数はReLU
        network = nn.Sequential()
        network.add_module('fc1', nn.Linear(in_features=in_size, out_features=mid1_size))
        network.add_module('act1', nn.ReLU())
        network.add_module('fc2', nn.Linear(in_features=mid1_size, out_features=mid2_size))
        network.add_module('act2', nn.ReLU())
        network.add_module('fc3', nn.Linear(in_features=mid2_size, out_features=out_size))

        return network
