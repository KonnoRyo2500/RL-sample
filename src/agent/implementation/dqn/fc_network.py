# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス(全結合型NN)

import torch.nn as nn


# DQN NNクラス(全結合型NN)
class FcNetwork(nn.Module):
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
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
        }
        network = nn.Sequential()
        n_layers = len(structure)

        for i in range(1, n_layers + 1):
            n_units, bias, actfunc_name = structure[f'layer_{i}']
            in_units = in_size if i == 1 else structure[f'layer_{i - 1}'][0]
            out_units = n_units

            layer_i = nn.Linear(in_features=in_units, out_features=out_units, bias=bias)
            actfunc_i = actfunc_from_name[actfunc_name]

            network.add_module(f'fc{i}', layer_i)
            network.add_module(f'act{i}', actfunc_i)

        layer_n = nn.Linear(
            in_features=structure[f'layer_{n_layers}'][0],
            out_features=out_size,
            bias=structure[f'layer_{n_layers}'][1])
        network.add_module(f'fc{n_layers + 1}', layer_n)

        return network
