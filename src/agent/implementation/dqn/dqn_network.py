# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス

import os.path as op

import yaml
import torch.nn as nn
import torch.nn.functional as F

from common.dirs import AGENT_CONFIG_DIR

# DQN NNクラス
class DqnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size, config):
        super().__init__()

        # ネットワークの作成
        structure = self._load_nn_structure(config['nn_structure_file_fc'])
        self.network = self._create_network(structure, in_size, out_size)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        output = self.network(x)

        return output

    # NN構造定義ファイルを読み込む
    def _load_nn_structure(self, filename):
        path = op.join(AGENT_CONFIG_DIR('dqn'), filename)
        with open(path, 'r', encoding='utf-8') as f:
            structure = yaml.safe_load(f)

        return structure

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
