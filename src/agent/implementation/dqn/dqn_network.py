# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス

import os.path as op

import torch.nn as nn
import yaml

from agent.implementation.dqn.fc_network import FcNetwork
from agent.implementation.dqn.cnn_network import CnnNetwork
from common.dirs import AGENT_CONFIG_DIR


# DQN NNクラス
class DqnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size, config):
        super().__init__()

        # 設定に応じたNNの作成
        nn_class, structure_definition_file_path = {
            'FC': (FcNetwork, config.nn_structure_file_fc),
            'CNN': (CnnNetwork, config.nn_structure_file_cnn),
        }[config.nn_structure]
        structure = self._load_nn_structure(structure_definition_file_path)
        self.nn = nn_class(in_size, out_size, structure)

    # 推論(順伝播計算)の実行
    def forward(self, x):
        return self.nn(x)

    # NN構造定義ファイルを読み込む
    def _load_nn_structure(self, filename):
        path = op.join(AGENT_CONFIG_DIR('dqn'), filename)
        with open(path, 'r', encoding='utf-8') as f:
            structure = yaml.safe_load(f)

        return structure
