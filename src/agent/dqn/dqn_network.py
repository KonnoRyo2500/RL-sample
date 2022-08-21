# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス

import torch.nn as nn
import torch.nn.functional as F

# DQN NNクラス
class DqnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size):
        super().__init__()

        # ネットワークを作成
        # S -> 100 -> 100 -> 100 -> Aの全結合ネットワーク(S, Aはそれぞれ状態・行動数)
        # 活性化関数はReLU
        mid1_size = 100
        mid2_size = 100
        mid3_size = 100
        self.fc1 = nn.Linear(in_features=in_size, out_features=mid1_size)
        self.fc2 = nn.Linear(in_features=mid1_size, out_features=mid2_size)
        self.fc3 = nn.Linear(in_features=mid2_size, out_features=mid3_size)
        self.fc4 = nn.Linear(in_features=mid3_size, out_features=out_size)
        self.relu = nn.ReLU()

    # 推論(順伝播計算)の実行
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)

        return x
