# 強化学習勉強用サンプルプログラム DQNエージェント用NNクラス(CNN)

import torch.nn as nn


# DQN NNクラス(CNN)
class CnnNetwork(nn.Module):
    # コンストラクタ
    def __init__(self, in_size, out_size, config):
        super().__init__()
