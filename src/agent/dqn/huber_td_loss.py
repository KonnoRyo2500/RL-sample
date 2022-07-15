# 強化学習勉強用サンプルプログラム DQNエージェント用損失関数クラス

import torch.nn as nn

# DQN損失関数クラス
# TD誤差を計算し、そのHuber誤差をとる
class HuberTDLoss(nn.Module):
    # コンストラクタ
    def __init__(self):
        super().__init__()

        # Huber誤差
        self.huber_loss = nn.HuberLoss()

    # 損失の計算
    def forward(self, q_values, next_target_q_values, rewards, gamma):
        # 目標となる行動価値の計算
        targets = rewards + gamma * next_target_q_values

        # TD誤差のHuber誤差の計算
        huber_td_loss = self.huber_loss(q_values, targets)

        return huber_td_loss
