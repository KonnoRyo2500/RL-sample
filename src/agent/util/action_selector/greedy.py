# 強化学習勉強用サンプルプログラム greedy行動選択クラス

import random

from agent.util.action_selector.action_selector_base import ActionSelectorBase


# greedy行動選択クラス
class Greedy(ActionSelectorBase):
    # コンストラクタ
    def __init__(self, action_space):
        super().__init__(action_space)

    # 行動価値に基づき、選択可能な行動の中から行動を選択する
    def select_action(self, available_actions, q_values):
        # 選択可能行動の各行動が、行動空間の何番目の行動なのかを取得する
        available_indices = [self.action_space.index(a) for a in available_actions]

        # 選択可能行動の各行動に対応する行動価値を取得する
        available_q = [q_values[i] for i in available_indices]

        # 行動価値が最大の行動を、選択可能行動の中から選択する
        # 行動価値が最大となるような行動は複数存在しうるので、そのような場合は
        # その中からランダムに行動を1つ選択することにする
        greedy_indices = [i for i, q in enumerate(available_q) if q == max(available_q)]
        greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]

        return available_actions[greedy_idx]
