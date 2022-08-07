# 強化学習勉強用サンプルプログラム greedy行動選択クラス

import random

from agent.util.action_selector.action_selector_base import ActionSelectorBase

# greedy行動選択クラス
class Greedy(ActionSelectorBase):
    # コンストラクタ
    def __init__(self, action_space):
        super().__init__(action_space)

    # 行動価値に基づき、行動を選択する
    def select_action(self, q_values):
        # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
        # ランダムにaを選択することにする
        greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
        greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
        return self.action_space[greedy_idx]
