# 強化学習勉強用サンプルプログラム ランダム行動選択クラス

import random

from agent.util.action_selector.action_selector_base import ActionSelectorBase

# ランダム行動選択クラス
class Random(ActionSelectorBase):
    # コンストラクタ
    def __init__(self, action_space):
        super().__init__(action_space)

    # ランダムに行動を選択する(行動価値は利用しない)
    def select_action(self, q_values=None):
        random_idx = random.randint(0, len(self.action_space) - 1)
        return self.action_space[random_idx]
