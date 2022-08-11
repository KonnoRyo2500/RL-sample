# 強化学習勉強用サンプルプログラム ランダム行動選択クラス

import random

from agent.util.action_selector.action_selector_base import ActionSelectorBase

# ランダム行動選択クラス
class Random(ActionSelectorBase):
    # コンストラクタ(行動空間は利用しない)
    def __init__(self, action_space=None):
        super().__init__(action_space)

    # ランダムに行動を選択する(行動価値は利用しない)
    def select_action(self, available_actions, q_values=None):
        random_idx = random.randint(0, len(available_actions) - 1)
        return available_actions[random_idx]
