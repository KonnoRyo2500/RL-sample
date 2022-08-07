# 強化学習勉強用サンプルプログラム ε-greedy行動選択クラス

import random

from agent.util.action_selector.action_selector_base import ActionSelectorBase
from agent.util.action_selector.greedy import Greedy
from agent.util.action_selector.random import Random

# ε-greedy行動選択クラス
class EpsilonGreedy(ActionSelectorBase):
    # コンストラクタ
    def __init__(self, action_space, epsilon_init, epsilon_diff=0, epsilon_min=0):
        super().__init__(action_space)

        self.random = Random(action_space) # ランダム行動選択
        self.greedy = Greedy(action_space) # greedy行動選択

        # εは、エージェントが指定したパラメータに従って線形に減少するものとする
        self.epsilon_diff = epsilon_diff # εの公差
        self.epsilon_min = epsilon_min # εの最小値
        self.epsilon = epsilon_init # 現在のεの値

    # 行動価値に基づき、行動を選択する
    def select_action(self, q_values):
        v = random.uniform(0, 1)
        if v <= self.epsilon:
            # ランダム選択(探索)
            action = self.random.select_action()
        else:
            # greedy選択(利用)
            action = self.greedy.select_action(q_values)

        return action

    # εを公差の分だけ減少させる
    def decrement_epsilon(self):
        next_epsilon = self.epsilon - self.epsilon_diff
        if next_epsilon < self.epsilon_min:
            next_epsilon = self.epsilon_min

        self.epsilon = next_epsilon
