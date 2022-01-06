# 強化学習勉強用サンプルプログラム Q学習エージェントクラス

from itertools import product
import random
from pprint import pprint

from common.const_val import *
from agent.agent_base import AgentBase

# Q学習エージェントクラス
class QAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env)
        self.q_func = self._make_initial_q_function()

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        pass

    # 学習の実行
    def train(self):
        for i in range(N_STEP):
            self.step()
            if self.env.get_pos() in GOAL_POS:
                self.env.reset()

        pprint(self.q_func)
        self.env.reset()

    # エピソードの実行
    def episode(self):
        pass

    # 1ステップ実行
    def step(self):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γmax(a')Q(s', a') - Q(s, a))

        # 現状態sを得る
        state = self.env.get_pos()

        # 状態sで可能な行動を得る
        actions = self.env.get_available_direction()

        # 行動aを決定する
        action = self._select_action(state, actions)

        # 次状態s'と報酬rを得る
        reward = self.env.move(action)
        next_state = self.env.get_pos()

        # max(a')Q(s', a')を得る
        if next_state in GOAL_POS:
            next_max_q = 0
        else:
            next_actions = self.env.get_available_direction()
            next_max_q = max([self.q_func[(next_state, a)] for a in next_actions])

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[(state, action)]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + GAMMA * next_max_q - q
        self.q_func[(state, action)] = q + ALPHA * td_error

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        for x, y in product(range(GRID_WIDTH), range(GRID_HEIGHT)):
            for name in Direction._member_names_:
                state = (x, y)
                action = Direction[name]
                init_q_func[(state, action)] = 0

        return init_q_func

    # 行動を選択する
    def _select_action(self, state, actions):
        # ε-greedy法を用いる。
        v = random.uniform(0, 1)
        if v <= EPSILON:
            random_idx = random.randint(0, len(actions) - 1)
            return actions[random_idx]
        else:
            # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
            # ランダムにaを選択することにする
            q_values = [self.q_func[(state, a)] for a in actions]
            greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
            greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
            return actions[greedy_idx]

