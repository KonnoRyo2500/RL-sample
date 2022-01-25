# 強化学習勉強用サンプルプログラム Q学習エージェントクラス

from itertools import product
import random
from pprint import pprint

from common.const_val import *
from agent.agent_base import AgentBase

# Q学習エージェントクラス
class QAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)
        self.env_config = self.env.get_config()
        self.q_func = self._make_initial_q_function()

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        step_count = 0
        step_limit = self.config['step_limit']
        state = self.env.get_pos()
        while state not in self.env_config['goal_pos']:
            # greedy法で行動を選択
            actions = self.env.get_available_direction()
            action = self._select_action_with_greedy(state, actions)
            print(f'状態 {state} で行動 {action} を選択しました。')

            # 行動する
            reward = self.env.move(action)

            # 状態を更新
            state = self.env.get_pos()

            # 無限ループ防止のため、一定回数移動してもゴールしなかったら
            # エピソードを途中で打ち切る
            step_count += 1
            if step_count > step_limit:
                print(f'行動数が上限({step_limit}回)を超えたため、エピソードを終了します。')
                break

        if reward != 0:
            print(f'エピソードをプレイし、報酬 {reward} が得られました。')
        self.env.reset()

    # エージェントを学習させる
    def train(self):
        for i in range(self.config['num_episode']):
            self.episode()

        pprint(self.q_func)
        self.env.reset()

    # エピソードの実行(学習用)
    def episode(self):
        state = self.env.get_pos()
        while state not in self.env_config['goal_pos']:
            self.step()
            state = self.env.get_pos()

        self.env.reset()

    # 1ステップ実行
    def step(self):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γmax(a')Q(s', a') - Q(s, a))

        # 現状態sを得る
        state = self.env.get_pos()

        # 状態sで可能な行動を得る
        actions = self.env.get_available_direction()

        # 行動aを決定する
        action = self._select_action_with_epsilon_greedy(state, actions)

        # 次状態s'と報酬rを得る
        reward = self.env.move(action)
        next_state = self.env.get_pos()

        # max(a')Q(s', a')を得る
        if next_state in self.env_config['goal_pos']:
            next_max_q = 0
        else:
            next_actions = self.env.get_available_direction()
            next_max_q = max([self.q_func[(next_state, a)] for a in next_actions])

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[(state, action)]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + self.config['gamma'] * next_max_q - q
        self.q_func[(state, action)] = q + self.config['alpha'] * td_error

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        width = self.env_config['width']
        height = self.env_config['height']
        for x, y in product(range(width), range(height)):
            for name in Direction._member_names_:
                state = (x, y)
                action = Direction[name]
                init_q_func[(state, action)] = 0

        return init_q_func

    # ε-greedy法で行動を選択する
    def _select_action_with_epsilon_greedy(self, state, actions):
        v = random.uniform(0, 1)
        if v <= self.config['epsilon']:
            # ランダム選択
            random_idx = random.randint(0, len(actions) - 1)
            return actions[random_idx]
        else:
            # greedy法による選択
            return self._select_action_with_greedy(state, actions)

    # greedy法で行動を選択する
    def _select_action_with_greedy(self, state, actions):
        # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
        # ランダムにaを選択することにする
        q_values = [self.q_func[(state, a)] for a in actions]
        greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
        greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
        return actions[greedy_idx]

