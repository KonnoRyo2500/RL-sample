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
        self.q_func = self._make_initial_q_function()

    # 学習済みエージェントにエピソードをプレイさせる
    def play(self):
        step_count = 0
        step_limit = self.config['step_limit']
        while not self.env.is_terminal_state():
            # 状態を取得(表示用)
            state = self.env.get_state()

            # greedy法で行動を選択
            action = self._select_action_with_greedy()
            print(f'状態 {state} で行動 {action} を選択しました。')

            # 行動する
            reward = self.env.exec_action(action)

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
        state = self.env.get_state()
        while not self.env.is_terminal_state():
            self.step()
            state = self.env.get_state()

        self.env.reset()

    # 1ステップ実行
    def step(self):
        # 更新式: Q(s, a) = Q(s, a) + α(r + γmax(a')Q(s', a') - Q(s, a))

        # 現状態sを取得
        state = self.env.get_state()

        # 行動aを決定する
        action = self._select_action_with_epsilon_greedy()

        # 次状態s'と報酬rを得る
        reward = self.env.exec_action(action)
        next_state = self.env.get_state()

        # max(a')Q(s', a')を得る
        if self.env.is_terminal_state():
            next_max_q = 0
        else:
            next_actions = self.env.get_actions()
            next_max_q = max([self.q_func[(next_state, a)] for a in next_actions])

        # Q(s, a)はそのまま記述すると表記が長いため、短い名前の変数に入れておく
        q = self.q_func[(state, action)]

        # 行動価値関数Q(s, a)を更新する
        td_error = reward + self.config['gamma'] * next_max_q - q
        self.q_func[(state, action)] = q + self.config['alpha'] * td_error

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        for state, action in product(self.env.get_all_states(), self.env.get_actions(require_all=True)):
            init_q_func[(state, action)] = 0

        return init_q_func

    # ε-greedy法で行動を選択する
    def _select_action_with_epsilon_greedy(self):
        v = random.uniform(0, 1)
        actions = self.env.get_actions()
        if v <= self.config['epsilon']:
            # ランダム選択
            random_idx = random.randint(0, len(actions) - 1)
            return actions[random_idx]
        else:
            # greedy法による選択
            return self._select_action_with_greedy()

    # greedy法で行動を選択する
    def _select_action_with_greedy(self):
        # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
        # ランダムにaを選択することにする
        state = self.env.get_state()
        actions = self.env.get_actions()
        q_values = [self.q_func[(state, a)] for a in actions]
        greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
        greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
        return actions[greedy_idx]

