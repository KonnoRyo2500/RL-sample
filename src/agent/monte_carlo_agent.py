# 強化学習勉強用サンプルプログラム モンテカルロ法エージェントクラス

import random
from pprint import pprint
from itertools import product

from agent.agent_base import AgentBase

# モンテカルロ法エージェントクラス
class MonteCarloAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env, config):
        super().__init__(env, config)
        self.q_func = self._make_initial_q_function()

    # 学習した価値関数を基にエピソードをプレイ
    def play(self):
        while not self.env.is_terminal_state():
            # 状態を取得(表示用)
            state = self.env.get_state()

            # greedy法で行動を選択
            action = self._select_action_with_greedy()

            # 行動する
            reward = self.env.exec_action(action)
            print(f'状態 {state} で行動 {action} を選択しました。')

        if reward != 0:
            print(f'エピソードをプレイし、報酬 {reward} が得られました。')
        self.env.reset()

    # 学習の実行
    def train(self):
        # モンテカルロ法には複数の亜種が存在するが、
        # ここでは方策オン型の初回訪問モンテカルロ法を採用する。
        s_space = self.env.get_state_space()
        a_space = self.env.get_whole_action_space()

        # Returnsの初期化
        returns = {}
        for state, action in product(s_space, a_space):
            returns[(state, action)] = []

        # プレイアウト実施 + 行動価値関数の更新
        for _ in range(self.config['num_playout']):
            # プレイアウトの実施
            exp_history = self._playout()

            # Returnsの更新
            returns = self._update_returns(exp_history, returns)

            # 行動価値関数の更新
            for (s, a) in returns.keys():
                profits = returns[(s, a)]

                if len(profits) == 0:
                    self.q_func[(s, a)] = 0
                else:
                    self.q_func[(s, a)] = sum(profits) / len(profits)

        # 学習後の行動価値関数を表示する
        pprint(self.q_func)

    # プレイアウトの実施
    def _playout(self):
        step_count = 0
        exp_history = []
        while not self.env.is_terminal_state():
            # 現在の状態sを取得
            state = self.env.get_state()

            # ε-greedy法により、行動aを選択する
            action = self._select_action_with_epsilon_greedy()

            # 行動aを行い、報酬rを得る
            reward = self.env.exec_action(action)
            step_count += 1

            # 経験(s, a, r)を経験履歴に追加
            # 終端状態では行動しないため、終端状態の経験は追加不要
            exp = (state, action, reward)
            exp_history.append(exp)

        # 環境をリセット
        self.env.reset()

        return exp_history

    # ε-greedy法で行動を選択する
    def _select_action_with_epsilon_greedy(self):
        v = random.uniform(0, 1)

        if v <= self.config['epsilon']:
            # ランダム選択
            actions = self.env.get_current_action_space()
            idx = random.randint(0, len(actions) - 1)
            selected_action = actions[idx]
        else:
            # greedy法による選択
            selected_action = self._select_action_with_greedy()

        return selected_action

    # greedy法で行動を選択する
    def _select_action_with_greedy(self):
        # Q(s, a)が最大となるようなaは複数存在しうるので、そのような場合は
        # ランダムにaを選択することにする
        state = self.env.get_state()
        actions = self.env.get_current_action_space()
        q_values = [self.q_func[(state, a)] for a in actions]
        greedy_indices = [i for i, q in enumerate(q_values) if q == max(q_values)]
        greedy_idx = greedy_indices[random.randint(0, len(greedy_indices) - 1)]
        return actions[greedy_idx]

    # Returnsを更新する
    def _update_returns(self, exp_history, returns):
        occurred_states_and_actions = []
        for i, exp in enumerate(exp_history):
            state, action, reward = exp
            # すでに出現した状態+行動であれば、収益を計算しない
            if (state, action) in occurred_states_and_actions:
                continue

            # 初めて出現する状態+行動のみ、収益を計算してReturns[(s, a)]に追加
            profit = self._calc_profit(exp_history[i:])
            returns[(state, action)].append(profit)
            occurred_states_and_actions.append((state, action))

        return returns

    # 与えられた経験履歴の収益を計算する
    # exp_historyは、エピソード全体の経験履歴でなくても良い(部分集合でもよい)
    def _calc_profit(self, exp_history):
        # 収益の計算においては、報酬にしか興味がない
        profit = 0
        contribution_rate = 1 # 寄与率。ステップt(t=0,1,...,T)において、γ^t
        for _, _, reward in exp_history:
            profit += reward * contribution_rate
            contribution_rate *= self.config['gamma']

        return profit

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self):
        init_q_func = {}
        s_space = self.env.get_state_space()
        a_space = self.env.get_whole_action_space()
        for state, action in product(s_space, a_space):
            init_q_func[(state, action)] = 0

        return init_q_func
