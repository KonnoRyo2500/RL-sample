# 強化学習勉強用サンプルプログラム モンテカルロ法エージェントクラス

from itertools import product

from agent.agent_base import AgentBase
from agent.util.action_selector.greedy import Greedy
from agent.util.action_selector.epsilon_greedy import EpsilonGreedy
from common.const_val import Agent


# モンテカルロ法エージェントクラス
# ここでは方策オン型の初回訪問モンテカルロ法を用いる
class MonteCarloAgent(AgentBase):
    # コンストラクタ
    def __init__(self, env):
        super().__init__(env, Agent.MonteCarlo.value)
        action_space = env.get_action_space()
        state_space = env.get_state_space()

        # 行動価値関数
        self.q_func = self._make_initial_q_function(action_space, state_space)

        # 行動選択インスタンス
        self.greedy = Greedy(action_space)  # greedy
        self.epsilon_greedy = EpsilonGreedy(action_space, self.config['epsilon'])  # ε-greedy

        # Returns(各状態・行動に対する収益を記録しておく配列)
        self.returns = {}
        for s, a in product(state_space, action_space):
            self.returns[(s, a)] = []

        # 1プレイアウト分の経験
        # 経験は、(状態s, 行動a, 報酬r)からなる
        self.experience_of_playout = []

        # 直前の状態とそこで選択した行動
        self.last_state_action = None

    # 次の行動を決定する
    def decide_action(self, env):
        state = env.get_state()
        action_space = env.get_action_space()
        available_actions = env.get_available_actions()
        q_values = [self.q_func[(state, a)] for a in action_space]

        if self.mode == MonteCarloAgent.OperationMode.Train:
            action = self.epsilon_greedy.select_action(available_actions, q_values)
        else:
            action = self.greedy.select_action(available_actions, q_values)

        # Returnsに追加するための経験をfeedback内で作成するので、現状態と選択した行動を記録しておく
        self.last_state_action = (state, action)

        return action

    # 環境からの情報を自身にフィードバックする
    def feedback(self, reward, env):
        # 1ステップ分の経験を作成
        state, action = self.last_state_action
        exp_step = (state, action, reward)
        self.experience_of_playout.append(exp_step)

        if env.is_terminal_state():
            # プレイアウトが完了していれば、1プレイアウト分の経験を用いて行動価値関数を更新
            self._update_returns()
            self._update_q_func()
            self.experience_of_playout.clear()

    # Returnsを更新する
    def _update_returns(self):
        occurred_states_and_actions = []
        for i, exp in enumerate(self.experience_of_playout):
            state, action, reward = exp
            # すでに出現した状態+行動であれば、収益を計算しない
            if (state, action) in occurred_states_and_actions:
                continue

            # 初めて出現する状態+行動のみ、収益を計算してReturns[(s, a)]に追加
            profit = self._calc_profit(self.experience_of_playout[i:])
            self.returns[(state, action)].append(profit)
            occurred_states_and_actions.append((state, action))

    # 与えられた経験履歴の収益を計算する
    # exp_historyは、エピソード全体の経験履歴でなくても良い(部分集合でもよい)
    def _calc_profit(self, exp_history):
        # 収益の計算においては、報酬にしか興味がない
        profit = 0
        contribution_rate = 1  # 寄与率。ステップt(t=0,1,...,T)において、γ^t
        for _, _, reward in exp_history:
            profit += reward * contribution_rate
            contribution_rate *= self.config['gamma']

        return profit

    # 行動価値関数を更新する
    def _update_q_func(self):
        for (s, a) in self.returns.keys():
            profits = self.returns[(s, a)]

            if len(profits) == 0:
                self.q_func[(s, a)] = 0
            else:
                self.q_func[(s, a)] = sum(profits) / len(profits)

    # 行動価値関数を初期化して返す
    def _make_initial_q_function(self, action_space, state_space):
        init_q_func = {}
        for state, action in product(state_space, action_space):
            init_q_func[(state, action)] = 0

        return init_q_func
